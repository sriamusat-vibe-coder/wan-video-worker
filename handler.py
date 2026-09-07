"""
RunPod Serverless handler - text-to-video and image-to-video generation using
Wan2.2 (Alibaba's open video generation model), running on your own GPU
worker.

This is the code that runs INSIDE the RunPod serverless container, on your
GPU pod. RunPod's runtime invokes handler() once per job. Do not run this
file on your local machine - it needs an actual CUDA GPU and the downloaded
model weights, which only exist inside the built container.

Model choice: Wan2.2-TI2V-5B - a single unified dense 5B-parameter model
that does BOTH text-to-video and image-to-video from one set of weights.
Chosen because it comfortably fits a 48GB GPU without the CPU-offload
slowdowns the larger Wan2.2-A14B MoE variants need (those are ~27B combined
params, officially recommended at 80GB+). If you move to a bigger GPU later
and want higher quality, switch MODEL_ID_T2V/MODEL_ID_I2V below to
"Wan-AI/Wan2.2-T2V-A14B-Diffusers" / "Wan-AI/Wan2.2-I2V-A14B-Diffusers".

Audio: Stable Audio Open (stabilityai/stable-audio-open-1.0) generates an
ambient sound-effects/atmosphere track matched to the video's duration, muxed
in with ffmpeg. This is scene-level ambiance, not frame-synced foley or
lip-synced dialogue - a real limitation of current open audio models, not a
bug here. NOTE: Stable Audio Open ships under Stability AI's own community
license (free under a revenue threshold, paid enterprise license above it) -
this is different from Wan2.2's permissive Apache-2.0 license, so check
https://huggingface.co/stabilityai/stable-audio-open-1.0 for current terms
before using this at commercial scale.

Verified: text_to_video has been run end-to-end against a live 48GB GPU
successfully (see ../README.md "Known-good vs. not yet verified" for exactly
what has and hasn't been confirmed - notably, the audio path added later has
NOT yet been live-tested, and image_to_video mode hasn't either).
"""

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
import runpod
import soundfile as sf
import torch
from diffusers import AutoencoderKLWan, StableAudioPipeline, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID_T2V = os.environ.get("WAN_MODEL_T2V", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
MODEL_ID_I2V = os.environ.get("WAN_MODEL_I2V", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
MODEL_ID_AUDIO = os.environ.get("WAN_MODEL_AUDIO", "stabilityai/stable-audio-open-1.0")
DTYPE = torch.bfloat16

# Cached per-worker-process - a warm worker (RunPod keeps one alive briefly
# between jobs) reuses this, so only a cold start pays the weight-load cost.
_pipelines = {}
_audio_pipeline = None


def _get_pipeline(mode: str):
    if mode in _pipelines:
        return _pipelines[mode]

    if mode == "text_to_video":
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID_T2V, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(MODEL_ID_T2V, vae=vae, torch_dtype=DTYPE)
    elif mode == "image_to_video":
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID_I2V, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID_I2V, vae=vae, torch_dtype=DTYPE)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected text_to_video or image_to_video)")

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()  # safety margin on 48GB, small speed cost
    _pipelines[mode] = pipe
    return pipe


def _get_audio_pipeline():
    global _audio_pipeline
    if _audio_pipeline is None:
        pipe = StableAudioPipeline.from_pretrained(MODEL_ID_AUDIO, torch_dtype=DTYPE)
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        _audio_pipeline = pipe
    return _audio_pipeline


def _generate_audio(prompt: str, duration_s: float, generator, tmp_dir: str) -> str:
    """
    Generates an ambient sound/effects track matching the scene description.
    This is NOT frame-synced foley - Stable Audio Open produces general
    atmosphere/sound-effects audio for the described scene, not audio tied to
    specific visual moments or lip-synced dialogue.
    """
    pipe = _get_audio_pipeline()
    audio = pipe(
        prompt=prompt,
        negative_prompt="music, singing, speech, silence, low quality",
        num_inference_steps=100,
        audio_end_in_s=duration_s,
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios
    output = audio[0].T.float().cpu().numpy()
    audio_path = str(Path(tmp_dir) / "audio.wav")
    sf.write(audio_path, output, pipe.vae.sampling_rate)
    return audio_path


def _mux_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def _video_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _save_input_image(image_b64: str, tmp_dir: str) -> str:
    image_path = str(Path(tmp_dir) / "input_image.png")
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    return image_path


def handler(job):
    """
    job["input"] shape:
      {
        "mode": "text_to_video" | "image_to_video",
        "prompt": "...",
        "negative_prompt": "...",          (optional)
        "image_base64": "...",             (required if mode == image_to_video)
        "width": 1280, "height": 704,      (must stay divisible by 16)
        "num_frames": 121,                 (~5s at fps=24)
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 40,
        "seed": null,
        "generate_audio": true,            (optional, default true)
        "sound_prompt": "...",             (optional - defaults to a
                                             sound-effects framing of `prompt`)
      }
    Returns: {"video_base64": "...", "seconds": <generation wall time>}
    or       {"error": "..."} on a handled failure.

    Audio note: generate_audio adds an ambient sound-effects/atmosphere track
    (via Stable Audio Open) matched to the video's duration - it is NOT
    frame-synced foley or lip-synced dialogue, just general scene sound.
    """
    inp = job["input"]
    mode = inp.get("mode", "text_to_video")
    prompt = inp["prompt"]

    if mode == "image_to_video" and "image_base64" not in inp:
        return {"error": "image_to_video mode requires 'image_base64' in input."}

    generator = torch.Generator(device="cuda")
    if inp.get("seed") is not None:
        generator = generator.manual_seed(inp["seed"])

    try:
        pipe = _get_pipeline(mode)
    except Exception as e:
        return {"error": f"Failed to load model pipeline: {e}"}

    start = time.time()
    fps = inp.get("fps", 24)
    num_frames = inp.get("num_frames", 121)
    common_kwargs = dict(
        prompt=prompt,
        negative_prompt=inp.get("negative_prompt", ""),
        height=inp.get("height", 704),
        width=inp.get("width", 1280),
        num_frames=num_frames,
        guidance_scale=inp.get("guidance_scale", 5.0),
        num_inference_steps=inp.get("num_inference_steps", 40),
        generator=generator,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if mode == "image_to_video":
                image_path = _save_input_image(inp["image_base64"], tmp_dir)
                frames = pipe(image=load_image(image_path), **common_kwargs).frames[0]
            else:
                frames = pipe(**common_kwargs).frames[0]

            video_path = str(Path(tmp_dir) / "video.mp4")
            export_to_video(frames, video_path, fps=fps)

            final_path = video_path
            if inp.get("generate_audio", True):
                sound_prompt = inp.get(
                    "sound_prompt", f"Ambient sound effects and atmosphere for: {prompt}"
                )
                duration_s = num_frames / fps
                audio_path = _generate_audio(sound_prompt, duration_s, generator, tmp_dir)
                final_path = str(Path(tmp_dir) / "output.mp4")
                _mux_audio_video(video_path, audio_path, final_path)

            elapsed = round(time.time() - start, 1)
            return {"video_base64": _video_to_base64(final_path), "seconds": elapsed}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
