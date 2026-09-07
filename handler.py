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

IMPORTANT - verify before your first Docker build:
Wan2.2's diffusers support is recent and moves fast. Before building, check
https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers for the exact
`diffusers` version required and confirm the pipeline class names below
(WanPipeline / WanImageToVideoPipeline) still match what the model card
shows. This has not been run against a live GPU yet - treat the first
deploy as a smoke test, not a known-good build.
"""

import base64
import os
import tempfile
import time
from pathlib import Path

import runpod
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID_T2V = os.environ.get("WAN_MODEL_T2V", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
MODEL_ID_I2V = os.environ.get("WAN_MODEL_I2V", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DTYPE = torch.bfloat16

# Cached per-worker-process - a warm worker (RunPod keeps one alive briefly
# between jobs) reuses this, so only a cold start pays the weight-load cost.
_pipelines = {}


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
        "seed": null
      }
    Returns: {"video_base64": "...", "seconds": <generation wall time>}
    or       {"error": "..."} on a handled failure.
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
    common_kwargs = dict(
        prompt=prompt,
        negative_prompt=inp.get("negative_prompt", ""),
        height=inp.get("height", 704),
        width=inp.get("width", 1280),
        num_frames=inp.get("num_frames", 121),
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

            video_path = str(Path(tmp_dir) / "output.mp4")
            export_to_video(frames, video_path, fps=inp.get("fps", 24))

            elapsed = round(time.time() - start, 1)
            return {"video_base64": _video_to_base64(video_path), "seconds": elapsed}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


runpod.serverless.start({"handler": handler})
