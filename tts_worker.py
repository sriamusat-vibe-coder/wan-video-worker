"""
Standalone Indic Parler-TTS generation script - runs in its OWN isolated
venv (/opt/tts-venv, set up in the Dockerfile), invoked as a subprocess from
handler.py. Isolated because parler-tts pins transformers==4.46.1 exactly,
which conflicts with the newer transformers version (>=4.50) the main
Wan2.2/diffusers venv needs for unrelated reasons (see handler.py comments).

Not cached/reused across calls the way the main process's models are - each
call pays a fresh model-load cost since this runs as a new subprocess every
time. A real inefficiency worth revisiting (e.g. a small persistent server
process) if TTS call volume ever makes it matter; left simple for now.
"""

import argparse

import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = "ai4bharat/indic-parler-tts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice-description", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    description_ids = description_tokenizer(args.voice_description, return_tensors="pt").to(device)
    prompt_ids = tokenizer(args.text, return_tensors="pt").to(device)

    generation = model.generate(
        input_ids=description_ids.input_ids,
        attention_mask=description_ids.attention_mask,
        prompt_input_ids=prompt_ids.input_ids,
        prompt_attention_mask=prompt_ids.attention_mask,
    )
    audio_arr = generation.cpu().float().numpy().squeeze()
    sf.write(args.output, audio_arr, model.config.sampling_rate)


if __name__ == "__main__":
    main()
