# CUDA runtime (not "devel") - we only run inference here, never compile
# anything, so the full CUDA toolkit isn't needed. This keeps the image and
# its layers much smaller than RunPod's devel-based image, which had single
# layers close to 3GB that kept failing to upload on this network.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# LatentSync (human lip-sync, Apache-2.0) lives in its OWN isolated venv,
# not the main one above. It's a standalone research repo with its own
# pinned torch/diffusers/xformers versions that are very likely to conflict
# with the carefully-pinned Wan2.2 stack (see git history for how fragile
# that pinning was to get right) - isolating it avoids repeating that fight.
# handler.py shells out to this venv's own python interpreter as a
# subprocess rather than importing it in-process.
RUN python -m venv /opt/latentsync-venv \
    && git clone --depth 1 https://github.com/bytedance/LatentSync.git /opt/LatentSync \
    && /opt/latentsync-venv/bin/pip install --no-cache-dir -r /opt/LatentSync/requirements.txt

# Weights are NOT baked into the image (that made the image ~90GB and the
# local build unreliable). handler.py's from_pretrained() calls download
# them into the container's own storage on first cold start instead - slower
# first request per fresh worker, but a small, fast, reliable image build.
# If cold-start latency becomes a problem, revisit with a RunPod Network
# Volume (persistent weights mounted across workers) rather than baking
# weights back into the image. LatentSync's checkpoints download the same
# way, on first lip-sync request, via huggingface_hub inside handler.py.

CMD ["python", "-u", "handler.py"]
