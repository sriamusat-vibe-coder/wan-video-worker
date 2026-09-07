# CUDA runtime (not "devel") - we only run inference here, never compile
# anything, so the full CUDA toolkit isn't needed. This keeps the image and
# its layers much smaller than RunPod's devel-based image, which had single
# layers close to 3GB that kept failing to upload on this network.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Weights are NOT baked into the image (that made the image ~90GB and the
# local build unreliable). handler.py's from_pretrained() calls download
# them into the container's own storage on first cold start instead - slower
# first request per fresh worker, but a small, fast, reliable image build.
# If cold-start latency becomes a problem, revisit with a RunPod Network
# Volume (persistent weights mounted across workers) rather than baking
# weights back into the image.

CMD ["python", "-u", "handler.py"]
