# ─── Base image ──────────────────────────────────────────
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# ─── System deps ─────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ─── Python deps ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Pre-download models at build time (saves cold-start) ─
RUN python -c "\
from diffusers import StableDiffusionXLPipeline; \
StableDiffusionXLPipeline.from_pretrained( \
    'stabilityai/stable-diffusion-xl-base-1.0', \
    torch_dtype='auto', variant='fp16'); \
print('SDXL downloaded')"

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('h94/IP-Adapter', \
    filename='ip-adapter_sdxl.safetensors', \
    subfolder='sdxl_models'); \
print('IP-Adapter downloaded')"

# ─── Pre-setup SadTalker (talking-head + lip-sync) ───────
RUN git clone --depth 1 https://github.com/OpenTalker/SadTalker.git /app/SadTalker && \
    cd /app/SadTalker && \
    pip install --no-cache-dir -r requirements.txt || true

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='vinthony/SadTalker-V002', \
    local_dir='/app/SadTalker/checkpoints', \
    local_dir_use_symlinks=False); \
print('SadTalker checkpoints downloaded')"

# ─── Copy app code ───────────────────────────────────────
COPY app/ .

# ─── Expose Gradio UI port ───────────────────────────────
EXPOSE 7860

# ─── Default: RunPod serverless handler ──────────────────
# To run the Gradio UI instead:  CMD ["python", "ui.py"]
CMD ["python", "handler.py"]
