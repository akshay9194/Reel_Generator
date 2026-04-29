"""
generate.py — Image generation with SDXL + IP-Adapter for face consistency.

Loads the pipeline once, then generates N frames that share the same face
identity by conditioning on a reference face image via IP-Adapter.
"""

import gc
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from pathlib import Path

from config import (
    SDXL_MODEL_ID,
    IP_ADAPTER_REPO,
    IP_ADAPTER_SUBFOLDER,
    IP_ADAPTER_WEIGHT,
    OUTPUT_DIR,
    DEFAULT_NUM_FRAMES,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEPS,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEVICE,
    DTYPE_STR,
)

# ── resolve torch dtype ──────────────────────────────────
_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
TORCH_DTYPE = _DTYPE_MAP.get(DTYPE_STR, torch.float16)

# ── singleton pipeline ───────────────────────────────────
_pipe = None


def _get_pipe():
    """Lazy-load and cache the SDXL + IP-Adapter pipeline."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"[generate] Loading SDXL from {SDXL_MODEL_ID} …")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to(DEVICE)

    # Enable memory-efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[generate] xformers enabled")
    except Exception:
        print("[generate] xformers not available, using default attention")

    # Load IP-Adapter for face conditioning
    print("[generate] Loading IP-Adapter …")
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT,
    )
    pipe.set_ip_adapter_scale(DEFAULT_IP_ADAPTER_SCALE)
    print("[generate] Pipeline ready.")

    _pipe = pipe
    return _pipe


def unload_pipeline():
    """Free VRAM explicitly (useful between RunPod calls)."""
    global _pipe
    if _pipe is not None:
        _pipe.to("cpu")
        del _pipe
        _pipe = None
        gc.collect()
        torch.cuda.empty_cache()


def generate_images(
    prompt: str,
    face_path: str,
    *,
    num_images: int = DEFAULT_NUM_FRAMES,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int | None = None,
) -> list[str]:
    """
    Generate `num_images` frames conditioned on the face in `face_path`.

    Returns a list of saved PNG paths.
    """
    pipe = _get_pipe()
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    face_image = Image.open(face_path).convert("RGB")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Pose / scene variations baked into the prompt per frame
    variations = [
        "looking straight at camera",
        "looking slightly to the left",
        "looking slightly to the right",
        "soft smile, slightly tilted head",
        "candid laugh, natural expression",
        "serious expression, direct gaze",
        "profile view, dramatic lighting",
        "over the shoulder look",
    ]

    image_paths: list[str] = []

    for i in range(num_images):
        variation = variations[i % len(variations)]
        frame_prompt = f"{prompt}, {variation}, same person, ultra realistic, 8k"

        print(f"[generate] Frame {i + 1}/{num_images} — {variation}")

        result = pipe(
            prompt=frame_prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=face_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        )

        image = result.images[0]
        path = str(OUTPUT_DIR / f"frame_{i:03d}.png")
        image.save(path)
        image_paths.append(path)

    return image_paths
