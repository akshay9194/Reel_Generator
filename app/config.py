import os
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model IDs ───────────────────────────────────────────
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.safetensors"

# ─── Generation defaults ─────────────────────────────────
DEFAULT_NUM_FRAMES = 6
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_STEPS = 30
DEFAULT_IP_ADAPTER_SCALE = 0.6
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NEGATIVE_PROMPT = (
    "deformed, ugly, bad anatomy, disfigured, poorly drawn face, "
    "mutation, mutated, extra limbs, blurry, watermark, text, "
    "low quality, worst quality"
)

# ─── Video defaults ──────────────────────────────────────
DEFAULT_FPS = 5
DEFAULT_HOLD_FRAMES = 8  # how many duplicated frames per image (for pacing)

# ─── Device ──────────────────────────────────────────────
DEVICE = "cuda" if os.environ.get("FORCE_CPU") is None else "cpu"
DTYPE_STR = os.environ.get("DTYPE", "float16")  # float16 | bfloat16
