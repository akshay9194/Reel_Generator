"""
lipsync.py — Talking-head video generation using SadTalker.

Unlike Wav2Lip (which only modifies lips on an existing video),
SadTalker generates a FULL talking-head video from:
  - A single face image
  - A voice/speech audio clip

It produces natural head movement, eye blinks, and accurate lip sync —
exactly what's needed for AI reels.

SadTalker paper: https://arxiv.org/abs/2211.12194
GitHub: https://github.com/OpenTalker/SadTalker
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path

from config import OUTPUT_DIR, BASE_DIR

SADTALKER_DIR = BASE_DIR / "SadTalker"
CHECKPOINTS_DIR = SADTALKER_DIR / "checkpoints"

# HuggingFace model repo for SadTalker weights
HF_REPO = "vinthony/SadTalker-V002"


def setup_sadtalker() -> bool:
    """
    One-time setup: clone SadTalker repo, install deps, download checkpoints.
    Returns True when ready, raises on failure.
    """
    # 1. Clone repo
    if not (SADTALKER_DIR / "inference.py").exists():
        print("[lipsync] Cloning SadTalker repository …")
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/OpenTalker/SadTalker.git",
                str(SADTALKER_DIR),
            ],
            check=True,
        )

    # 2. Install SadTalker Python deps
    req_file = SADTALKER_DIR / "requirements.txt"
    if req_file.exists():
        print("[lipsync] Installing SadTalker dependencies …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            check=True,
        )

    # 3. Download checkpoints
    _download_checkpoints()

    print("[lipsync] SadTalker is ready.")
    return True


def _download_checkpoints():
    """Download SadTalker model weights from HuggingFace."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for key files)
    marker_files = [
        CHECKPOINTS_DIR / "SadTalker_V0.0.2_256.safetensors",
        CHECKPOINTS_DIR / "mapping_00109-model.pth.tar",
    ]

    if all(f.exists() for f in marker_files):
        print("[lipsync] Checkpoints already present.")
        return

    print("[lipsync] Downloading SadTalker checkpoints …")

    try:
        from huggingface_hub import snapshot_download

        # Download the full checkpoint set
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(CHECKPOINTS_DIR),
            local_dir_use_symlinks=False,
        )
        print("[lipsync] Checkpoints downloaded successfully.")

    except Exception as e:
        print(f"[lipsync] HuggingFace download failed: {e}")
        print("[lipsync] Trying SadTalker's built-in download script …")

        download_script = SADTALKER_DIR / "scripts" / "download_models.sh"
        if download_script.exists():
            subprocess.run(
                ["bash", str(download_script)],
                cwd=str(SADTALKER_DIR),
                check=True,
            )
        else:
            raise RuntimeError(
                "Could not download SadTalker checkpoints. Please run:\n"
                f"  cd {SADTALKER_DIR}\n"
                "  bash scripts/download_models.sh"
            ) from e


def is_sadtalker_ready() -> bool:
    """Check if SadTalker is set up without modifying anything."""
    return (
        (SADTALKER_DIR / "inference.py").exists()
        and CHECKPOINTS_DIR.exists()
        and any(CHECKPOINTS_DIR.iterdir()) if CHECKPOINTS_DIR.exists() else False
    )


def run_lipsync(
    face_image_path: str,
    audio_path: str,
    *,
    output_path: str | None = None,
    still_mode: bool = False,
    pose_style: int = 0,
    expression_scale: float = 1.0,
    preprocess: str = "crop",
    size: int = 256,
) -> str:
    """
    Generate a talking-head video from a face image + audio.

    Args:
        face_image_path: Path to a face image (the source identity).
        audio_path: Path to voice audio (WAV/MP3) to drive the animation.
        output_path: Where to save the final video. Auto-generated if None.
        still_mode: If True, reduce head motion (useful for more stable output).
        pose_style: Head pose variation style (0–45). 0 = natural.
        expression_scale: How exaggerated expressions are (1.0 = normal).
        preprocess: Face preprocessing — "crop" (default), "resize", "full", "extcrop", "extfull".
        size: Resolution — 256 (fast) or 512 (better quality, slower).

    Returns: Path to the generated talking-head video.
    """
    if not is_sadtalker_ready():
        print("[lipsync] First run — setting up SadTalker …")
        setup_sadtalker()

    # SadTalker outputs to a directory; we'll move the result after
    result_dir = str(OUTPUT_DIR / "sadtalker_results")
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        sys.executable,
        str(SADTALKER_DIR / "inference.py"),
        "--driven_audio", audio_path,
        "--source_image", face_image_path,
        "--result_dir", result_dir,
        "--checkpoint_dir", str(CHECKPOINTS_DIR),
        "--preprocess", preprocess,
        "--pose_style", str(pose_style),
        "--expression_scale", str(expression_scale),
        "--size", str(size),
    ]

    if still_mode:
        cmd.append("--still")

    print("[lipsync] Running SadTalker …")
    print(f"          Face : {face_image_path}")
    print(f"          Audio: {audio_path}")
    print(f"          Mode : {'still' if still_mode else 'animated'}, size={size}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(SADTALKER_DIR),
    )

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        raise RuntimeError(
            f"SadTalker inference failed (exit code {result.returncode}):\n{error_msg}"
        )

    # Find the generated video (SadTalker names it based on inputs)
    result_path = Path(result_dir)
    generated_videos = sorted(result_path.glob("*.mp4"), key=os.path.getmtime, reverse=True)

    if not generated_videos:
        raise FileNotFoundError(
            f"SadTalker did not produce any output in {result_dir}.\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )

    latest_video = generated_videos[0]

    # Move to final output path
    if output_path is None:
        output_path = str(OUTPUT_DIR / "output_lipsync.mp4")

    shutil.move(str(latest_video), output_path)

    print(f"[lipsync] Talking-head video → {output_path}")
    return output_path
