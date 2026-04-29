"""
test_local.py — Test the pipeline locally WITHOUT a GPU.

Creates dummy coloured frames with the face image composited on top,
then stitches them into a video. Validates the full pipeline wiring
without loading any ML models.

Usage:
    cd app
    python test_local.py                       # uses a generated placeholder face
    python test_local.py --face face.jpg       # uses your actual face image
"""

import argparse
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ensure app/ imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, DEFAULT_FPS, DEFAULT_HOLD_FRAMES


def _create_test_face(path: str):
    """Generate a simple placeholder face image."""
    img = Image.new("RGB", (512, 512), (200, 180, 160))
    draw = ImageDraw.Draw(img)
    # crude face
    draw.ellipse([156, 106, 356, 406], fill=(220, 200, 180))  # head
    draw.ellipse([200, 200, 240, 230], fill=(60, 40, 30))     # left eye
    draw.ellipse([270, 200, 310, 230], fill=(60, 40, 30))     # right eye
    draw.arc([220, 280, 290, 330], 0, 180, fill=(150, 80, 80), width=3)  # mouth
    img.save(path)
    print(f"[test] Created placeholder face → {path}")


def _generate_mock_frames(prompt: str, face_path: str, num_frames: int) -> list[str]:
    """Generate coloured test frames with text overlay."""
    face = Image.open(face_path).convert("RGB").resize((200, 200))

    colors = [
        (30, 60, 90), (90, 30, 60), (60, 90, 30),
        (90, 60, 30), (30, 90, 60), (60, 30, 90),
        (50, 50, 80), (80, 50, 50), (50, 80, 50),
        (70, 40, 60), (40, 70, 60), (60, 40, 70),
        (45, 75, 45), (75, 45, 75), (45, 45, 75),
        (80, 80, 30),
    ]

    variations = [
        "looking straight", "looking left", "looking right",
        "soft smile", "candid laugh", "serious gaze",
        "profile view", "over shoulder",
    ]

    paths = []
    for i in range(num_frames):
        bg = Image.new("RGB", (1024, 1024), colors[i % len(colors)])
        draw = ImageDraw.Draw(bg)

        # paste face thumbnail in center
        x, y = 412, 200
        bg.paste(face, (x, y))

        # text overlay
        variation = variations[i % len(variations)]
        draw.text((100, 50), f"Frame {i + 1}/{num_frames}", fill="white")
        draw.text((100, 90), f"Pose: {variation}", fill="white")
        draw.text((100, 850), prompt[:80], fill=(180, 180, 180))
        draw.text((100, 900), "[TEST MODE — no ML model loaded]", fill=(255, 200, 100))

        path = str(OUTPUT_DIR / f"frame_{i:03d}.png")
        bg.save(path)
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Local test — no GPU needed")
    parser.add_argument("--face", default=None, help="Face image path (optional)")
    parser.add_argument("--prompt", default="test prompt — girl in cafe, cinematic lighting")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--hold", type=int, default=DEFAULT_HOLD_FRAMES)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Face image
    if args.face and Path(args.face).is_file():
        face_path = args.face
    else:
        face_path = str(OUTPUT_DIR / "_test_face.png")
        _create_test_face(face_path)

    t0 = time.time()

    print("━" * 50)
    print("🧪  Reel Generator — LOCAL TEST (no GPU)")
    print("━" * 50)
    print(f"  Frames : {args.frames}")
    print(f"  FPS    : {args.fps}")
    print(f"  Hold   : {args.hold}")
    total_frames = args.frames * args.hold
    duration = total_frames / args.fps
    print(f"  Video  : {total_frames} total frames → {duration:.1f}s")
    print("━" * 50)

    # Step 1 — mock frames
    print("\n[1/2] Generating mock frames …")
    image_paths = _generate_mock_frames(args.prompt, face_path, args.frames)
    print(f"       → {len(image_paths)} images saved")

    # Step 2 — stitch (uses the real video.py)
    from video import create_video

    print("\n[2/2] Stitching video …")
    video_path = create_video(image_paths, fps=args.fps, hold_frames=args.hold)

    elapsed = time.time() - t0
    print(f"\n✅  Test passed in {elapsed:.1f}s")
    print(f"    Video → {video_path}")
    print(f"    Duration: {duration:.1f}s ({total_frames} frames @ {args.fps} fps)")
    print(f"\n    Open the video to verify it plays correctly.")


if __name__ == "__main__":
    main()
