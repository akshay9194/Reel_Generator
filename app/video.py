"""
video.py — Stitch generated frames into an MP4 video.

Uses OpenCV for frame assembly with optional hold-frame duplication
to control pacing, then produces an mp4v-encoded video.
"""

import cv2
from pathlib import Path

from config import OUTPUT_DIR, DEFAULT_FPS, DEFAULT_HOLD_FRAMES


def create_video(
    image_paths: list[str],
    *,
    output_path: str | None = None,
    fps: int = DEFAULT_FPS,
    hold_frames: int = DEFAULT_HOLD_FRAMES,
) -> str:
    """
    Stitch images into a video.

    Each image is repeated `hold_frames` times so a 6-image reel
    with hold_frames=8 at fps=5 lasts ~10 seconds.

    Returns the path to the output .mp4 file.
    """
    if output_path is None:
        output_path = str(OUTPUT_DIR / "output.mp4")

    if not image_paths:
        raise ValueError("No images provided to create_video")

    first = cv2.imread(image_paths[0])
    if first is None:
        raise FileNotFoundError(f"Cannot read image: {image_paths[0]}")
    height, width, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                print(f"[video] WARNING: skipping unreadable frame {path}")
                continue
            # Resize if dimensions mismatch the first frame
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height))
            for _ in range(hold_frames):
                writer.write(frame)
    finally:
        writer.release()

    print(f"[video] Saved → {output_path}  ({len(image_paths)} frames, {fps} fps)")
    return output_path
