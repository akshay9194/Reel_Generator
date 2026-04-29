"""
audio.py — Merge background audio onto the generated video.

Supports:
  - Background music (looped or trimmed to match video duration)
  - Voice-over audio (for lip-sync pipeline)
  - Volume mixing (music quieter when voice is present)

Requires: ffmpeg installed on the system (included in Dockerfile).
"""

import subprocess
import shutil
from pathlib import Path

from config import OUTPUT_DIR


def _check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install it: apt install ffmpeg (Linux) "
            "or download from https://ffmpeg.org"
        )


def get_video_duration(video_path: str) -> float:
    """Get duration of a video in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def add_background_music(
    video_path: str,
    audio_path: str,
    *,
    output_path: str | None = None,
    music_volume: float = 0.3,
    loop_audio: bool = True,
    fade_out_seconds: float = 2.0,
) -> str:
    """
    Merge background music onto a video.

    Args:
        video_path: Input video (mp4).
        audio_path: Music file (mp3/wav/m4a).
        output_path: Where to save. Defaults to outputs/output_with_audio.mp4.
        music_volume: Music volume 0.0–1.0 (0.3 = 30% — sits behind visuals).
        loop_audio: Loop the music if it's shorter than the video.
        fade_out_seconds: Fade out music at the end.

    Returns: Path to the output video.
    """
    _check_ffmpeg()

    if output_path is None:
        output_path = str(OUTPUT_DIR / "output_with_audio.mp4")

    duration = get_video_duration(video_path)

    # Build audio filter
    audio_filter = f"volume={music_volume}"
    if fade_out_seconds > 0:
        fade_start = max(0, duration - fade_out_seconds)
        audio_filter += f",afade=t=out:st={fade_start:.2f}:d={fade_out_seconds:.2f}"

    cmd = ["ffmpeg", "-y"]

    # Input: video
    cmd += ["-i", video_path]

    # Input: audio (with optional looping)
    if loop_audio:
        cmd += ["-stream_loop", "-1", "-i", audio_path]
    else:
        cmd += ["-i", audio_path]

    # Map video from first input, build audio filter
    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-filter:a", audio_filter,
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]

    print(f"[audio] Adding background music (vol={music_volume}) …")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[audio] Saved → {output_path}")
    return output_path


def mix_voice_and_music(
    video_path: str,
    voice_path: str,
    music_path: str | None = None,
    *,
    output_path: str | None = None,
    voice_volume: float = 1.0,
    music_volume: float = 0.15,
) -> str:
    """
    Mix voice-over (and optional background music) onto a video.

    Used after lip-sync: the voice is primary, music is quiet underneath.

    Returns: Path to the output video.
    """
    _check_ffmpeg()

    if output_path is None:
        output_path = str(OUTPUT_DIR / "output_final.mp4")

    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", voice_path]

    if music_path:
        cmd += ["-stream_loop", "-1", "-i", music_path]
        # Mix voice + music
        filter_str = (
            f"[1:a]volume={voice_volume}[voice];"
            f"[2:a]volume={music_volume}[music];"
            f"[voice][music]amix=inputs=2:duration=shortest[aout]"
        )
        cmd += [
            "-map", "0:v:0",
            "-filter_complex", filter_str,
            "-map", "[aout]",
        ]
    else:
        # Voice only
        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-filter:a", f"volume={voice_volume}",
            "-shortest",
        ]

    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", output_path]

    print(f"[audio] Mixing voice (vol={voice_volume}) + music (vol={music_volume}) …")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[audio] Saved → {output_path}")
    return output_path
