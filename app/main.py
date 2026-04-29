"""
main.py — Local entry-point / CLI for the Reel Generator pipeline.

Two modes:
  A) Slideshow reel (no voice)  → generate frames → stitch → optional music
  B) Talking-head reel (voice)  → generate best frame → SadTalker animates
                                  with head motion + lip-sync → optional music

Usage:
    python main.py                                             # basic slideshow
    python main.py --music bg.mp3                              # slideshow + music
    python main.py --voice dialogue.wav                        # talking-head + lip-sync
    python main.py --voice dialogue.wav --music bg.mp3         # talking-head + music
"""

import argparse
import sys
import time
from pathlib import Path

from generate import generate_images, unload_pipeline
from video import create_video
from audio import add_background_music, mix_voice_and_music
from lipsync import run_lipsync


def run_pipeline(
    prompt: str,
    face_path: str,
    num_frames: int = 6,
    seed: int | None = None,
    fps: int = 5,
    hold_frames: int = 8,
    music_path: str | None = None,
    voice_path: str | None = None,
    music_volume: float = 0.3,
) -> str:
    if not Path(face_path).is_file():
        raise FileNotFoundError(f"Face image not found: {face_path}")
    if music_path and not Path(music_path).is_file():
        raise FileNotFoundError(f"Music file not found: {music_path}")
    if voice_path and not Path(voice_path).is_file():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")

    has_voice = voice_path is not None
    has_music = music_path is not None
    mode = "talking-head" if has_voice else "slideshow"

    t0 = time.time()

    print("━" * 50)
    print("🎬  Reel Generator Pipeline")
    print("━" * 50)
    print(f"  Mode     : {mode}")
    print(f"  Prompt   : {prompt}")
    print(f"  Face     : {face_path}")
    if not has_voice:
        print(f"  Frames   : {num_frames}")
    print(f"  Music    : {music_path or '(none)'}")
    print(f"  Voice    : {voice_path or '(none)'}")
    print("━" * 50)

    if has_voice:
        # ── MODE B: Talking-head ──────────────────────────
        # Generate one best frame with SDXL+IP-Adapter, then
        # let SadTalker animate it (head motion + lip-sync)
        step, total = 0, 2 + (1 if has_music else 0)

        step += 1
        print(f"\n[{step}/{total}] Generating face frame with SDXL …")
        image_paths = generate_images(
            prompt,
            face_path,
            num_images=1,       # only need one good frame
            seed=seed,
        )
        source_frame = image_paths[0]
        print(f"       → {source_frame}")

        step += 1
        print(f"\n[{step}/{total}] Running SadTalker (lip-sync + head motion) …")
        video_path = run_lipsync(source_frame, voice_path)

        if has_music:
            step += 1
            print(f"\n[{step}/{total}] Mixing in background music …")
            video_path = mix_voice_and_music(
                video_path, voice_path, music_path,
                music_volume=music_volume,
            )

    else:
        # ── MODE A: Slideshow reel ────────────────────────
        step, total = 0, 2 + (1 if has_music else 0)

        step += 1
        print(f"\n[{step}/{total}] Generating images …")
        image_paths = generate_images(
            prompt,
            face_path,
            num_images=num_frames,
            seed=seed,
        )
        print(f"       → {len(image_paths)} images saved")

        step += 1
        print(f"\n[{step}/{total}] Stitching video …")
        video_path = create_video(image_paths, fps=fps, hold_frames=hold_frames)

        if has_music:
            step += 1
            print(f"\n[{step}/{total}] Adding background music …")
            video_path = add_background_music(
                video_path, music_path,
                music_volume=music_volume,
            )

    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.1f}s → {video_path}")
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Reel Generator – face-consistent video pipeline")
    parser.add_argument("--prompt", type=str, default="beautiful indian girl in cafe, cinematic lighting, instagram style")
    parser.add_argument("--face", type=str, default="face.jpg")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--hold", type=int, default=8, help="Duplicate each frame N times for pacing")
    parser.add_argument("--music", type=str, default=None, help="Background music file (mp3/wav)")
    parser.add_argument("--voice", type=str, default=None, help="Voice audio for lip-sync (mp3/wav)")
    parser.add_argument("--music-volume", type=float, default=0.3, help="Music volume 0.0–1.0")
    args = parser.parse_args()

    try:
        run_pipeline(
            prompt=args.prompt,
            face_path=args.face,
            num_frames=args.frames,
            seed=args.seed,
            fps=args.fps,
            hold_frames=args.hold,
            music_path=args.music,
            voice_path=args.voice,
            music_volume=args.music_volume,
        )
    except Exception as exc:
        print(f"\n❌  Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        unload_pipeline()


if __name__ == "__main__":
    main()
