"""
handler.py — RunPod Serverless handler.

Receives a JSON payload, runs the pipeline, and returns
the video as a base64-encoded string (or a presigned URL
if you wire up S3/R2 — see upload_result()).

Expected input:
{
    "input": {
        "prompt": "beautiful girl in cafe, cinematic",
        "face_image_base64": "<base64 jpg/png>",
        "num_frames": 6,
        "seed": 42,                          // optional
        "guidance_scale": 7.5,               // optional
        "num_inference_steps": 30,           // optional
        "ip_adapter_scale": 0.6,             // optional
        "fps": 5,                            // optional
        "hold_frames": 8,                    // optional
        "music_base64": "<base64 mp3/wav>",  // optional — background music
        "voice_base64": "<base64 mp3/wav>",  // optional — voice for lip-sync
        "music_volume": 0.3                  // optional
    }
}
"""

import base64
import io
import os
import tempfile
import traceback

import runpod
from PIL import Image

from generate import generate_images, unload_pipeline
from video import create_video
from audio import add_background_music, mix_voice_and_music
from lipsync import run_lipsync
from config import (
    DEFAULT_NUM_FRAMES,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEPS,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_FPS,
    DEFAULT_HOLD_FRAMES,
)


def _decode_face(b64: str) -> str:
    """Decode a base64 face image and save to a temp file. Returns the path."""
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")

    path = os.path.join(tempfile.gettempdir(), "face_input.jpg")
    img.save(path, "JPEG", quality=95)
    return path


def _decode_audio(b64: str, name: str) -> str:
    """Decode a base64 audio file and save to a temp file. Returns the path."""
    data = base64.b64decode(b64)
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _encode_video(path: str) -> str:
    """Read a video file and return its base64 representation."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(event: dict) -> dict:
    """RunPod serverless handler entry-point."""
    try:
        inp = event.get("input", {})

        prompt = inp.get("prompt")
        face_b64 = inp.get("face_image_base64")

        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        if not face_b64:
            return {"error": "Missing 'face_image_base64' in input"}

        # Decode face image
        face_path = _decode_face(face_b64)

        # Decode optional audio
        music_b64 = inp.get("music_base64")
        voice_b64 = inp.get("voice_base64")
        music_path = _decode_audio(music_b64, "bg_music.mp3") if music_b64 else None
        voice_path = _decode_audio(voice_b64, "voice.wav") if voice_b64 else None
        music_volume = float(inp.get("music_volume", 0.3))

        # Optional overrides
        num_frames = int(inp.get("num_frames", DEFAULT_NUM_FRAMES))
        seed = inp.get("seed")
        if seed is not None:
            seed = int(seed)
        guidance_scale = float(inp.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        num_steps = int(inp.get("num_inference_steps", DEFAULT_NUM_STEPS))
        ip_scale = float(inp.get("ip_adapter_scale", DEFAULT_IP_ADAPTER_SCALE))
        fps = int(inp.get("fps", DEFAULT_FPS))
        hold = int(inp.get("hold_frames", DEFAULT_HOLD_FRAMES))

        # Generate frames
        features = []

        if voice_path:
            # ── TALKING-HEAD MODE ─────────────────────────
            # Generate one frame, let SadTalker animate it
            image_paths = generate_images(
                prompt, face_path,
                num_images=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                ip_adapter_scale=ip_scale,
                seed=seed,
            )

            video_path = run_lipsync(image_paths[0], voice_path)
            features.append("sadtalker_lipsync")

            if music_path:
                video_path = mix_voice_and_music(
                    video_path, voice_path, music_path,
                    music_volume=music_volume,
                )
                features.append("voice+music")

        else:
            # ── SLIDESHOW MODE ────────────────────────────
            image_paths = generate_images(
                prompt, face_path,
                num_images=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                ip_adapter_scale=ip_scale,
                seed=seed,
            )

            video_path = create_video(image_paths, fps=fps, hold_frames=hold)
            features.append("slideshow")

            if music_path:
                video_path = add_background_music(
                    video_path, music_path,
                    music_volume=music_volume,
                )
                features.append("music")

        # Return base64 video
        video_b64 = _encode_video(video_path)

        return {
            "video_base64": video_b64,
            "num_frames": len(image_paths),
            "features": features,
            "video_path": video_path,
        }

    except Exception:
        traceback.print_exc()
        return {"error": traceback.format_exc()}


# RunPod entry
runpod.serverless.start({"handler": handler})
