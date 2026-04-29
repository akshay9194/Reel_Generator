"""
ui.py — Gradio web interface for Reel Generator.

Launch:
    cd app
    python ui.py

Opens a browser UI where you can:
  1. Upload a face photo
  2. Type a prompt
  3. (Optional) Upload background music
  4. (Optional) Upload voice audio for lip-sync
  5. Adjust settings (frames, seed, ip-adapter strength, etc.)
  6. Click Generate → get a downloadable video with audio
"""

import gradio as gr
import tempfile
import shutil
from pathlib import Path

from generate import generate_images, unload_pipeline
from video import create_video
from audio import add_background_music, mix_voice_and_music
from lipsync import run_lipsync, is_sadtalker_ready
from config import (
    DEFAULT_NUM_FRAMES,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEPS,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_FPS,
    DEFAULT_HOLD_FRAMES,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
)


def run(
    face_image,
    prompt,
    num_frames,
    guidance_scale,
    num_steps,
    ip_adapter_scale,
    fps,
    hold_frames,
    width,
    height,
    seed,
    music_file,
    voice_file,
    music_volume,
):
    """Gradio callback — two modes depending on whether voice audio is provided."""
    if face_image is None:
        raise gr.Error("Please upload a face image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    # Save uploaded face to a temp file
    face_path = Path(tempfile.gettempdir()) / "gradio_face_input.jpg"
    face_image.save(str(face_path), "JPEG", quality=95)

    seed_val = int(seed) if seed and seed > 0 else None
    has_voice = voice_file is not None
    has_music = music_file is not None
    status_parts = []

    if has_voice:
        # ── TALKING-HEAD MODE ─────────────────────────────
        # Generate one best frame, then SadTalker animates it
        image_paths = generate_images(
            prompt.strip(),
            str(face_path),
            num_images=1,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_steps),
            ip_adapter_scale=float(ip_adapter_scale),
            width=int(width),
            height=int(height),
            seed=seed_val,
        )
        status_parts.append("Generated 1 frame for talking-head")

        # SadTalker: face image + voice → animated video with lip-sync
        video_path = run_lipsync(image_paths[0], voice_file)
        status_parts.append("SadTalker: head motion + lip-sync applied")

        # Optional background music
        if has_music:
            video_path = mix_voice_and_music(
                video_path, voice_file, music_file,
                music_volume=float(music_volume),
            )
            status_parts.append(f"Background music mixed (vol={music_volume})")

    else:
        # ── SLIDESHOW MODE ────────────────────────────────
        # Generate N frames and stitch into video
        image_paths = generate_images(
            prompt.strip(),
            str(face_path),
            num_images=int(num_frames),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_steps),
            ip_adapter_scale=float(ip_adapter_scale),
            width=int(width),
            height=int(height),
            seed=seed_val,
        )

        video_path = create_video(
            image_paths,
            fps=int(fps),
            hold_frames=int(hold_frames),
        )

        total_frames = int(num_frames) * int(hold_frames)
        duration = total_frames / int(fps)
        status_parts.append(f"Slideshow: {int(num_frames)} frames → {duration:.1f}s at {int(fps)} fps")

        if has_music:
            video_path = add_background_music(
                video_path, music_file,
                music_volume=float(music_volume),
            )
            status_parts.append(f"Background music added (vol={music_volume})")

    status = "✅ " + " | ".join(status_parts)
    return video_path, status


def calculate_duration(num_frames, hold_frames, fps):
    """Live preview of video duration."""
    total = int(num_frames) * int(hold_frames)
    dur = total / int(fps)
    return f"📐 {int(num_frames)} images × {int(hold_frames)} repeats = {total} frames ÷ {int(fps)} fps = **{dur:.1f} seconds**"


# ─── Build UI ────────────────────────────────────────────
with gr.Blocks(
    title="🎬 Reel Generator",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 960px !important; }",
) as app:

    gr.Markdown("# 🎬 Reel Generator\nFace image + prompt → consistent AI reel video with audio & lip-sync")

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            face_input = gr.Image(
                label="📸 Face Photo",
                type="pil",
                height=256,
            )
            prompt_input = gr.Textbox(
                label="✏️ Prompt",
                placeholder="beautiful girl in cafe, cinematic lighting, instagram style",
                lines=3,
            )

            with gr.Accordion("🎵 Audio & Lip-Sync", open=True):
                music_input = gr.Audio(
                    label="🎶 Background Music (optional)",
                    type="filepath",
                    sources=["upload"],
                )
                voice_input = gr.Audio(
                    label="🗣️ Voice Audio for Lip-Sync (optional)",
                    type="filepath",
                    sources=["upload"],
                )
                music_volume = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05,
                    label="Music Volume (0.3 = subtle background)",
                )
                gr.Markdown(
                    "💡 **Voice → Talking Head:** Upload voice audio and SadTalker will generate "
                    "a video with natural head motion, eye blinks, and lip-sync from a single AI-generated frame.\n\n"
                    "**No voice = Slideshow mode:** multiple frames stitched together."
                )

            with gr.Accordion("⚙️ Generation Settings", open=False):
                num_frames = gr.Slider(2, 16, value=DEFAULT_NUM_FRAMES, step=1, label="Number of frames")
                guidance_scale = gr.Slider(1.0, 20.0, value=DEFAULT_GUIDANCE_SCALE, step=0.5, label="Guidance scale")
                num_steps = gr.Slider(10, 50, value=DEFAULT_NUM_STEPS, step=5, label="Inference steps")
                ip_scale = gr.Slider(0.0, 1.0, value=DEFAULT_IP_ADAPTER_SCALE, step=0.05, label="IP-Adapter scale (face strength)")
                fps_slider = gr.Slider(1, 30, value=DEFAULT_FPS, step=1, label="FPS")
                hold_slider = gr.Slider(1, 30, value=DEFAULT_HOLD_FRAMES, step=1, label="Hold frames (repeats per image)")
                width_input = gr.Dropdown([512, 768, 1024], value=DEFAULT_WIDTH, label="Width")
                height_input = gr.Dropdown([512, 768, 1024], value=DEFAULT_HEIGHT, label="Height")
                seed_input = gr.Number(value=0, label="Seed (0 = random)", precision=0)

            duration_preview = gr.Markdown(
                value=calculate_duration(DEFAULT_NUM_FRAMES, DEFAULT_HOLD_FRAMES, DEFAULT_FPS)
            )

            generate_btn = gr.Button("🎬 Generate Reel", variant="primary", size="lg")

        # Right column — output
        with gr.Column(scale=1):
            video_output = gr.Video(label="🎥 Generated Reel")
            status_output = gr.Markdown("Waiting for generation…")

    # ─── Events ───────────────────────────────────────────
    # Live duration calculator
    for control in [num_frames, hold_slider, fps_slider]:
        control.change(
            fn=calculate_duration,
            inputs=[num_frames, hold_slider, fps_slider],
            outputs=duration_preview,
        )

    # Generate button
    generate_btn.click(
        fn=run,
        inputs=[
            face_input,
            prompt_input,
            num_frames,
            guidance_scale,
            num_steps,
            ip_scale,
            fps_slider,
            hold_slider,
            width_input,
            height_input,
            seed_input,
            music_input,
            voice_input,
            music_volume,
        ],
        outputs=[video_output, status_output],
    )

    gr.Markdown(
        "---\n"
        "**Tips:**\n"
        "- **With voice audio** → SadTalker mode: generates one frame, animates it with head motion + lip-sync\n"
        "- **Without voice** → Slideshow mode: generates N frames, stitches into video\n"
        "- **IP-Adapter scale** 0.5–0.7 is the sweet spot (higher = stronger face lock)\n"
        "- **Music volume** 0.2–0.3 works best as background behind voice\n"
        "- Set a **seed** for reproducible results"
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,    # set True to get a public Gradio link
    )
