import gc
import math
import os
import uuid
import warnings

# Silence non-fatal Transformers advisory/deprecation logger messages before
# importing Diffusers or Transformers.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# This is a harmless Transformers deprecation warning emitted by a dependency.
warnings.filterwarnings(
    "ignore",
    message=r".*Siglip2ImageProcessorFast.*deprecated.*",
)

import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

# MoviePy v1/v2 compatibility.
try:
    from moviepy.editor import AudioFileClip, ImageSequenceClip
except ImportError:
    from moviepy import AudioFileClip, ImageSequenceClip


# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_MODEL_ID = "segmind/SSD-1B"
MUSIC_MODEL_ID = "facebook/musicgen-melody"

# Five unique generated images are used for every second of video.
IMAGES_PER_SECOND = 24

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[System] Device: {DEVICE}")
print(f"[System] Torch dtype: {DTYPE}")


# =====================================================
# UTILITIES
# =====================================================

def cleanup_memory():
    """Release Python and CUDA memory that is no longer in use."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def output_filename(prefix, extension):
    """Return a collision-resistant output path."""
    name = f"{prefix}_{uuid.uuid4().hex[:10]}.{extension}"
    return os.path.join(OUTPUT_DIR, name)


def clip_duration(audio_path):
    """Read an audio file's duration and close it immediately."""
    clip = AudioFileClip(audio_path)
    try:
        return float(clip.duration)
    finally:
        clip.close()


# =====================================================
# MUSIC GENERATION
# =====================================================

class MusicGenerationAgent:
    def __init__(self, model_id=MUSIC_MODEL_ID):
        print("[MusicAgent] Loading MusicGen...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_id
        ).to(DEVICE)
        self.model.eval()
        print("[MusicAgent] Ready.")

    def run(self, prompt, max_new_tokens=128, guidance_scale=3.0):
        print(f"[MusicAgent] Generating: {prompt}")
        cleanup_memory()

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                guidance_scale=float(guidance_scale),
            )

        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio = audio_values[0, 0].detach().float().cpu().numpy()

        audio = np.nan_to_num(audio)
        maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
        if maximum > 0:
            audio = audio / maximum

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_path = output_filename("music", "wav")
        write_wav(audio_path, sampling_rate, audio_int16)
        print(f"[MusicAgent] Saved: {audio_path}")
        return audio_path

    def close(self):
        del self.model
        del self.processor
        cleanup_memory()


# =====================================================
# IMAGE GENERATION
# =====================================================

class ImageGenerationAgent:
    def __init__(self, model_id=IMAGE_MODEL_ID):
        print("[ImageAgent] Loading Tiny Stable Diffusion...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
        ).to(DEVICE)

        # These reduce peak memory usage.
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

        print("[ImageAgent] Ready.")

    def generate_sequence(
        self,
        prompt,
        frame_count,
        images_per_second=IMAGES_PER_SECOND,
        width=1920,
        height=1080,
        steps=20,
        seed=42,
        motion_direction="Content moves up",
        pan_amount=0.80,
        zoom_amount=0.08,
        progress=None,
    ):
        """Generate one scene, then turn progressive crops into video frames.

        All output frames come from exactly the same master scene. The crop
        position and size change gradually, so neighboring images behave like
        consecutive video frames instead of unrelated AI generations.
        """
        frame_count = max(1, int(frame_count))
        width = int(width)
        height = int(height)
        pan_amount = max(0.0, min(float(pan_amount), 1.0))
        zoom_amount = max(0.0, min(float(zoom_amount), 0.30))

        vertical_motion = motion_direction in {
            "Content moves up",
            "Content moves down",
        }

        if vertical_motion:
            # A tall master provides extra scene above and below the output.
            master_width = int(math.ceil(width / 8) * 8)
            master_height = max(height + 256, int(height * 1.5))
            master_height = int(math.ceil(master_height / 8) * 8)
            composition_prompt = (
                "tall cinematic composition, extra environment visible "
                "above and below, single continuous scene"
            )
        else:
            # A wide master provides extra scene left and right of the output.
            master_width = max(width + 256, int(width * 1.5))
            master_width = int(math.ceil(master_width / 8) * 8)
            master_height = int(math.ceil(height / 8) * 8)
            composition_prompt = (
                "wide cinematic composition, extra environment visible "
                "on the left and right, single continuous scene"
            )

        negative_prompt = (
            "blurry, low quality, distorted, deformed, duplicate subject, "
            "watermark, logo, letters, text"
        )

        if progress is not None:
            progress(0.20, desc="Generating one wide master scene")

        generator = torch.Generator(device=DEVICE).manual_seed(
            int(seed)
        )

        cleanup_memory()
        with torch.inference_mode():
            master_image = self.pipe(
                prompt=(
                    f"{prompt}, {composition_prompt}"
                ),
                negative_prompt=negative_prompt,
                width=master_width,
                height=master_height,
                num_inference_steps=int(steps),
                generator=generator,
            ).images[0].convert("RGB")

        master_path = output_filename("master_scene", "png")
        master_image.save(master_path)
        print(f"[ImageAgent] Master scene saved: {master_path}")

        source_width, source_height = master_image.size
        output_ratio = width / float(height)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image_paths = []

        for index in range(frame_count):
            timestamp = index / float(images_per_second)
            linear_t = index / max(frame_count - 1, 1)

            # Smoothstep creates gentle acceleration and deceleration instead
            # of robotic constant-speed movement.
            motion_t = linear_t * linear_t * (3.0 - 2.0 * linear_t)

            if progress is not None:
                completed = index / max(frame_count, 1)
                progress(
                    0.65 + (completed * 0.25),
                    desc=f"Rendering motion frame {index + 1}/{frame_count}",
                )

            if motion_direction == "Zoom out":
                zoom = 1.0 + zoom_amount * (1.0 - motion_t)
            else:
                zoom = 1.0 + zoom_amount * motion_t

            # Start with the largest output-aspect crop that fits the master,
            # then progressively reduce it for a real zoom on both wide and
            # tall source images.
            base_crop_width = min(
                float(source_width),
                float(source_height) * output_ratio,
            )
            base_crop_height = base_crop_width / output_ratio

            crop_width = max(8, int(base_crop_width / zoom))
            crop_height = max(8, int(base_crop_height / zoom))
            crop_width = min(crop_width, source_width)
            crop_height = min(crop_height, source_height)

            horizontal_space = max(0.0, source_width - crop_width)
            travel = horizontal_space * pan_amount

            # Moving the crop right makes the visible content move left.
            if motion_direction == "Content moves left":
                crop_left = (horizontal_space - travel) / 2.0
                crop_left += travel * motion_t
            elif motion_direction == "Content moves right":
                crop_left = (horizontal_space + travel) / 2.0
                crop_left -= travel * motion_t
            else:
                crop_left = horizontal_space / 2.0

            vertical_space = max(0.0, source_height - crop_height)
            vertical_travel = vertical_space * pan_amount

            # Moving the crop downward makes the visible content move upward.
            if motion_direction == "Content moves up":
                crop_top = (vertical_space - vertical_travel) / 2.0
                crop_top += vertical_travel * motion_t
            elif motion_direction == "Content moves down":
                crop_top = (vertical_space + vertical_travel) / 2.0
                crop_top -= vertical_travel * motion_t
            else:
                crop_top = vertical_space / 2.0

            left = int(round(max(0.0, min(crop_left, horizontal_space))))
            top = int(round(max(0.0, min(crop_top, vertical_space))))
            right = min(source_width, left + crop_width)
            bottom = min(source_height, top + crop_height)

            frame = master_image.crop((left, top, right, bottom))
            frame = frame.resize((width, height), resampling)

            image_path = output_filename(
                f"frame_{index + 1:04d}",
                "png",
            )
            frame.save(image_path)
            image_paths.append(image_path)

            print(
                f"[ImageAgent] Motion frame {index + 1}/{frame_count} "
                f"({timestamp:.1f}s, x={left}, zoom={zoom:.3f})"
            )

        return image_paths

    def close(self):
        del self.pipe
        cleanup_memory()


# =====================================================
# VIDEO GENERATION
# =====================================================

class VideoGenerationAgent:
    def run(self, image_paths, audio_path, fps=IMAGES_PER_SECOND):
        if not image_paths:
            raise ValueError("No images were supplied for video generation.")

        print("[VideoAgent] Building MP4...")
        audio_clip = AudioFileClip(audio_path)
        video_clip = None

        try:
            duration = float(audio_clip.duration)
            video_clip = ImageSequenceClip(image_paths, fps=int(fps))

            # Trim the final partial image to the exact audio duration.
            if hasattr(video_clip, "set_duration"):
                video_clip = video_clip.set_duration(duration)
            else:
                video_clip = video_clip.with_duration(duration)

            if hasattr(video_clip, "set_audio"):
                video_clip = video_clip.set_audio(audio_clip)
            else:
                video_clip = video_clip.with_audio(audio_clip)

            video_path = output_filename("video", "mp4")
            write_options = {
                "filename": video_path,
                "fps": int(fps),
                "codec": "libx264",
                "audio_codec": "aac",
                "ffmpeg_params": ["-pix_fmt", "yuv420p"],
                "logger": None,
            }

            # MoviePy v1 accepts verbose; MoviePy v2 removed it.
            try:
                video_clip.write_videofile(verbose=False, **write_options)
            except TypeError as error:
                if "verbose" not in str(error):
                    raise
                video_clip.write_videofile(**write_options)

            print(f"[VideoAgent] Saved: {video_path}")
            return video_path
        finally:
            if video_clip is not None:
                video_clip.close()
            audio_clip.close()
            cleanup_memory()


# =====================================================
# COMPLETE WORKFLOW
# =====================================================

class MediaOrchestratorAgent:
    def run(
        self,
        main_prompt,
        image_steps=20,
        music_tokens=128,
        music_guidance=3.0,
        motion_direction="Content moves left",
        pan_amount=0.80,
        zoom_amount=0.08,
        seed=42,
        progress=None,
    ):
        image_prompt = (
            f"{main_prompt}, cinematic, detailed, high quality, "
            "beautiful lighting, visually striking"
        )
        music_prompt = (
            f"instrumental background music for {main_prompt}, "
            "cinematic, atmospheric, no vocals, clean audio"
        )

        # Generate and unload music first so both large models do not occupy
        # memory simultaneously. This is particularly helpful on CPU systems.
        if progress is not None:
            progress(0.02, desc="Loading the music model")

        music_agent = MusicGenerationAgent()
        try:
            if progress is not None:
                progress(0.08, desc="Generating background music")
            audio_path = music_agent.run(
                prompt=music_prompt,
                max_new_tokens=music_tokens,
                guidance_scale=music_guidance,
            )
        finally:
            music_agent.close()

        duration = clip_duration(audio_path)
        frame_count = max(1, math.ceil(duration * IMAGES_PER_SECOND))
        print(
            f"[Orchestrator] Duration: {duration:.2f}s | "
            f"Images: {frame_count} | Rate: {IMAGES_PER_SECOND}/second"
        )

        if progress is not None:
            progress(
                0.18,
                desc=f"Loading image model for {frame_count} images",
            )

        image_agent = ImageGenerationAgent()
        try:
            image_paths = image_agent.generate_sequence(
                prompt=image_prompt,
                frame_count=frame_count,
                images_per_second=IMAGES_PER_SECOND,
                width=512,
                height=512,
                steps=image_steps,
                seed=seed,
                motion_direction=motion_direction,
                pan_amount=pan_amount,
                zoom_amount=zoom_amount,
                progress=progress,
            )
        finally:
            image_agent.close()

        if progress is not None:
            progress(0.93, desc="Creating the final MP4")

        video_path = VideoGenerationAgent().run(
            image_paths=image_paths,
            audio_path=audio_path,
            fps=IMAGES_PER_SECOND,
        )

        if progress is not None:
            progress(1.0, desc="Complete")

        return image_paths, audio_path, video_path


orchestrator = MediaOrchestratorAgent()


def generate_all(
    main_prompt,
    image_steps,
    music_tokens,
    music_guidance,
    motion_direction,
    pan_amount,
    zoom_amount,
    seed,
    progress=gr.Progress(),
):
    if not main_prompt or not main_prompt.strip():
        raise gr.Error("Please enter a prompt first.")

    try:
        return orchestrator.run(
            main_prompt=main_prompt.strip(),
            image_steps=int(image_steps),
            music_tokens=int(music_tokens),
            music_guidance=float(music_guidance),
            motion_direction=str(motion_direction),
            pan_amount=float(pan_amount),
            zoom_amount=float(zoom_amount),
            seed=int(seed),
            progress=progress,
        )
    except Exception as error:
        cleanup_memory()
        print(f"[Error] {type(error).__name__}: {error}")
        raise gr.Error(f"Generation failed: {error}") from error


# =====================================================
# GRADIO INTERFACE
# =====================================================

demo = gr.Interface(
    fn=generate_all,
    inputs=[
        gr.Textbox(
            label="Main Prompt",
            placeholder="Example: A futuristic classroom with AI holograms",
            value=(
                "A futuristic classroom with AI holograms and students "
                "learning technology"
            ),
        ),
        gr.Slider(
            minimum=10,
            maximum=40,
            value=20,
            step=1,
            label="Image Generation Steps",
        ),
        gr.Slider(
            minimum=64,
            maximum=512,
            value=128,
            step=64,
            label="Music Length Tokens (approximately 50 tokens/second)",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=6.0,
            value=3.0,
            step=0.5,
            label="Music Guidance Scale",
        ),
        gr.Dropdown(
            choices=[
                "Content moves left",
                "Content moves right",
                "Zoom in",
                "Zoom out",
                "Content moves up",
                "Content moves down",
            ],
            value="Content moves left",
            label="Camera Motion",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.80,
            step=0.05,
            label="Pan Distance",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=0.25,
            value=0.08,
            step=0.01,
            label="Progressive Zoom",
        ),
        gr.Number(
            value=42,
            precision=0,
            label="Seed",
        ),
    ],
    outputs=[
        gr.Gallery(
            label="Generated Images (5 per second)",
            columns=5,
            height=420,
        ),
        gr.Audio(label="Generated Background Music"),
        gr.Video(label="Final Video"),
    ],
    title="AI Image + Music + Video Generator",
    description=(
        "The app generates one wide AI scene and renders five unique motion "
        "frames per second by progressively panning and zooming across it. "
        "Every frame continues from the previous position, so the video keeps "
        "the same characters and environment without AI redraw flicker."
    ),
)


if __name__ == "__main__":
    print(
        "[System] Starting Gradio at http://127.0.0.1:7860",
        flush=True,
    )
    try:
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            show_error=True,
            prevent_thread_lock=False,
        )
    except Exception as error:
        print(
            f"[Startup Error] {type(error).__name__}: {error}",
            flush=True,
        )
        raise
