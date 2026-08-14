import gc
import math
import os
import uuid
import warnings

# This is a harmless Transformers deprecation warning emitted by a dependency.
warnings.filterwarnings(
    "ignore",
    message=r".Siglip2ImageProcessorFast.*deprecated.",
)

import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from PIL import Image
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# MoviePy v1/v2 compatibility.
try:
    from moviepy.editor import AudioFileClip, ImageSequenceClip
except ImportError:
    from moviepy import AudioFileClip, ImageSequenceClip


# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_MODEL_ID = "segmind/tiny-sd"
MUSIC_MODEL_ID = "facebook/musicgen-small"

# Five unique generated images are used for every second of video.
IMAGES_PER_SECOND = 5

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
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
    def _init_(self, model_id=MUSIC_MODEL_ID):
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
    def _init_(self, model_id=IMAGE_MODEL_ID):
        print("[ImageAgent] Loading Tiny Stable Diffusion...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
        ).to(DEVICE)

        # Reuse the same model components for image-to-image generation.
        # Later frames will be derived from the preceding frame instead of
        # being unrelated text-to-image results.
        self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)

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
        width=512,
        height=512,
        steps=20,
        seed=42,
        continuity_strength=0.22,
        progress=None,
    ):
        image_paths = []
        negative_prompt = (
            "different person, different character, different clothing, "
            "different background, scene change, camera angle change, "
            "blurry, low quality, distorted, deformed, watermark, text"
        )

        continuity_strength = float(continuity_strength)
        if not 0.0 < continuity_strength < 1.0:
            raise ValueError("Continuity strength must be between 0 and 1.")

        # Generate one text-to-image reference frame.
        if progress is not None:
            progress(0.20, desc="Generating the reference frame")

        reference_generator = torch.Generator(device=DEVICE).manual_seed(
            int(seed)
        )

        cleanup_memory()
        with torch.inference_mode():
            reference_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                generator=reference_generator,
            ).images[0].convert("RGB")

        reference_path = output_filename("frame_0001", "png")
        reference_image.save(reference_path)
        image_paths.append(reference_path)
        previous_image = reference_image

        print(f"[ImageAgent] Reference frame 1/{frame_count}")

        # Derive every later frame from the preceding frame. Blending a small
        # amount of the first frame back in prevents cumulative scene drift.
        for index in range(1, int(frame_count)):
            timestamp = index / float(images_per_second)

            if progress is not None:
                completed = index / max(frame_count, 1)
                progress(
                    0.20 + (completed * 0.70),
                    desc=f"Generating consistent image {index + 1}/{frame_count}",
                )

            frame_prompt = (
                f"{prompt}, same person, same character, same face, "
                "same clothing, same environment, same background, "
                "same camera angle, same lighting, same cinematic style, "
                "subtle natural movement"
            )

            # 15% reference-frame anchoring keeps the sequence from gradually
            # becoming a completely different scene.
            anchored_image = Image.blend(
                previous_image.convert("RGB"),
                reference_image,
                alpha=0.15,
            )

            frame_generator = torch.Generator(device=DEVICE).manual_seed(
                int(seed) + index
            )

            cleanup_memory()
            with torch.inference_mode():
                current_image = self.img2img_pipe(
                    prompt=frame_prompt,
                    negative_prompt=negative_prompt,
                    image=anchored_image,
                    strength=continuity_strength,
                    guidance_scale=6.0,
                    num_inference_steps=max(int(steps), 20),
                    generator=frame_generator,
                ).images[0].convert("RGB")

            image_path = output_filename(
                f"frame_{index + 1:04d}",
                "png",
            )
            current_image.save(image_path)
            image_paths.append(image_path)
            previous_image = current_image

            print(
                f"[ImageAgent] Consistent frame {index + 1}/{frame_count} "
                f"({timestamp:.1f}s)"
            )

        return image_paths

    def close(self):
        del self.img2img_pipe
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
        continuity_strength=0.22,
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
                continuity_strength=continuity_strength,
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
    continuity_strength,
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
            continuity_strength=float(continuity_strength),
            seed=int(seed),
            progress=progress,
        )
    except Exception as error:
        cleanup_memory()
        print(f"[Error] {type(error)._name_}: {error}")
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
        gr.Slider(
            minimum=0.10,
            maximum=0.40,
            value=0.22,
            step=0.01,
            label=(
                "Frame Change Strength "
                "(lower = more consistent, higher = more motion)"
            ),
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
        "The app creates music, generates five images per second, and uses each "
        "preceding image as the reference for the next frame to preserve the "
        "scene. Lower Frame Change Strength gives stronger visual consistency. "
        "The first run downloads the models and can take several minutes."
    ),
)


if _name_ == "_main_":
    print("[System] Starting Gradio at http://127.0.0.1:7860")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        prevent_thread_lock=False,
    )