# AI Image + Music + Video Generator

## Project Documentation

### 1. Project Overview

This project is a **multimodal Generative AI application** that converts a single text prompt into:

1. A generated background music track
2. A sequence of AI-generated images
3. A final MP4 video created from the generated images and music

The application uses a modular, agent-style architecture in which separate components are responsible for music generation, image generation, video creation, and overall workflow orchestration.

A **Gradio web interface** is provided so users can control generation parameters and run the complete pipeline from a browser.

---

## 2. Objective

The main objective of the project is to demonstrate how multiple generative AI models can be combined into one automated multimedia generation pipeline.

Instead of generating an image, music track, or video independently, the system coordinates multiple models to produce a unified multimedia output from a single user prompt.

### Example

Input:

```text
A futuristic classroom with AI holograms and students learning technology
```

The application:

```text
User Prompt
    ↓
Music Prompt + Image Prompt
    ↓
AI Music Generation
    ↓
Determine Audio Duration
    ↓
Generate Image Sequence
    ↓
Maintain Frame-to-Frame Consistency
    ↓
Combine Frames + Music
    ↓
Final MP4 Video
```

---

## 3. Main Features

- Text-to-music generation
- Text-to-image generation
- Image-to-image frame continuation
- Automated video generation
- Background music synchronization
- Frame continuity control
- Configurable random seed
- Adjustable image generation quality
- Adjustable music generation parameters
- Automatic CPU/GPU device selection
- Memory cleanup between large AI models
- MoviePy v1/v2 compatibility
- Browser-based Gradio user interface
- Automatic output file generation

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application language |
| PyTorch | Model inference and hardware acceleration |
| Hugging Face Transformers | MusicGen model loading and generation |
| Hugging Face Diffusers | Stable Diffusion image generation |
| Tiny Stable Diffusion | Text-to-image and image-to-image generation |
| MusicGen | AI-generated background music |
| Gradio | Web-based user interface |
| MoviePy | Video and audio composition |
| Pillow | Image processing and frame blending |
| NumPy | Audio array processing |
| SciPy | Saving generated audio as WAV |
| FFmpeg | MP4 video encoding through MoviePy |

---

## 5. AI Models

### Image Model

```text
segmind/tiny-sd
```

The project uses Tiny Stable Diffusion through the Hugging Face Diffusers library.

It is used in two modes:

- **Text-to-Image** for the initial reference frame
- **Image-to-Image** for subsequent video frames

### Music Model

```text
facebook/musicgen-small
```

MusicGen Small is used to create instrumental music based on the original user prompt.

---

## 6. System Architecture

The project follows a modular agent-oriented design.

```text
                         ┌──────────────────────┐
                         │      Gradio UI       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ MediaOrchestratorAgent│
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  │                                    │
                  ▼                                    ▼
       ┌──────────────────────┐            ┌──────────────────────┐
       │ MusicGenerationAgent │            │ ImageGenerationAgent │
       └──────────┬───────────┘            └──────────┬───────────┘
                  │                                    │
                  ▼                                    ▼
       facebook/musicgen-small               segmind/tiny-sd
                  │                                    │
                  ▼                                    ▼
              WAV Audio                         PNG Frames
                  │                                    │
                  └─────────────────┬──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ VideoGenerationAgent │
                         └──────────┬───────────┘
                                    │
                                    ▼
                               MP4 Video
```

---

## 7. Project Components

### 7.1 Configuration

The application defines the model IDs and generation rate:

```python
IMAGE_MODEL_ID = "segmind/tiny-sd"
MUSIC_MODEL_ID = "facebook/musicgen-small"
IMAGES_PER_SECOND = 5
```

The generated files are stored inside:

```text
output/
```

The application automatically detects whether CUDA is available:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

For CUDA systems, the image model uses `float16`. On CPU, it uses `float32`.

---

## 8. Utility Functions

### `cleanup_memory()`

Releases unused Python memory and clears the CUDA cache when a GPU is available.

This is especially important because both the image model and music model can consume significant memory.

---

### `output_filename(prefix, extension)`

Creates unique output filenames using UUID values.

Example:

```text
music_a31f09ab21.wav
frame_0001_e8ab91c100.png
video_4f30ba213c.mp4
```

This prevents generated files from overwriting previous outputs.

---

### `clip_duration(audio_path)`

Uses MoviePy to determine the generated audio duration.

The duration is later used to calculate how many image frames need to be generated.

---

# 9. Music Generation Agent

## Class

```python
MusicGenerationAgent
```

### Responsibility

The agent generates instrumental audio from a text prompt.

### Model

```text
facebook/musicgen-small
```

### Workflow

```text
Prompt
  ↓
AutoProcessor
  ↓
MusicGen
  ↓
Generated Audio Tensor
  ↓
Normalize Audio
  ↓
Convert to 16-bit PCM
  ↓
Save as WAV
```

### Generated Music Prompt

The original user prompt is automatically converted into a music-oriented prompt similar to:

```text
instrumental background music for <USER PROMPT>,
cinematic, atmospheric, no vocals, clean audio
```

### Important Parameters

| Parameter | Description |
|---|---|
| `max_new_tokens` | Controls approximate music generation length |
| `guidance_scale` | Controls how strongly the model follows the music prompt |

---

# 10. Image Generation Agent

## Class

```python
ImageGenerationAgent
```

### Responsibility

Generates the visual frames used to create the final video.

### Initial Frame

The first frame is generated using:

```text
Text Prompt
    ↓
Tiny Stable Diffusion
    ↓
Reference Image
```

### Subsequent Frames

Later frames are generated using the preceding image as an image-to-image reference.

```text
Previous Frame
      +
Reference Frame
      +
Prompt
      ↓
Image-to-Image Diffusion
      ↓
Next Frame
```

This approach attempts to preserve:

- Character identity
- Clothing
- Scene
- Camera angle
- Lighting
- Visual style

while still introducing small changes between frames.

---

## 11. Frame Continuity Strategy

One of the main design features of the project is visual continuity.

Generating every frame independently with text-to-image diffusion would normally result in large visual changes.

Instead, the project:

1. Generates one initial reference frame.
2. Uses each previous frame as the source for the next frame.
3. Blends the previous frame with the original reference frame.
4. Runs image-to-image generation using a low transformation strength.
5. Uses a slightly different seed for every frame.

### Reference Anchoring

The code blends approximately 15% of the first reference image back into each new source image.

Conceptually:

```text
Anchored Image =
85% Previous Frame
+
15% Original Reference Frame
```

This helps reduce long-term visual drift.

---

## 12. Frame Change Strength

The Gradio interface exposes the parameter:

```text
Frame Change Strength
```

Default:

```text
0.22
```

Available range:

```text
0.10 – 0.40
```

### Effect

Lower values:

```text
More visual consistency
Less movement
```

Higher values:

```text
More visual change
More apparent motion
Higher risk of scene drift
```

The application validates that the internal continuity value stays between `0` and `1`.

---

# 13. Video Generation Agent

## Class

```python
VideoGenerationAgent
```

### Responsibility

Combines the generated image frames and generated music into the final MP4 file.

### Workflow

```text
Generated PNG Frames
        +
Generated WAV Audio
        ↓
MoviePy
        ↓
ImageSequenceClip
        ↓
Audio Attachment
        ↓
H.264 + AAC Encoding
        ↓
MP4 Video
```

### Video Encoding

The application uses:

```text
Video Codec: H.264 / libx264
Audio Codec: AAC
Pixel Format: yuv420p
```

`yuv420p` improves compatibility with common media players and browsers.

---

# 14. Media Orchestrator

## Class

```python
MediaOrchestratorAgent
```

This class coordinates the complete generation workflow.

### Execution Sequence

```text
1. Receive main prompt

2. Create:
   - Image-specific prompt
   - Music-specific prompt

3. Load MusicGen

4. Generate music

5. Unload MusicGen

6. Determine generated audio duration

7. Calculate required frame count

8. Load Tiny Stable Diffusion

9. Generate reference image

10. Generate continuous image sequence

11. Unload image model

12. Build video using MoviePy

13. Attach generated music

14. Save final MP4

15. Return:
    - Image gallery
    - Audio
    - Video
```

Loading the music and image models sequentially instead of keeping both models loaded at once helps reduce memory requirements.

---

# 15. Frame Count Calculation

The project generates:

```text
5 images per second
```

The number of frames is calculated using:

```text
frames = ceil(audio_duration × 5)
```

For example:

| Music Duration | Generated Frames |
|---:|---:|
| 2 seconds | 10 |
| 5 seconds | 25 |
| 10 seconds | 50 |
| 20 seconds | 100 |

Therefore, longer generated music significantly increases image-generation time.

---

# 16. Gradio Interface

The application exposes the complete workflow through a Gradio interface.

Default local URL:

```text
http://127.0.0.1:7860
```

The browser opens automatically when the script starts.

---

## User Inputs

### Main Prompt

Describes the desired multimedia scene.

Example:

```text
A futuristic classroom with AI holograms and students learning technology
```

---

### Image Generation Steps

Range:

```text
10 – 40
```

Default:

```text
20
```

Higher values can improve image refinement but require more generation time.

---

### Music Length Tokens

Range:

```text
64 – 512
```

Default:

```text
128
```

The interface describes the rough relationship as approximately:

```text
50 tokens ≈ 1 second
```

Actual generated duration may vary depending on the model.

---

### Music Guidance Scale

Range:

```text
1.0 – 6.0
```

Default:

```text
3.0
```

Controls how strongly MusicGen follows the supplied text description.

---

### Frame Change Strength

Range:

```text
0.10 – 0.40
```

Default:

```text
0.22
```

Controls the amount of visual change between generated frames.

---

### Seed

Default:

```text
42
```

The seed improves repeatability during image generation.

---

# 17. Application Outputs

The Gradio interface returns three outputs.

### 1. Generated Images

Displayed as a gallery.

```text
Generated Images (5 per second)
```

### 2. Generated Background Music

Returned as a playable WAV audio file.

### 3. Final Video

Returned as an MP4 video.

---

# 18. Installation

## Clone the Repository

```bash
git clone https://github.com/kcdchawda/AI-Project.git
cd AI-Project
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

A typical dependency installation command is:

```bash
pip install torch transformers diffusers gradio numpy pillow scipy moviepy accelerate
```

FFmpeg must also be available to MoviePy for video encoding.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

### Windows

Install FFmpeg and ensure that its executable is available through the system `PATH`.

---

# 19. Suggested `requirements.txt`

```text
torch
transformers
diffusers
gradio
numpy
Pillow
scipy
moviepy
accelerate
```

For reproducible deployment, package versions should eventually be pinned after testing.

---

# 20. Running the Project

Run:

```bash
python Tiny.py
```

The application starts on:

```text
http://127.0.0.1:7860
```

The browser should open automatically.

---

# 21. First Run

During the first execution, Hugging Face model files need to be downloaded.

Models used:

```text
segmind/tiny-sd
facebook/musicgen-small
```

Therefore, the first run requires an internet connection.

After the models have been cached locally, subsequent execution can generally reuse the downloaded model files, subject to the Hugging Face cache and environment configuration.

---

# 22. Hardware Considerations

## GPU

CUDA-compatible NVIDIA hardware is preferred.

Advantages:

- Faster diffusion inference
- Faster frame generation
- `float16` image-model inference
- Better performance for longer videos

---

## CPU

The program can select CPU automatically when CUDA is unavailable.

However:

- Image generation will be substantially slower
- Music generation can also be slow
- Longer videos require many diffusion passes
- System RAM usage may become significant

The code deliberately unloads one large model before loading the next to reduce peak memory usage.

---

# 23. Output Directory Structure

Generated files are stored under:

```text
AI-Project/
│
├── Tiny.py
├── output/
│   ├── music_<unique-id>.wav
│   ├── frame_0001_<unique-id>.png
│   ├── frame_0002_<unique-id>.png
│   ├── ...
│   └── video_<unique-id>.mp4
│
└── ...
```

The exact filenames change because UUID-based suffixes are used.

---

# 24. End-to-End Data Flow

```text
USER
 │
 │ Main Prompt
 ▼
GRADIO UI
 │
 ▼
generate_all()
 │
 ▼
MediaOrchestratorAgent
 │
 ├──────────────► Build Music Prompt
 │                    │
 │                    ▼
 │          MusicGenerationAgent
 │                    │
 │                    ▼
 │              MusicGen Small
 │                    │
 │                    ▼
 │                 WAV File
 │                    │
 │                    ▼
 │              Audio Duration
 │                    │
 │                    ▼
 │             Frame Calculation
 │
 ├──────────────► Build Image Prompt
 │                    │
 │                    ▼
 │          ImageGenerationAgent
 │                    │
 │        ┌───────────┴────────────┐
 │        ▼                        ▼
 │  Text-to-Image            Image-to-Image
 │        │                        │
 │        └───────────┬────────────┘
 │                    ▼
 │                PNG Frames
 │
 ▼
VideoGenerationAgent
 │
 ├── PNG Frames
 └── WAV Audio
 │
 ▼
MoviePy + FFmpeg
 │
 ▼
FINAL MP4
```

---

# 25. Key Design Decisions

## Separate AI Agents

The project separates generation logic into:

- Music agent
- Image agent
- Video agent
- Orchestrator agent

This makes the code easier to maintain and allows individual models to be replaced later.

---

## Sequential Model Loading

MusicGen is loaded first and removed before Stable Diffusion is loaded.

This reduces the chance of running out of RAM or VRAM.

---

## Image-to-Image Continuity

Subsequent images are based on the previous image rather than generated independently.

This improves visual continuity across the resulting video.

---

## Reference Frame Anchoring

A small percentage of the original reference frame is continuously blended into later source frames.

This reduces cumulative scene drift.

---

## Unique Filenames

UUID-based file names prevent accidental overwriting of previous outputs.

---

# 26. Error Handling

The Gradio execution wrapper catches generation errors and displays them to the user.

Example cases include:

- Missing prompt
- Model loading failure
- Memory limitations
- Invalid continuity values
- FFmpeg/video encoding failure

If the prompt is blank, the UI raises:

```text
Please enter a prompt first.
```

---

# 27. Current Limitations

### 1. Frame-Based Video Generation

The project does not use a dedicated text-to-video or image-to-video model.

Instead, it creates video from individually generated diffusion frames.

---

### 2. Slow Generation

Five diffusion-generated images are created for every second of output video.

For example:

```text
20-second audio
=
approximately 100 diffusion-generated frames
```

This can be computationally expensive.

---

### 3. Limited Temporal Consistency

Using the previous image as an image-to-image reference improves continuity, but it does not provide the same temporal modeling as dedicated video diffusion models.

Possible issues include:

- facial changes
- object deformation
- flickering
- inconsistent details
- gradual scene drift

---

### 4. Fixed Resolution

The orchestrator currently generates images at:

```text
512 × 512
```

The user interface does not expose resolution controls.

---

### 5. Fixed Frame Rate

The project currently uses:

```text
5 FPS
```

This is much lower than standard playback rates such as 24 or 30 FPS.

The result is closer to an AI-generated animated sequence or slideshow than smooth conventional video.

---

### 6. Local Gradio Binding

The server launches on:

```text
127.0.0.1
```

Therefore, it is intended for local use unless the deployment settings are changed.

---

# 28. Possible Future Improvements

The project can be extended in several directions:

- Replace frame-by-frame diffusion with a dedicated video-generation model
- Add prompt enhancement using an LLM
- Add negative-prompt input to the UI
- Add image resolution selection
- Add aspect-ratio selection
- Add video duration control
- Add FPS configuration
- Add interpolation between generated frames
- Add generated captions
- Add text-to-speech narration
- Add speech + background music mixing
- Add prompt history
- Add output cleanup
- Add model caching controls
- Add progress metrics and runtime estimates
- Add Hugging Face Spaces deployment support
- Add Docker deployment
- Add GPU cloud deployment
- Add REST API / FastAPI backend
- Add job queue for multiple users
- Store generation metadata
- Add authentication
- Add persistent project history

---

# 29. Suggested Production Architecture

For a larger version of the project:

```text
Frontend
   │
   ▼
Gradio / React
   │
   ▼
FastAPI Backend
   │
   ▼
Job Queue
   │
   ├────────► Music Generation Worker
   │
   ├────────► Image / Video Generation Worker
   │
   └────────► Encoding Worker
   │
   ▼
Object Storage
   │
   ▼
Generated Media URL
```

Possible infrastructure:

```text
FastAPI
Redis
Celery / RQ
Docker
CUDA GPU Worker
S3-compatible storage
PostgreSQL
Nginx
```

---

# 30. Recommended Project Structure

As the application grows, the current single-file implementation can be divided into modules:

```text
AI-Project/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── music_agent.py
│   ├── image_agent.py
│   ├── video_agent.py
│   └── orchestrator.py
│
├── utils/
│   ├── __init__.py
│   ├── memory.py
│   └── files.py
│
├── output/
│
└── tests/
    ├── test_music.py
    ├── test_images.py
    └── test_video.py
```

This structure improves maintainability, testing, and deployment readiness.

---

# 31. Summary

The **AI Image + Music + Video Generator** is a multimodal Generative AI application that transforms a text prompt into an AI-generated audiovisual experience.

Its pipeline combines:

```text
MusicGen
+
Tiny Stable Diffusion
+
Image-to-Image Continuity
+
MoviePy
+
Gradio
```

The application demonstrates several useful AI engineering concepts:

- Multimodal model orchestration
- Generative AI pipelines
- Resource-aware model loading
- Diffusion-based image generation
- Music generation
- Image-to-image temporal approximation
- Automated media composition
- Interactive AI application development

Although it is currently a local prototype, its modular design provides a useful foundation for building a more advanced AI media-generation platform.

---

## Source Code

GitHub repository:

```text
https://github.com/kcdchawda/AI-Project
```

Main implementation:

```text
Tiny.py
```
