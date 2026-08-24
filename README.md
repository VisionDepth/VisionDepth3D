<p align="center">
  <img width="800" height="450" alt="NewVD3D-Logo" src="https://github.com/user-attachments/assets/5dbaaaed-0a09-496a-aaec-5072b1cdce70" />
  <br>
</p>

<h2 align="center">The All-in-One 3D Suite for Creators</h2>

<p align="center">
  <em>This is Hybrid 3D.<br>
  Built from AI depth + custom stereo logic —<br>
  Designed for cinema in VR.</em>
</p>
<h3 align="center">
  <a href="https://github.com/VisionDepth/VisionDepth3D/releases">
    <img src="https://img.shields.io/badge/historical%20downloads-26K%2B-brightgreen" alt="26K+ Historical Downloads">
  </a>
  <a href="https://github.com/VisionDepth/VisionDepth3D/releases">
    <img src="https://img.shields.io/github/downloads/VisionDepth/VisionDepth3D/total.svg" alt="Current GitHub Release Downloads">
  </a>
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python Version">
  <img src="https://img.shields.io/github/stars/VisionDepth/VisionDepth3D?style=social" alt="GitHub Stars">
</h3>

<p align="center">
  <em>Click to download or support the project 💙</em>
  <br><br>
  <a href="https://visiondepth3d.itch.io/visiondepth3d" target="_blank" rel="noopener">
    <img src="assets/widget-preview.png"
         alt="Download VisionDepth3D on Itch.io"
         width="208" height="167"
         style="border-radius: 8px;">
  </a>
</p>

<p align="center">
  <a href="https://visiondepth.github.io/VisionDepth3D/" target="_blank" rel="noopener">
    <strong>Official website out now →</strong>
  </a>
</p>

---

## Table of Contents

- [Notice](#notice)
- [Free vs Pro](#free-vs-pro)
- [All-in-One 3D Suite](#all-in-one-3d-suite)
- [Depth Estimation](#depth-estimation-ai-depth-engine)
- [Depth Blender](#depth-blender-multi-source-depth-fusion)
- [FPS / Upscale Enhancer](#fps--upscale-enhancer-rife--real-esrgan)
- [Live 3D](#live-3d--realtime-stereo)
- [Output Formats](#output-formats--aspect-ratios)
- [Official Depth Model List](#official-depth-model-list)
- [Install and Update Guide](#install-and-update-guide)
- [Documentation](#documentation)
- [Legal and Third-Party Notices](#legal-and-third-party-notices)
- [Acknowledgments & Credits](#acknowledgments--credits)
- [Dev Notes](#dev-notes)

---

# Notice

VisionDepth3D is now distributed through official installer builds.

This repository serves as the public release, documentation, legal notice, installer download, issue tracking, and bug reporting page for VisionDepth3D.

Current protected application source code is maintained privately and is no longer distributed as public source.

VisionDepth3D is licensed under a proprietary, no-derivatives license. Forking, copying, redistributing, modifying, repackaging, publishing, or creating derivative works from VisionDepth3D or protected application files is not permitted without express written permission from Johnathan Carpenter / VisionDepth.

Older public source snapshots, forks, or archived copies do not represent current official VisionDepth3D builds. Only installer builds released through the official VisionDepth3D release channels should be considered current, supported, and official.

VisionDepth3D may use, integrate, download, reference, or interoperate with third-party tools, libraries, frameworks, AI models, and model weights. Third-party components remain owned by their respective creators, authors, organizations, and rights holders, and may be subject to their own separate license terms.

---

# Free vs Pro

VisionDepth3D is available in **Free** and **Pro** tiers.

**VisionDepth3D Pro is a one-time $59.99 license unlock. There is no subscription.**

The Free tier is designed for testing, evaluation, casual use, and exploring the core VisionDepth3D workflow before upgrading.

## VisionDepth3D Free

Free includes the core local 2D-to-3D workflow:

- Single video 3D conversion up to **3 minutes**
- Single image 3D conversion
- Single video and image depth generation
- Half-SBS, Full-SBS, and Anaglyph output
- Basic 3D Assistant controls
- Basic depth previews
- Basic Depth Blender image workflow
- RIFE frame interpolation up to **2x**
- Frame extraction
- Local/private desktop processing

Free exports are limited to **1080p output height** and include a **VisionDepth3D watermark**.

Some advanced models, batch workflows, rendering formats, and production tools are reserved for Pro.

## VisionDepth3D Pro

VisionDepth3D Pro unlocks the full individual desktop production workflow:

- **No fixed video length limit**
- **No Free-tier watermark**
- **No fixed output height limit**
- Full 3D Generator controls and advanced stereo tuning
- Advanced 3D keyframes
- VR and VR180 output
- Passive interlaced output
- Live 3D
- Batch video and image-folder processing
- Advanced depth inference controls
- Expanded official depth model access
- Full Depth Blender video and frame-folder workflows
- High FPS RIFE interpolation
- AI Super-Resolution and upscaling workflows
- Scene detection and extraction
- Accelerated RIFE + SR processing
- Advanced encoding controls
- 4K and higher-resolution workflows
- Job queue support

**Pro: $59.99 USD one-time**

Pro licenses may be used by individual creators for professional and commercial work, provided they have the necessary rights to the media being processed and comply with applicable third-party licenses.

A Pro license unlocks VisionDepth3D software features. It does not transfer ownership or licensing rights for third-party AI models, libraries, FFmpeg, media, characters, films, games, music, images, or other third-party content.

Team, studio, company-wide, resale, redistribution, hosted-service, or custom business use may require separate written permission or licensing.

---

# All-in-One 3D Suite

<h3 align="center">3D Generator / Stereo Composer</h3>

<p align="center">
  <img width="700" height="400" alt="3D Generator Tab" src="https://github.com/user-attachments/assets/31541274-90e3-485e-9f3d-d56730e715e8" />
  <br>
  <em>3D Generator Tab</em>
</p>

- **Multiple stereo rendering methods**, including:
  - **Subpixel Z-Splat Renderer (New in v5.1)** for more advanced depth-aware stereo projection, smoother subpixel displacement, and improved handling around depth boundaries and occlusions.
  - **Classic Stereo Warp** using GPU-accelerated per-pixel, depth-aware parallax shifting.
- Built on the [**VisionDepth3D Method**](VisionDepth3D_Method.md), including:
  - Depth shaping and pop controls
  - Subject-anchored convergence
  - Scene-aware stereo scaling
  - Edge-aware masking and feathering
  - Floating-window edge protection
  - Occlusion healing and edge repair
- **Simple 3D Assistant** with ready-to-use styles and adjustable controls for:
  - 3D strength
  - Foreground pop-out
  - Viewing comfort
  - Stereo stability
  - Screen depth
  - Subject screen plane
- **Live preview and diagnostics**:
  - Anaglyph preview
  - Side-by-side preview
  - Heatmaps
  - Edge/mask inspection
  - Stereo difference views
- **Clip-range rendering** for testing difficult scenes before committing to full renders.
- **FFmpeg encoding pipeline** with CPU and hardware encoders when available.

---

# Depth Estimation (AI Depth Engine)

<p align="center">
  <img width="700" height="400" alt="Depth Estimation Tab" src="https://github.com/user-attachments/assets/bdee3d2b-43f6-4a05-9558-90ed08400353" />
  <br>
  <em>Depth Estimation Tab</em>
</p>

VisionDepth3D includes a flexible AI depth pipeline for generating, refining, and preparing depth maps for stereoscopic 3D rendering.

- **Multiple AI depth models**, including:
  - Depth Anything V1
  - Depth Anything V2
  - Distill Any Depth
  - Video Depth Anything
  - Depth Anything 3 (DA3)
- **One-click model switching** with automatic local model caching
- Support for **PyTorch, TorchHub, Diffusers, and ONNXRuntime** inference backends
- **Image, video, and frame-folder depth generation**
- **Video-aware depth estimation** for improved temporal consistency across moving footage
- **Depth normalization and temporal stabilization** to reduce:
  - Depth breathing
  - Flicker
  - Sudden depth-range changes
  - Frame-to-frame instability
- **Built-in depth previews and colormaps** for inspecting generated depth before rendering
- Automatic **resolution matching, shape validation, codec probing, and safe fallbacks**

---

# Depth Blender (Multi-Source Depth Fusion)

<p align="center">
  <img width="700" height="400" alt="Depth Blender Tab" src="https://github.com/user-attachments/assets/44ac6910-6ea0-43fb-b3d7-a338638f33fb" />
  <br>
  <em>Depth Blender Tab</em>
</p>

VisionDepth3D includes a dedicated depth fusion workflow for combining two independently generated depth sources into a cleaner and more balanced final depth map.

- **Blend two depth sources** into a single refined depth map or depth video.
- Supports:
  - Two depth video sources
  - Two PNG depth-frame folders
- **Adjustable blend weighting** for controlling how much influence each depth source contributes.
- **Live preview and frame scrubber** for comparing the blended result before processing.
- **Edge-focused blending controls** to preserve important object boundaries and depth transitions.
- **CLAHE contrast shaping** for improving local depth separation and recovering flatter regions.
- **Bilateral edge-preserving denoise** for smoothing noisy depth while retaining important structure.
- **Depth normalization** to keep the blended result aligned with the original depth scale.
- Helps combine the strengths of different AI depth models while reducing weaknesses or inconsistencies from any single model.
- Output can be used directly as the depth source for the **3D Generator / Stereo Composer**.

---

# FPS / Upscale Enhancer (RIFE + AI Super-Resolution)

<p align="center">
  <img width="700" height="400" alt="FPS / Upscale Enhancer Tab" src="https://github.com/user-attachments/assets/7df0c7ee-c710-42a6-860b-5d2822936db1" />
  <br>
  <em>FPS / Upscale Enhancer Tab</em>
</p>

VisionDepth3D v5.1 introduces a major overhaul of the FPS and upscaling pipeline, with faster RIFE interpolation, expanded AI super-resolution support, improved batching, and more control over performance and output quality.

- **Modern RIFE frame interpolation**, including:
  - RIFE 4.25 Lite for faster high-resolution and VR processing
  - RIFE 4.25 for higher-quality interpolation
  - Legacy RIFE ONNX models for compatibility
- **2×, 4×, and 8× FPS generation**
- Adjustable **RIFE backend, scale, precision, and timestep batching**
- Automatic high-resolution RIFE scaling for large Full-SBS and multi-megapixel sources
- **AI Super-Resolution pipeline** with:
  - RealESR
  - Real-ESRGAN
  - BSRGAN
  - Automatic fast and quality model selection
- Automatic discovery of compatible local **ONNX super-resolution models**, including SPAN-style exports
- **Scale-aware SR processing** that can automatically choose an appropriate model and input resolution for the requested output
- Adjustable:
  - Upscale backend
  - FP16 / FP32 precision
  - Batch size
  - Input resolution percentage
  - AI blending
- Built-in **SR benchmarking** for testing the selected model and runtime settings
- **Single-video and batch-folder workflows**
- Built-in **scene detection and scene extraction** for processing longer videos in manageable sections
- Optional **merged and accelerated RIFE + SR processing** when both stages are enabled
- **VR-optimized final merge** with constant-frame-rate output options for smoother headset playback
- Automatic output naming for RIFE and upscale passes
- Original audio preservation and flexible FFmpeg encoding
- Detailed **progress, FPS, ETA, profiling, logs, pause/resume, and safe cancellation**

---

# Live 3D / Realtime Stereo

<p align="center">
  <img width="700" height="400" alt="Live 3D Tab" src="https://github.com/user-attachments/assets/048e6313-ac57-4c2e-afba-a9d9a0711e3f" />
  <br>
  <em>Live 3D Tab</em>
</p>

The Live 3D tab is a realtime stereo sandbox for testing depth models, stereo controls, screen capture, and live 2D-to-3D conversion before final rendering.

- Camera / capture-card input
- Screen capture input
- Configurable capture resolution and FPS
- Depth model selection
- Foreground, midground, and background shift controls
- Parallax balance
- Subject tracking
- Dynamic convergence
- Edge masking
- Feathering
- Floating-window support
- SBS preview workflow

---

# Smart GUI + Workflow

VisionDepth3D is built as a complete local desktop workflow, with tools designed to keep long renders and multi-stage processing manageable.

- Multi-tab desktop interface
- Persistent project and processing settings
- Pause, resume, and cancel controls for long-running jobs
- Multi-language UI support
- CPU and hardware encoding options
- Batch and queue-oriented workflows
- Local/private processing without uploading source media to the cloud

---

# Output Formats & Aspect Ratios

- Stereo formats: **Half-SBS, Full-SBS, VR180, Anaglyph, Passive Interlaced**
- Aspect ratios: **16:9, 2.39:1, 2.76:1, 4:3, 21:9, 1:1, 2.35:1**
- Containers: **MP4, MKV, AVI**
- Encoders: CPU + FFmpeg hardware options, including NVENC, AMF, and QSV when available

---

# Official Depth Model List

The list below reflects the current official Free/Pro model registry for VisionDepth3D v5.0.

Models marked as non-commercial, research-only, CC-BY-NC, missing-license, or review-only are not included as official Free/Pro commercial-safe features.

## Free Tier Models

| Tier | Model | Backend | License | Repository / ID |
|---|---|---|---|---|
| Free + Pro | **Depth Anything v2 Small** | `hf` | Apache-2.0 | [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| Free + Pro | **Distill-Any-Depth Small (xingyang1)** | `hf` | MIT | [`xingyang1/Distill-Any-Depth-Small-hf`](https://huggingface.co/xingyang1/Distill-Any-Depth-Small-hf) |
| Free + Pro | **Video Depth Anything Small** | `vda` | Apache-2.0 | [`depth-anything/Video-Depth-Anything-Small`](https://huggingface.co/depth-anything/Video-Depth-Anything-Small) |
| Free + Pro | **Marigold Depth v1.0** | `diffusers` | Apache-2.0 | [`prs-eth/marigold-depth-v1-0`](https://huggingface.co/prs-eth/marigold-depth-v1-0) |

## Pro Tier Models

| Tier | Model | Backend | License | Repository / ID |
|---|---|---|---|---|
| Pro | **Depth Anything v2 Small(ONNX)** | `onnx` | Apache-2.0 | [`onnx-community/depth-anything-v2-small-ONNX`](https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX) |
| Pro | **Distill-Any-Depth Large (xingyang1)** | `hf` | MIT | [`xingyang1/Distill-Any-Depth-Large-hf`](https://huggingface.co/xingyang1/Distill-Any-Depth-Large-hf) |
| Pro | **Distill-Any-Depth Small(ONNX)** | `onnx` | MIT | [`FuryTMP/Distill-Any-Depth-Small-onnx`](https://huggingface.co/FuryTMP/Distill-Any-Depth-Small-onnx) |
| Pro | **Distill-Any-Depth Base(ONNX)** | `onnx` | MIT | [`FuryTMP/Distill-Any-Depth-Base-onnx`](https://huggingface.co/FuryTMP/Distill-Any-Depth-Base-onnx) |
| Pro | **Distill-Any-Depth Large(ONNX)** | `onnx` | MIT | [`FuryTMP/Distill-Any-Depth-Large-onnx`](https://huggingface.co/FuryTMP/Distill-Any-Depth-Large-onnx) |
| Pro | **Video Depth Anything (ONNX)** | `onnx` | Apache-2.0 | [`FuryTMP/Video-Depth-Anything-L-ONNX-512x288`](https://huggingface.co/FuryTMP/Video-Depth-Anything-L-ONNX-512x288) |
| Pro | **DA3-SMALL** | `da3` | Apache-2.0 | [`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL) |
| Pro | **DA3-BASE** | `da3` | Apache-2.0 | [`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE) |
| Pro | **DA3MONO-LARGE** | `da3` | Apache-2.0 | [`depth-anything/DA3MONO-LARGE`](https://huggingface.co/depth-anything/DA3MONO-LARGE) |
| Pro | **DA3METRIC-LARGE** | `da3` | Apache-2.0 | [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE) |
| Pro | **DA3-LARGE-1.1** | `da3` | Apache-2.0 | [`depth-anything/DA3-LARGE-1.1`](https://huggingface.co/depth-anything/DA3-LARGE-1.1) |
| Pro | **Depth Anything v2 Metric Outdoor (Large)** | `hf` | Apache-2.0 | [`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf) |
| Pro | **Depth Anything v1 Small** | `hf` | Apache-2.0 | [`LiheYoung/depth-anything-small-hf`](https://huggingface.co/LiheYoung/depth-anything-small-hf) |
| Pro | **Depth Anything v1 Base** | `hf` | Apache-2.0 | [`LiheYoung/depth-anything-base-hf`](https://huggingface.co/LiheYoung/depth-anything-base-hf) |
| Pro | **Depth Anything v1 Large** | `hf` | Apache-2.0 | [`LiheYoung/depth-anything-large-hf`](https://huggingface.co/LiheYoung/depth-anything-large-hf) |
| Pro | **ZoeDepth N (NYU)** | `zoe` | MIT | `ZoeD_N` |
| Pro | **ZoeDepth K (KITTI)** | `zoe` | MIT | `ZoeD_K` |
| Pro | **ZoeDepth NK (Recommended)** | `zoe` | MIT | `ZoeD_NK` |
| Pro | **DPT BEiT Large 512** | `hf` | MIT | [`Intel/dpt-beit-large-512`](https://huggingface.co/Intel/dpt-beit-large-512) |
| Pro | **Prompt Depth Anything VITS Transparent** | `hf` | Apache-2.0 | [`depth-anything/prompt-depth-anything-vits-transparent-hf`](https://huggingface.co/depth-anything/prompt-depth-anything-vits-transparent-hf) |

Users are responsible for following the license terms of any third-party model, library, or tool they download or use.

---

# Install and Update Guide

### Recommended Install Method

VisionDepth3D is now installed through the official **VisionDepth3D Setup Hub**.


To install VisionDepth3D:

1. Go to the official VisionDepth3D Itch.io page:

   [Download VisionDepth3D on Itch.io](https://visiondepth3d.itch.io/visiondepth3d)

2. Download the **VisionDepth3D Setup Hub**.

3. Open the Setup Hub.

4. In the release dropdown, choose the build you want to install:

   * **CUDA build** for NVIDIA GPU systems
   * **DirectML build** for supported AMD / Intel GPU systems

5. Choose where the installer files should be downloaded.

   The selected folder is where the installer `.exe` and required `.bin` files will be saved.

6. Keep **Launch INNO Installer after Download** enabled if you want the Setup Hub to automatically start the installer after the download finishes.

7. Click **Install**.

8. Finish the Inno Setup installer process.

9. Launch VisionDepth3D from the Desktop shortcut or Start Menu.

After installation is complete, you do not need to run the Setup Hub again unless you are updating, uninstalling, or reinstalling VisionDepth3D.

---

### System Requirements

* Windows recommended
* NVIDIA GPU recommended for best performance
* CUDA-capable GPU recommended for AI depth generation, upscale, interpolation, and video workflows
* AMD / Intel GPU support may vary depending on DirectML support
* CPU fallback available, but much slower
* FFmpeg is used for video/audio processing
* Internet connection required for downloading installers and supported model files

---

### CUDA vs DirectML Builds

Choose the build that matches your system.

#### CUDA Build

Use the CUDA build if you have an NVIDIA GPU.

This is the recommended build for best performance with:

* AI depth generation
* 3D rendering workflows
* Frame interpolation
* Upscaling
* Long video processing

#### DirectML Build

Use the DirectML build if you have a supported AMD or Intel GPU on Windows.

DirectML support may vary depending on your GPU, drivers, Windows version, and model compatibility. Some models may run slower or fall back to CPU.

---

## Updating VisionDepth3D

Updating VisionDepth3D uses the same Setup Hub workflow.

To update:

1. Open the **VisionDepth3D Setup Hub**.

2. Click **Uninstall VD3D**.

   This runs the installed VisionDepth3D uninstaller.

3. After uninstalling the old version, choose the latest CUDA or DirectML build from the release dropdown.

4. Choose the download folder for the installer `.exe` and required `.bin` files.

5. Keep **Launch INNO Installer after Download** enabled if you want the Setup Hub to automatically start the installer.

6. Click **Install**.

7. Finish the Inno Setup installer process.

8. Launch VisionDepth3D from the Desktop shortcut or Start Menu.

---

### Cleaning Up Setup Files

After VisionDepth3D has been installed successfully, the downloaded installer files are no longer needed.

You can click:

```text
Delete setup files
```

inside the Setup Hub to remove the downloaded installer `.exe` and required `.bin` files.

This only removes the temporary setup files downloaded by the Setup Hub. It does not remove your installed VisionDepth3D app.

---

### Notes About Weights, Presets, and Pro Activation

VisionDepth3D may create folders such as:

```text
weights/
presets/
cache/
```

These folders are used for downloaded models, presets, cached files, and runtime data.

Pro activation is stored locally after successful license activation. If you reinstall, reset your system, or move to another computer, you may need to reactivate depending on your license state.

---

---

# Documentation

VisionDepth3D includes guides and workflow documentation.

Start here:

- [UserGuide.md](UserGuide.md)
- [VisionDepth3D Method](VisionDepth3D_Method.md)
- `legal/MODEL_LICENSE_NOTICE.md`
- `legal/THIRD_PARTY_MODEL_ACKNOWLEDGMENTS.md`

---

# Legal and Third-Party Notices

The root `legal/` folder should be included with official builds.

Recommended structure:

```text
legal/
  README.md
  THIRD_PARTY_MODEL_ACKNOWLEDGMENTS.md
  MODEL_LICENSE_NOTICE.md
  ABOUT_LEGAL_BLURB.md
```

These files explain:

* third-party model acknowledgments
* model license notes
* official Free/Pro model policy
* excluded non-commercial or review-only model categories
* FFmpeg and third-party tool notices

VisionDepth3D does not claim ownership of third-party AI models, model weights, libraries, frameworks, or external tools.

Some repositories use one license for source code and a different license for model weights. VisionDepth3D treats the model weight license separately from the source-code license.

VisionDepth3D uses FFmpeg for video/audio processing through subprocess calls. FFmpeg licensing depends on how the FFmpeg binary was built. Users and distributors should follow the applicable FFmpeg license terms for the binary included or used with VisionDepth3D.

---

# Acknowledgments & Credits

Thank you to the researchers, developers, and open-source contributors behind the depth estimation models, video tools, AI libraries, and multimedia frameworks that make projects like VisionDepth3D possible.


Special thanks to the creators and maintainers of:

- Depth Anything
- Distill-Any-Depth
- Video Depth Anything
- Marigold
- ZoeDepth
- DPT / BEiT / MiDaS-related research
- ONNXRuntime
- PyTorch
- Hugging Face
- Diffusers
- FFmpeg
- RIFE
- Real-ESRGAN

---

# Dev Notes

VisionDepth3D is developed by a solo developer and continues to grow through testing, user feedback, bug reports, and community support.

Pro helps support continued development, better testing hardware, more GPU/backend coverage, future Mac/Linux/AMD/Intel improvements, and ongoing 3D pipeline research.
