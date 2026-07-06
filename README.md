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
    <img src="https://img.shields.io/badge/historical%20downloads-23K%2B-brightgreen" alt="23K+ Historical Downloads">
  </a>
  <a href="https://github.com/VisionDepth/VisionDepth3D/releases">
    <img src="https://img.shields.io/github/downloads/VisionDepth/VisionDepth3D/total.svg" alt="Current GitHub Release Downloads">
  </a>
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python Version">
  <img src="https://img.shields.io/github/stars/VisionDepth/VisionDepth3D?style=social" alt="GitHub Stars">
</h3>

<p align="center">
  <em style="font-size: 14px; color: #888;">
    Click to download or support the project 💙
  </em>
  <br>

  <a href="https://visiondepth3d.itch.io/visiondepth3d" target="_blank" rel="noopener">
    <img src="assets/widget-preview.png"
         alt="Download VisionDepth3D on Itch.io"
         width="208" height="167"
         style="border-radius: 8px; margin-top: 6px;">
  </a>
  <br>
  
  <p align="center">
	  <a href="https://visiondepth.github.io/VisionDepth3D/" target="_blank" rel="noopener"
	     style="display:inline-block; margin-top:8px; padding:10px 16px; border-radius:12px;
	            border:1px solid #39c6ff; background:linear-gradient(180deg, rgba(57,198,255,.15), rgba(57,198,255,.08));
	            color:#39c6ff; font-weight:700; text-decoration:none;">
	    Official website out now →
	  </a>
	</p>


---

# Notice

VisionDepth3D is now distributed through official installer builds.

This repository serves as the public release, documentation, legal notice, installer download, issue tracking, and bug reporting page for VisionDepth3D.

Current protected application source code is maintained privately and is no longer distributed as public source.

VisionDepth3D is licensed under a proprietary, no-derivatives license.

Forking, copying, redistributing, modifying, repackaging, publishing, or creating derivative works from VisionDepth3D or any protected application files is not permitted without express written permission from Johnathan Carpenter / VisionDepth.

Older public source snapshots, forks, or archived copies do not represent current official VisionDepth3D builds. Only installer builds released through the official VisionDepth3D release channels should be considered current, supported, and official.

---

## Table of Contents

- [Free vs Pro](#free-vs-pro)
- [All-in-One 3D Suite](#all-in-one-3d-suite)
- [Depth Estimation](#depth-estimation-ai-depth-engine)
- [Depth Blender](#depth-blender-multi-source-depth-fusion)
- [FPS / Upscale Enhancer](#fps--upscale-enhancer-rife--real-esrgan)
- [Live 3D](#live-3d--realtime-stereo)
- [Output Formats](#output-formats--aspect-ratios)
- [Official Depth Model List](#official-depth-model-list)
- [Install Guide](#guide-sheet-install)
- [Updating VisionDepth3D](#guide-sheet-updating-visiondepth3d)
- [Legal and Third-Party Notices](#legal-and-third-party-notices)
- [Acknowledgments & Credits](#acknowledgments--credits)

---

# Free vs Pro

VisionDepth3D uses a **Free** and **Pro** tier structure.

The Free tier lets users try the core VisionDepth3D workflow, including single video/image conversion, depth generation, basic 3D output formats, preview tools, and limited FPS interpolation.

The Pro tier unlocks the full production workflow, including unlimited render length, no watermark, batch processing, advanced keyframes, advanced encoding, Live 3D, VR/VR180 output, full Depth Blender workflows, Real-ESRGAN upscaling, high FPS multipliers, 4K output, job queue support, and the expanded official model list.

| Feature / Workflow                  |              Free |                     Pro |
| ----------------------------------- | ----------------: | ----------------------: |
| 3D Generator tab                    |                 ✅ |                       ✅ |
| Depth Engine tab                    |                 ✅ |                       ✅ |
| Depth Blender tab                   |         ✅ Limited |                  ✅ Full |
| FPS / Upscale tab                   |         ✅ Limited |                  ✅ Full |
| Live 3D tab                         |         🔒 Locked |                       ✅ |
| Single video 3D conversion          | ✅ Up to 3 minutes |             ✅ Unlimited |
| Single image 3D conversion          |                 ✅ |                       ✅ |
| Video length limit                  |         3 minutes |          No fixed limit |
| Output height limit                 |       Up to 1080p |          No fixed limit |
| Free-tier watermark                 |         ✅ Applies |          ❌ No watermark |
| Half-SBS output                     |                 ✅ |                       ✅ |
| Full-SBS output                     |                 ✅ |                       ✅ |
| Anaglyph output                     |                 ✅ |                       ✅ |
| VR / VR180 output                   |         🔒 Locked |                       ✅ |
| Passive interlaced output           |         🔒 Locked |                       ✅ |
| Batch video conversion              |         🔒 Locked |                       ✅ |
| Image-folder conversion             |         🔒 Locked |                       ✅ |
| Advanced keyframes                  |         🔒 Locked |                       ✅ |
| Advanced encoding controls          |         🔒 Locked |                       ✅ |
| Depth single image                  |                 ✅ |                       ✅ |
| Depth single video                  | ✅ Up to 3 minutes |             ✅ Unlimited |
| Depth image-folder batch            |         🔒 Locked |                       ✅ |
| Depth video-folder batch            |         🔒 Locked |                       ✅ |
| Depth preview samples               |                 ✅ |                       ✅ |
| Advanced depth inference controls   |         🔒 Locked |                       ✅ |
| Basic Depth Blender image workflow  |                 ✅ |                       ✅ |
| Depth Blender frame-folder workflow |         🔒 Locked |                       ✅ |
| Depth Blender video workflow        |         🔒 Locked |                       ✅ |
| GPU Depth Blender processing        |                 ✅ |                       ✅ |
| Extract frames                      |                 ✅ |                       ✅ |
| Basic RIFE FPS interpolation        |        ✅ Up to 2x |                       ✅ |
| High FPS multipliers                |         🔒 Locked |                       ✅ |
| Scene detection                     |         🔒 Locked |                       ✅ |
| Threaded FPS pipeline               |         🔒 Locked |                       ✅ |
| Real-ESRGAN upscale                 |         🔒 Locked |                       ✅ |
| 4K output workflows                 |         🔒 Locked |                       ✅ |
| Job queue                           |         🔒 Locked |                       ✅ |
| Expanded official Pro model list    |         🔒 Locked |                       ✅ |
| License activation                  |      Not required | Required for Pro unlock |
| Local/private desktop processing    |                 ✅ |                       ✅ |

## What Free Includes

VisionDepth3D Free includes the core local desktop workflow for testing and personal evaluation.

Free includes:

* Single video 3D conversion up to 3 minutes
* Single image 3D conversion
* Single video and single image depth generation
* Half-SBS, Full-SBS, and Anaglyph output
* Basic 3D Assistant / guided controls
* Basic depth preview tools
* Basic Depth Blender image workflow
* Basic RIFE FPS interpolation up to 2x
* Frame extraction
* GPU Depth Blender processing
* Local desktop processing without uploading your videos or images to the cloud

Free exports include a VisionDepth3D watermark and are limited to 1080p output height.

## What Pro Unlocks

VisionDepth3D Pro is a one-time license unlock for the full desktop production workflow.

Pro unlocks:

* Unlimited video length with no Free-tier render limit
* No Free-tier watermark
* No fixed output height limit
* Batch video conversion
* Batch image-folder workflows
* Depth video-folder batch processing
* Depth image-folder batch processing
* Advanced keyframes
* Advanced encoding controls
* Advanced depth inference controls
* VR and VR180 output
* Passive interlaced output
* Live 3D
* Full Depth Blender frame-folder and video workflows
* Threaded FPS processing
* Scene detection
* Real-ESRGAN upscaling
* High FPS multipliers
* 4K output workflows
* Job queue support
* Expanded official Pro model list
* Local desktop processing without uploading your videos or images to the cloud

VisionDepth3D Pro is sold as access to the VisionDepth3D desktop application and Pro software features. It does not grant ownership of third-party AI models or override the original licenses of third-party models, libraries, or tools.

## Third-Party Model Notice

VisionDepth3D uses and integrates third-party AI models, libraries, and tools. Each third-party model remains under its original license and terms.

The Free and Pro tiers control access to VisionDepth3D software features and official model integrations inside the app. They do not change the license of any third-party model.

For details, see the files in the `legal/` folder:

```text
legal/README.md
legal/THIRD_PARTY_MODEL_ACKNOWLEDGMENTS.md
legal/MODEL_LICENSE_NOTICE.md
legal/ABOUT_LEGAL_BLURB.md
```


---

<h2 align="center">All-in-One 3D Suite</h2>

<h3 align="center">3D Generator / Stereo Composer</h3>

<p align="center">
  <img width="700" height="400" alt="3D Generator Tab" src="https://github.com/user-attachments/assets/31541274-90e3-485e-9f3d-d56730e715e8" />
  <br>
  <em>3D Generator Tab</em>
</p>

- **GPU-accelerated stereo warping** using per-pixel, depth-aware parallax shifting.
- Built on the [**VisionDepth3D Method**](VisionDepth3D_Method.md), including:
  - Depth shaping and pop controls
  - Subject-anchored convergence
  - Scene-aware stereo scaling
  - Edge-aware masking and feathering
  - Floating-window edge protection
  - Occlusion healing and edge repair
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

VisionDepth3D includes a flexible AI depth engine for generating depth maps from images, videos, and frame folders.

- Official Free and Pro model filtering through `config/model_registry.json`
- One-click model switching with local caching
- PyTorch, TorchHub, Diffusers, and ONNXRuntime backends
- Image and video depth generation
- Optional high-precision depth output when supported
- Built-in preview modes and colormaps
- Depth normalization tools to reduce depth breathing and flicker
- Resolution handling, shape checks, codec probing, and safe fallbacks

---

# Depth Blender (Multi-Source Depth Fusion)

<p align="center">
  <img width="700" height="400" alt="Depth Blender Tab" src="https://github.com/user-attachments/assets/44ac6910-6ea0-43fb-b3d7-a338638f33fb" />
  <br>
  <em>Depth Blender Tab</em>
</p>

- Blend two depth sources into one cleaner depth map or depth video.
- Pair two PNG frame folders or two depth videos.
- Live preview and scrubber.
- Edge-focused blend controls.
- CLAHE contrast shaping.
- Bilateral edge-preserving denoise.
- Normalization back to base for consistent depth scale.

---

# FPS / Upscale Enhancer (RIFE + Real-ESRGAN)

<p align="center">
  <img width="700" height="400" alt="FPS / Upscale Enhancer Tab" src="https://github.com/user-attachments/assets/7df0c7ee-c710-42a6-860b-5d2822936db1" />
  <br>
  <em>FPS / Upscale Enhancer Tab</em>
</p>

- RIFE interpolation through ONNX.
- 2×, 4×, and 8× FPS generation.
- Real-ESRGAN upscaling through ONNX.
- Optional FP16 acceleration where supported.
- Merged and threaded processing modes.
- Scene splitting for long videos.
- Progress, FPS, ETA, logs, and safe cancel handling.

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

- Multi-tab desktop interface
- Persistent settings
- Pause, resume, and cancel for long GPU jobs
- Multi-language UI support
- Hardware encoding options
- Queue-oriented workflows
- Local/private processing

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

## Model Licensing Notice

VisionDepth3D does not claim ownership of third-party AI models.

Some repositories use one license for source code and a different license for model weights. VisionDepth3D treats the **model weight license** separately from the source-code license.

Official VisionDepth3D Free and Pro builds avoid listing models as official features when the model or weights are marked as:

- CC-BY-NC
- non-commercial
- research-only
- missing-license
- unclear commercial usage rights
- review-only

Users are responsible for following the license terms of any third-party model, library, or tool they download or use.

---

## Guide Sheet: Install

### Recommended Install Method

VisionDepth3D is now installed through the official **VisionDepth3D Setup Hub**.

The source code is no longer distributed as the recommended public installation method. This change helps protect the project from unauthorized edits, forks, redistribution, and modified builds.

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

## Guide Sheet: Updating VisionDepth3D

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

## Source Code Notice

VisionDepth3D is no longer distributed publicly as a source-code installation.

Public releases are provided through the official Setup Hub and installer builds. This helps protect the project from unauthorized modification, redistribution, forks, and unofficial builds.

VisionDepth3D remains licensed under its proprietary no-derivatives license. Forking, redistributing, modifying, or creating derivative works is not permitted unless explicitly authorized by the developer.

---

## Documentation

VisionDepth3D includes guides and workflow documentation.

Start here:

- [UserGuide.md](UserGuide.md)
- [VisionDepth3D Method](VisionDepth3D_Method.md)
- `legal/MODEL_LICENSE_NOTICE.md`
- `legal/THIRD_PARTY_MODEL_ACKNOWLEDGMENTS.md`

---

## Legal and Third-Party Notices

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

- third-party model acknowledgments
- model license notes
- official Free/Pro model policy
- excluded non-commercial or review-only model categories
- FFmpeg and third-party tool notices

VisionDepth3D uses FFmpeg for video/audio processing through subprocess calls. FFmpeg licensing depends on how the FFmpeg binary was built. Users and distributors should follow the applicable FFmpeg license terms for the binary included or used with VisionDepth3D.

---

## Acknowledgments & Credits

Thank you to the researchers, developers, and open-source contributors behind the depth estimation models, video tools, AI libraries, and multimedia frameworks that make projects like VisionDepth3D possible.

VisionDepth3D integrates with third-party models and libraries while keeping their original ownership and licenses intact.

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

## Dev Notes

VisionDepth3D is developed by a solo developer and continues to grow through testing, user feedback, bug reports, and community support.

Pro helps support continued development, better testing hardware, more GPU/backend coverage, future Mac/Linux/AMD/Intel improvements, and ongoing 3D pipeline research.
