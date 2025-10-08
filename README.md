<h1 align="center"></h1>

<p align="center">
  <img width="450" height="263" alt="VisionDepth3D900x527" src="https://github.com/user-attachments/assets/eaae16ce-57ff-48f8-8dbd-1aca699b8724" />
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
    <img src="https://img.shields.io/github/downloads/VisionDepth/VisionDepth3D/total.svg" alt="GitHub Downloads">
  </a>
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python Version">
  <img src="https://img.shields.io/github/last-commit/VisionDepth/VisionDepth3D" alt="Last Commit">
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

## Table of Contents
- [Key Features](#all-in-one-3d-suite)
- [Guide Sheet: Install](#guide-sheet-install)
- [Guide Sheet: GUI Inputs](#guide-sheet-gui-inputs)
- [Troubleshooting](#troubleshooting)
- [Dev Notes](#dev-notes)
- [Acknowledgments & Credits](#acknowledgments--credits)


<h2 align="center">All-in-One 3D Suite</h2>

<h3 align="center">Real-Time 3D Stereo Composer</h3>

<p align="center">
  <img width="700" height="598" alt="3Dtab" src="https://github.com/user-attachments/assets/f6c82115-c3ba-464f-91c1-7a623ed11007" />
  <br>
  <em>(3D Tab)</em>
</p>

- **CUDA + PyTorch accelerated parallax shifting** — per-pixel, depth-aware stereo warping  
- Built on the [**VisionDepth3D Method**](VisionDepth3D_Method.md), featuring:  
  - **Pop-Control depth shaping** — percentile stretch → subject recenter → signed-gamma curve  
  - **Subject-anchored zero-parallax (EMA stabilized)** — histogram/percentile subject tracking  
  - **Dynamic parallax scaling** — central depth variance → auto stereo intensity  
  - **Edge-aware shift suppression** — gradient → sigmoid feather masking prevents halos  
  - **Floating window stabilization** — auto side masks with jitter-clamped offsets  
  - **Matte sculpting + temporal smoothing** — distance transform + EMA to round/steady subjects  
  - **Motion-aware DOF** — subject-locked Gaussian pyramid for smooth bokeh  
  - **Gradient-based occlusion healing** — fills stereo gaps by blending warped + blurred data  
- **Export formats**: Half-SBS, Full-SBS, VR (equirectangular), Anaglyph, Passive Interlaced  
- **Live preview overlays**: shift heatmaps, edge masks, stereo difference maps  
- **Workflow upgrades in v3.6**:  
  - Direct **Left/Right eye export** (no split step needed)  
  - **Clip-range rendering** (process just the section you need)  
  - Extra **padding + edge reflection** reduce bleed-through  
  - Fixed **NVENC presets** (no more forced `-preset slow`)  
- **Fully interactive**: dynamic sliders, hotkey presets, real-time 3D preview, batch-ready pipeline  
- **FFmpeg streaming pipeline**: NVENC / AMF / QSV / CPU with CRF/CQ control — no temp files  

**Result:** The Stereo Composer has matured into a *production-ready 3D engine* — blending pixel-accurate warping with real-time preview, advanced parallax controls, and streamlined export for cinema, VR, or streaming.

# AI-Powered Depth Estimation (GPU Accelerated)

<p align="center">
  <img width="700" height="598" alt="Depthtab" src="https://github.com/user-attachments/assets/c2b32af2-5be5-4d3f-9b41-cceb77b30785" />
  <br>
  <em>(Depth Estimation Tab)</em>
</p>

- **Supports 25+ depth models** including:  
  `ZoeDepth`, `Depth Anything V1/V2`, `MiDaS`, `DPT (BEiT)`, `DepthPro`, `DINOv2`, `Distill-Any-Depth`, **Marigold Diffusion**, and new additions like **Depth Anything V2 Giant**.
- **One-click model switching** with auto-download + local caching — no CLI setup required.
- **GPU-accelerated inference backends**:  
  - PyTorch (Transformers / TorchHub)  
  - ONNXRuntime with CUDA / TensorRT  
  - Diffusers (FP16) for Stable Diffusion–based models (Marigold, etc.)
- **Batch-ready pipeline**:  
  - Process image folders  
  - Process full videos (frame extraction → depth inference → encode)
- **16-bit depth output support** for richer disparity maps (Marigold / Diffusers)  
  - Preserves high-precision for inversion and HDR workflows  
  - FFmpeg-based MKV/PNG export
- **Depth Blender integration** (new in v3.6):  
  - Blend outputs from multiple models in real time for cleaner separation and smoother parallax
- Built-in **colormaps & preview modes**: Viridis, Inferno, Magma, Plasma, Grayscale.
- **Smart batching** with `get_dynamic_batch_size()` adapts to your GPU VRAM automatically.
- **Resolution-safe ONNX engine**:  
  - Detects static input shapes (e.g., `518x518`)  
  - Warm-up patch avoids shape mismatch crashes
- **Video frame interpolation (RIFE)** supported for smoother previews and exports.
- **AV1 safeguard**: auto-detects unsupported codecs with ffprobe fallback + warning.

**Result:** Depth Estimation in VD3D is now faster, more stable, and more flexible — with expanded model support, precision 16-bit outputs, and the new Depth Blender pipeline for professional-quality depth maps.

# AI Upscaling & Interpolation (GPU Accelerated)

<p align="center">
  <img width="700" height="598" alt="frametools" src="https://github.com/user-attachments/assets/4abdc68f-b878-47b6-b185-2e39ace1ba1a" />
  <br>
  <em>(Frame Tools Tab)</em>
</p>

- Integrated **RIFE (ONNX)** for frame interpolation  
  - PyTorch-free, CUDA-accelerated  
  - Supports **2x, 4x, 8x FPS doubling**  
  - Preview or export directly from GUI  
- Integrated **Real-ESRGAN x4 (ONNX)** for super-resolution  
  - GPU accelerated with **fp16 inference**  
  - Upscale **720p → 1080p**, **1080p → 4K**, or custom targets  
  - Matches resolution of depth/interpolated frames automatically
- **Massive speed boost in v3.6**:  
  - RIFE, ESRGAN, and FFmpeg writing now run **concurrently**  
  - Render times dropped from **10h → ~1h** on long clips  
  - Intelligent frame indexing and buffering keep exact sync
- **Batch-ready pipeline**:  
  - Works on raw image folders or full videos  
  - Auto-reassembles videos with original **frame count, resolution, audio, and aspect ratio**
- **VRAM-aware batching**: dynamically adjusts batch size (1–8 frames) for stability
- **FFmpeg NVENC integration**:  
  - GPU codec support with proper presets  
  - AV1/H.264/H.265 export with faststart flags
- **Live feedback in GUI**:  
  - Progress bar, FPS, ETA, and logging for long renders  
  - Cancel/resume safe

**Result:** Upscaling & interpolation in VD3D are now faster, cleaner, and more flexible — letting you tackle full-length projects in a fraction of the time, without sacrificing visual quality.

# Depth Blender (New in v3.6)

<p align="center">
  <img width="700" height="598" alt="DepthBlendTab" src="https://github.com/user-attachments/assets/89c61a02-55e8-4ff6-8ed0-bd3bd739d04e" />
  <br>
  <em>(Depth Blender Tab)</em>
</p>

- **Blend depth maps from multiple models** (e.g., DA2, ZeoDepth, Marigold) into a single, cleaner map.  
- **Frame or video mode**:  
  - Batch process paired frame folders (PNG)  
  - Or process two full video files side by side  
- **Live Preview & Frame Scrubber**:  
  - Side-by-side preview (`V2 Base | Blended Output`)  
  - Scrubbable timeline with Next/Prev buttons  
  - Hot-reload when adjusting sliders  
- **GPU-accelerated blending** (PyTorch CUDA) with CPU fallback.  
- **Adjustable parameters** (all sliders update preview live):  
  - White Strength  
  - Feather Blur (kernel size)  
  - CLAHE Clip Limit & Tile Grid  
  - Bilateral Filter (d, sigmaColor, sigmaSpace)  
- **Smart normalization**: matches brightness/contrast of blended output back to the base map.  
- **Whites boosting & outlier suppression**: reduces halos and preserves fine detail.  
- **Batch mode options**:  
  - Overwrite V2 (frames mode)  
  - Write to new output folder or video file  
- **Scrubber & hotkeys**: Left/Right arrows nudge frame index for testing blends quickly.  
- **Output scaling**: optional width/height override with high-quality Lanczos resize.  
- **Robust logging & progress bar** with stop/resume support.

**Result:** Depth Blender combines strengths of two depth models, smoothing out fuzz, suppressing white-edge artifacts, and giving creators more consistent 3D parallax across full sequences.


# Audio to Video Sync (Updated in v3.6)

<p align="center">
  <img width="558" height="587" alt="AudioTool" src="https://github.com/user-attachments/assets/bd7775a3-f625-4e77-be53-0f820a5f1b0b" />
  <br>
  <em>(Audio Tool)</em>
</p>

- **Three modes**:  
  - **Rip** → extract audio tracks from videos  
  - **Attach** → re-mux or re-encode audio back into matching videos  
  - **Attach + Stitch** → auto-attach per-clip audio and stitch final video in one step  

- **Flexible audio matching**:  
  - Auto-match audio files from a folder to videos by filename/index  
  - Or choose a single audio file for all videos  

- **Sync offset control**:  
  - GUI slider to shift audio ±10s for real-time sync correction  

- **Codec and format options**:  
  - Rip: `copy, aac, mp3, opus, flac, wav, ac3, eac3` (bitrate configurable)  
  - Attach: choose re-encode or fast copy for both video and audio  
  - Final encode: full control over vcodec (`libx264`, `libx265`, `h264_nvenc`, `hevc_nvenc`), CRF/CQ, preset, acodec, and bitrate  

- **Batch processing**:  
  - Add multiple files or entire folders of videos  
  - Auto-naming and folder output for ripped/attached files  
  - Per-clip outputs + final stitched output  

- **Gapless stitching**:  
  - Normalizes fps, resolution, pixel format, and audio sample rate across clips  
  - Ensures seamless concatenation (no desync, no black frames)  

- **Live progress + logging**:  
  - Async FFmpeg runner with progress window  
  - Logs visible inside GUI while running  
  - Cancel/resume safe  

**Result:** The Audio Tool has grown from a simple rip/attach utility into a **pro-level sync and mux suite** — with full codec control, offset correction, and reliable batch stitching, all inside the VD3D workflow.

# Preview + Format Testing

<p align="center">
  <img width="700" height="587" alt="3Dpreview" src="https://github.com/user-attachments/assets/4ce33583-8db7-40c2-a35d-d5c78efd26d9" />
  <br>
  <em>(Live 3D Preview with Anaglyph and Parallax Controls)</em>
</p>

- Real-time preview: **Interlaced, HSBS, Depth Heatmap**
- On-frame previews with **convergence + parallax tuning**
- Preview exports as images – no temp videos needed
- Save Preview Frames to show off effects with different settings


# Smart GUI + Workflow
<img width="89" height="97" alt="image" src="https://github.com/user-attachments/assets/cb7dc3e9-403a-4e54-af0d-ac44120d1a8c" />
<img width="89" height="97" alt="HelpHotkeys" src="https://github.com/user-attachments/assets/9324a4e9-9f10-4de1-a1e9-0596711410c7" />
<img width="89" height="97" alt="Hotkeys" src="https://github.com/user-attachments/assets/45592879-ec6c-4e59-b9d5-f50144db40d9" />

- Language support: **EN, FR, ES, DE, JA**
- New Help Menu bar and Hotkeys 
- Responsive **multi-tab Tkinter interface** with persistent settings
- Full GPU render control: **pause, resume, cancel**
- Codec selector with **NVENC options** (H.264, HEVC, AV1-ready)

# Output Formats & Aspect Ratios
- Formats: **Half-SBS, Full-SBS, VR180, Anaglyph, Passive Interlaced**
- Aspect Ratios: **16:9**, **2.39:1**, **2.76:1**, **4:3**, **21:9**, **1:1**, **2.35:1**
- Export formats: **MP4, MKV, AVI**
- Codec support: **XVID, MP4V, MJPG, DIVX**, **FFmpeg NVENC**

---

## Guide Sheet: Install

### 📌 System Requirements
- ✔️ This program runs on python 3.12
- ✔️ This program has been tested on cuda 12.8
- ✔️ Conda (Optional, Recommended for Simplicity)

### 📌 Step 1: Download the VisionDepth3D Program
- 1️⃣ Download the VisionDepth3D zip file from the official download source. (green button)
- 2️⃣ Extract the zip file to your desired folder (e.g., c:\user\VisionDepth3D).
- 3️⃣ Download models [Here](https://drive.google.com/file/d/1eEMcKItBn8MqH6fTCJX890A9HD054Ei4/view?usp=sharing) and extract weights folder into VisionDepth3D Main Folder
- 4️⃣ Download Distill Any Depth onnx models [here](https://huggingface.co/collections/FuryTMP/distill-any-depth-onnx-models-681cad0ff43990f5dc2ff670) (if you want to use it) and put the Distill Any Depth Folder into Weights Folder
- 
### 📌 Step 2: Create Env and Install Required Dependencies 

### 🟢 Option 1: Install via pip (Standard CMD Method)
- **1️. press (Win + R), type cmd, and hit Enter.**
- **2. Clone the Repository (Skip the git clone if you downloaded the ZIP and start from cd)**
  ```
  git clone https://github.com/VisionDepth/VisionDepth3D.git
  cd C:\VisionDepth3D-main
  pip install -r requirements.txt
  ```
  - continue to step 3: installing pytorch with cuda
  - Update 'Start_VD3D_Windows.bat' script file
  - Double click the Script to launch VD3D
  
### 🔵 Option 2: Install via Conda (Recommended)

(Automatically manages dependencies & isolates environment.)

- **1. Clone the Repository (Skip the git clone if you downloaded the ZIP and start from cd)**
- **2. Create the Conda Environment**
	To create the environment, copy and past this in conda to run:
   ```
   git clone https://github.com/VisionDepth/VisionDepth3D.git
   cd VisionDepth3D-main
   conda create -n VD3D python=3.12
   conda activate VD3D
   pip install -r requirements.txt
   ```

### 📌 Step 3: Check if CUDA is installed
🔍 Find Your CUDA Version:
Before installing PyTorch, check which CUDA version your GPU supports:
- 1️⃣ Open Command Prompt (Win + R, type cmd, hit Enter)
- 2️⃣ Run the following command:
```
nvcc --version
```
or 
```
nvidia-smi
```
- 3️⃣ Look for the CUDA version (e.g., CUDA 11.8, 12.1, etc.)
  
### 📌 Install PyTorch with the Correct CUDA Version  
Go to the official PyTorch website to find the best install command for your setup:
🔗 [ https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

install Pytorch-Cuda 12.8 or which CUDA version you are running

if you are running AMD GPU select CPU build

- Once all dependancies are installed update the batch script for system you are running and run the following command:
```
Start_VD3D_Conda.bat
# or 
Start_VD3D_Linux.bat
# or 
Start_VD3D_Windows.bat

```
Congrats you have successfully downloaded VisionDepth3D! 
This quick setup ensures you clone the repository, configure your environment, and launch the app — all in just a few simple steps.

---
## 🛠️ Guide Sheet: **Updating VisionDepth3D**

When a new version of **VisionDepth3D** is released, follow these steps to ensure a smooth transition:
---

### **Update Instructions**

1. **Backup Your Weights**  
   Move your `weights` folder out of the old `VisionDepth3D-main` directory.

2. **Download the Latest Version**  
   Delete the old folder and extract or clone the updated version of `VisionDepth3D-main`.

3. **Restore Weights Folder**  
   Place your `weights` folder back inside the newly downloaded main directory:  
   `VisionDepth3D-main/weights`

4. **Update the Path in Startup Scripts**  
   Open the startup script matching your platform:

   - `Start_VD3D_Windows.bat`
   - `Start_VD3D_Conda.bat`
   - `Start_VD3D_Linux.sh`

   Edit the script and replace any **old folder path** with the **new path** to your updated `VisionDepth3D-main`.

5. **Activate Conda Environment (if needed)**  
   If you are using the Conda starter script:
   - Open a terminal or Anaconda Prompt.
   - Run:
     ```bash
     cd path/to/updated/VisionDepth3D-main
     Start_VD3D_Conda.bat
     ```

6. **Launch the App**  
   Once everything is in place, run the appropriate script or shortcut to launch VisionDepth3D with your latest settings.

---

> **Note:** If you customized any configuration, backup those files before replacing folders. and if you run into import errors
> ```
> pip install -r requirements.txt
> ```
> inside opened terminal and that will fix any dependancie errors


---
## Guide Sheet: GUI Inputs
Use the GUI to fine-tune your 3D conversion settings.

### 1. Codec
- **Description**: Sets the output video encoder.
- **Default**: `mp4v` (CPU)
- **Options**:
  - `mp4v`, `XVID`, `DIVX` – CPU-based
  - `libx264`, `libx265` – High-quality software (CPU)
  - `h264_nvenc`, `hevc_nvenc` – GPU-accelerated (NVIDIA)

---
### 2. Foreground Shift
- **Description**: Pops foreground objects out of the screen.
- **Default**: `6.5`
- **Range**: `3.0` to `8.0`
- **Effect**: Strong values create noticeable 3D "pop" in close objects.
---
### 3. Midground Shift
- **Description**: Depth for mid-layer transition between foreground and background.
- **Default**: `1.5`
- **Range**: `-3.0` to `5.0`
- **Effect**: Smooths the 3D transition — higher values exaggerate depth between layers.
---
### 4. Background Shift
- **Description**: Shift depth for background layers (far away).
- **Default**: `-6.0`
- **Range**: `-10.0` to `0.0`
- **Effect**: More negative pushes content into the screen (deeper background).
---
### 5. Sharpness Factor
- **Description**: Applies a sharpening filter to the output.
- **Default**: `0.2`
- **Range**: `-1.0` (softer) to `1.0` (sharper)
- **Effect**: Brings clarity to 3D edges; avoid over-sharpening to reduce halos.
---

### 6. Convergence Offset
- **Description**: Shifts the entire stereo image inward or outward to adjust the **overall convergence point** (zero-parallax plane).
- **Default**: `0.000`
- **Range**: `-0.050` to `+0.050`
- **Effect**:  
  - Positive values push the image **deeper into the screen** (stronger positive parallax).  
  - Negative values **pull the scene forward** (increased pop-out effect).
- **Tip**: Use small increments like `±0.010` for subtle depth balancing.

---

### 7. Max Pixel Shift (%)
- **Description**: Limits the **maximum pixel displacement** caused by stereo shifting, expressed as a percentage of video width.
- **Default**: `0.020` (2%)
- **Range**: `0.005` to `0.100`
- **Effect**:  
  - Low values reduce eye strain but can flatten the 3D effect.  
  - High values create more dramatic depth but may introduce ghosting or artifacts.
- **Best Use**: Keep between `0.015`–`0.030` for clean results.

---

### 8. Parallax Balance
- **Description**: Adjusts how **strongly the 3D effect favors the subject's depth** versus full-scene stereo balance.
- **Default**: `0.80`
- **Range**: `0.00` to `1.00`
- **Effect**:  
  - `1.0` = Full parallax (strong 3D depth everywhere).  
  - `0.0` = Subject stays fixed, depth minimized elsewhere.
- **Use For**: Tuning stereo focus around people or central motion while avoiding exaggerated background distortion.

---
### 9. FFmpeg Codec & CRF Quality
- **Codec**: Choose GPU-accelerated encoders (`h264_nvenc`, `hevc_nvenc`) for faster renders.
- **CRF (Constant Rate Factor)**:
  - **Default**: `23`
  - **Range**: `0` (lossless) to `51` (worst)
  - Lower values = better visual quality.
---
### 10. Stabilize Zero-Parallax (center-depth)
- **Checkbox**: **Stabilize Zero-Parallax (center-depth)**
- **Effect**: Enables **Dynamic Zero Parallax Tracking** — the depth plane will automatically follow the subject’s depth to minimize excessive 3D warping.  
- **Function**: Dynamically adjusts the zero-parallax plane to follow the estimated subject depth (typically the central object or character). This keeps key elements at screen depth, reducing eye strain and excessive parallax.
- **Effect**: Helps stabilize the 3D effect by anchoring the subject at screen level, especially useful for scenes with depth jumps or fast movement.
- **Recommended for**: Dialogue scenes, human-centric content, or anything where central focus should feel "on screen" rather than floating in depth.

---
### 11. Stereo Scaling (IPD)
- **Description**: Controls the **inter-pupillary distance (IPD)** scaling, effectively adjusting how strong the stereo separation feels.
- **Default**: `1.15`
- **Range**: `0.50` to `2.00`
- **Effect**:  
  - Higher values exaggerate stereo depth (more 3D).  
  - Lower values flatten depth (safer for long viewing).  
- **Tip**: Keep near `1.0–1.3` for natural results.

---
### 12. Depth Pop Gamma
- **Description**: Adjusts the gamma curve for depth, controlling how depth “pops” across the scene.
- **Default**: `1.0`
- **Range**: `0.5` to `2.0`
- **Effect**:  
  - Higher = stronger pop, can over-accentuate close objects.  
  - Lower = smoother, flatter depth distribution.

---
### 13. Subject Lock
- **Description**: Locks the subject’s depth position relative to the zero-parallax plane.
- **Default**: `1.30`
- **Range**: `1.0` to `2.0`
- **Effect**:  
  - Prevents subject from drifting too deep or too far out.  
  - Useful for keeping faces/characters consistently anchored.

---
### 14. FG / BG Push ×
- **Description**: Extra multipliers for pushing **foreground** or **background** layers.  
- **Default**: `FG: 1.20`, `BG: 1.10`
- **Range**: `0.5` to `2.0`
- **Effect**:  
  - FG Push emphasizes pop-out.  
  - BG Push exaggerates scene depth.  
- **Best Use**: Subtle tweaks to fine-tune stereo balance.

---
### 15. Color Grading Controls
- **Saturation**:  
  - Default: `1.35`  
  - Adjusts color intensity. >1 = more vivid, <1 = muted.  

- **Brightness**:  
  - Default: `0.04`  
  - Fine-tunes exposure; small values recommended.  

- **Contrast**:  
  - Default: `1.10`  
  - Enhances separation between light/dark regions.  

- **Effect**: These adjustments let you preview and render with tuned color grading **before** upscaling or final encoding.  
- **Tip**: Avoid extreme values to prevent clipping or oversaturation.

---
### 16. Floating Window (DFW)
- **Checkbox**: **Enable Floating Window**
- **Effect**: Shifts the visible “window” edges of the stereo image inward.  
- **Purpose**: Prevents window violations (objects being cut off by screen edges).  
- **Recommended for**: Full-screen 3D playback, cinema, or VR headsets.

---
## Depth Map Tips
- Match **resolution** and **FPS** between your input video and depth map.
- Use the **Inverse Depth** checkbox if bright = far instead of close.
- Recommended depth models:
  - `Distill Any Depth`, `Depth Anything V2`, `MiDaS`, `DPT-Large`, etc.
  - Choose *Large* models for better fidelity.
---
## Rendering Time Estimates
| Clip Length | Estimated Time (with GPU) |
|-------------|---------------------------|
| 30 seconds  | 1–4 mins                  |
| 5 minutes   | 10–25 mins                |
| Full Movie  | 6–24+ hours               |
---


## Example Workflow

1. Select your **depth model** from the dropdown.
2. Choose an **output directory** for saving results.
3. Enable your **preferred settings** (invert, colormap, etc.).
4. Set **batch size** depending on GPU/VRAM capacity.  
   *(Tip: Resize your video or switch to a lighter model if memory is limited.)*
5. Select your **image / video / folder** and start processing.
6. Once the **depth map video is generated**, head over to the **3D tab**.
7. Input your original video and the newly created depth map.
8. Adjust 3D settings for the preferred stereo effect.
9. Hit **"Generate 3D Video"** and let it roll!

---

## Post-Processing: RIFE + Real-ESRGAN (FPS + Upscale)

Use these models to clean up and enhance 3D videos:

1. In the **Upscale tab**, load your 3D video and enable **“Save Frames Only”**.
2. Input the **width × height** of the 3D video.  
   *(No need to set FPS or codec when saving frames.)*
3. Set batch size to `1` — batch processing is unsupported by some AI models.
4. Select **AI Blend Mode** and **Input Resolution**:

### AI Blend Mode

| Mode    | Blend Ratio (AI : Original) | Description                                                                 |
|---------|-----------------------------|-----------------------------------------------------------------------------|
| OFF     | 100% : 0%                   | Full AI effect (only the ESRGAN result is used).                           |
| LOW     | 85% : 15%                   | Strong AI enhancement with mild natural tone retention.                    |
| MEDIUM  | 50% : 50%                   | Balanced mix for natural image quality.                                    |
| HIGH    | 25% : 75%                   | Subtle upscale; mostly original with a hint of enhancement.                |

### Input Resolution Setting

| Input Resolution | Processing Behavior                              | Performance & Quality Impact                                               |
|------------------|--------------------------------------------------|-----------------------------------------------------------------------------|
| **100%**         | Uses full-resolution frames for AI upscaling.   | ✅ Best quality. ❌ Highest GPU usage.                                       |
| **75%**          | Slightly downsamples before feeding into AI.    | ⚖️ Good balance. Minimal quality loss.                                      |
| **50%**          | Halves frame size before AI.                    | ⚡ 2× faster. Some detail loss possible.                                    |
| **25%**          | Very low-resolution input.                      | 🚀 Fastest speed. Noticeable softness — best for previews/tests.           |

5. Select your **Upscale Model** and start the process.
6. Once done, open the **VDStitch tab**:
    - Input the upscaled frame folder.
    - Set the **video output directory and filename**.
    - Enter the same **resolution and FPS** as your original 3D video.
    - Enable **RIFE FPS Interpolation**.
7. Set the **RIFE multiplier to ×2** for smooth results.  
   *(⚠️ Higher multipliers like ×4 may cause artifacts on scene cuts.)*
8. Start processing — you now have an enhanced 3D video with upscaled clarity and smoother motion!


---
## Troubleshooting
- **Black/Empty Output**: Wrong depth map resolution or mismatch with input FPS.
- **Halo/Artifacts**:
  - Increase feather strength and blur size.
  - Enable subject tracking and clamp the zero parallax offset.
- **Out of Memory (OEM)**:
  - Enable FFmpeg rendering for better memory usage.
  - Use `libx264` or `h264_nvenc` and avoid long clips in one go.
---
## Dev Notes
This tool is being developed by a solo dev with nightly grind energy (🕐 ~4 hours a night). If you find it helpful, let me know — feedback, bug reports, and feature ideas are always welcome!

## Acknowledgments & Credits

**Thank You!**

A heartfelt thank you to all the researchers, developers, and contributors behind the incredible depth estimation models and open-source tools used in this project. Your dedication, innovation, and generosity have made it possible to explore the frontiers of 3D rendering and video processing. Your work continues to inspire and empower developers like me to build transformative, creative applications.

### **Supported Depth Models**
| Model Name | Creator / Organization | Hugging Face Repository |
|------------|------------------------|-------------------------|
| **Distil-Any-Depth-Large** | xingyang1 | [Distill-Any-Depth-Large-hf](https://huggingface.co/xingyang1/Distill-Any-Depth-Large-hf) |
| **Distil-Any-Depth-Small** | xingyang1 | [Distill-Any-Depth-Large-hf](https://huggingface.co/xingyang1/Distill-Any-Depth-Small-hf) |
| **Depth Anything V2 Large** | Depth Anything Team | [Depth-Anything-V2-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) |
| **Depth Anything V2 Base** | Depth Anything Team | [Depth-Anything-V2-Base-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) |
| **Depth Anything V2 Small** | Depth Anything Team | [Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| **Depth Anything V1 Large** | LiheYoung | [Depth-Anything-V2-Large](https://huggingface.co/LiheYoung/Depth-Anything-V2-Large) |
| **Depth Anything V1 Base** | LiheYoung | [depth-anything-base-hf](https://huggingface.co/LiheYoung/depth-anything-base-hf) |
| **Depth Anything V1 Small** | LiheYoung | [depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) |
| **V2-Metric-Indoor-Large** | Depth Anything Team | [Depth-Anything-V2-Metric-Indoor-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf) |
| **V2-Metric-Outdoor-Large** | Depth Anything Team | [Depth-Anything-V2-Metric-Outdoor-Large-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf) |
| **DA_vitl14** | LiheYoung | [depth_anything_vitl14](https://huggingface.co/LiheYoung/depth_anything_vitl14) |
| **DA_vits14** | LiheYoung | [depth_anything_vits14](https://huggingface.co/LiheYoung/depth_anything_vits14) |
| **DepthPro** | Apple | [DepthPro-hf](https://huggingface.co/apple/DepthPro-hf) |
| **ZoeDepth** | Intel | [zoedepth-nyu-kitti](https://huggingface.co/Intel/zoedepth-nyu-kitti) |
| **MiDaS 3.0** | Intel | [dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas) |
| **DPT-Large** | Intel | [dpt-large](https://huggingface.co/Intel/dpt-large) |
| **DinoV2** | Facebook | [dpt-dinov2-small-kitti](https://huggingface.co/facebook/dpt-dinov2-small-kitti) |
| **dpt-beit-large-512** | Intel | [dpt-beit-large-512](https://huggingface.co/Intel/dpt-beit-large-512) |


### **Multimedia Framework**
This project utilizes the FFmpeg multimedia framework for video/audio processing via subprocess invocation.
FFmpeg is licensed under the GNU GPL v3 or LGPL, depending on how it was built. No modifications were made to the FFmpeg source or binaries — the software simply executes FFmpeg as an external process.

You may obtain a copy of the FFmpeg license at:
https://www.gnu.org/licenses/

VisionDepth3D calls FFmpeg strictly for encoding, muxing, audio extraction, and frame rendering operations, in accordance with license requirements.
