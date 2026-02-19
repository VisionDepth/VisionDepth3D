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

# Notice
VisionDepth3D is licensed under a proprietary, no-derivatives license.  
Forking, redistributing, modifying, or creating derivative works is strictly prohibited.

---

## Table of Contents
- [Key Features](#all-in-one-3d-suite)
- [Guide Sheet: Install](#guide-sheet-install)
- [Documentation](#-documentation)
- [Dev Notes](#dev-notes)
- [Acknowledgments & Credits](#acknowledgments--credits)

<h2 align="center">All-in-One 3D Suite</h2>

<h3 align="center">3D Generator (Stereo Composer)</h3>

<p align="center">
  <img width="700" height="598" alt="3Dtab" src="https://github.com/user-attachments/assets/f6c82115-c3ba-464f-91c1-7a623ed11007" />
  <br>
  <em>(3D Generator Tab)</em>
</p>

- **GPU-accelerated stereo warping**: per-pixel, depth-aware parallax shifting (CUDA + PyTorch)
- Built on the [**VisionDepth3D Method**](VisionDepth3D_Method.md), including:
  - **Depth shaping (Pop Controls)**: percentile stretch + subject recenter + curve shaping for natural separation
  - **Subject-anchored convergence**: EMA-stabilized zero-parallax tracking for comfort and consistency
  - **Dynamic stereo scaling (IPD)**: scene-aware intensity that adapts to depth variance
  - **Edge-aware masking + feathering**: suppress halos and clean up subject boundaries
  - **Floating window (DFW)**: cinematic edge protection to prevent window violations
  - **Occlusion healing**: fills stereo gaps and reduces edge artifacts
- **Live preview + diagnostics**: anaglyph, SBS, heatmaps, edge/mask inspection, stereo difference views
- **Clip-range rendering** for fast testing on difficult scenes before full renders
- **Export formats**: Half-SBS, Full-SBS, VR (equirectangular), Anaglyph, Passive Interlaced
- **Encoding pipeline**: FFmpeg with CPU and hardware encoders (NVENC/AMF/QSV) plus quality controls (CRF/CQ)

**Result:** A production-ready 2D-to-3D engine with real-time tuning tools, stability features, and flexible export formats for VR and cinema workflows.

---

# Depth Estimation (AI Depth Engine)

<p align="center">
  <img width="700" height="598" alt="Depthtab" src="https://github.com/user-attachments/assets/c2b32af2-5be5-4d3f-9b41-cceb77b30785" />
  <br>
  <em>(Depth Estimation Tab)</em>
</p>

- **25+ supported depth models** (ZoeDepth, MiDaS, DPT/BEiT, DINOv2, DepthPro, Depth Anything V1/V2, Distill-Any-Depth, Marigold, and more)
- **One-click model switching** with auto-download + local caching
- **Multiple inference backends**:
  - PyTorch (Transformers / TorchHub)
  - ONNXRuntime (CUDA / TensorRT)
  - Diffusers FP16 (for diffusion-based depth)
- **Image + video + batch workflows**:
  - Single image
  - Image folder batch
  - Full video depth rendering
  - Video folder batch
- **Optional high precision output** (when supported) for cleaner disparity and stronger stability in post
- **Built-in preview modes + colormaps** for fast inspection
- **Stability + safety features**: resolution/shape handling, codec probing, and fallback behavior to avoid common crashes

**Result:** Fast, flexible depth generation for everything from quick tests to full-length depth videos ready for stereo conversion.

---

# FPS / Upscale Enhancer (RIFE + Real-ESRGAN)

<p align="center">
  <img width="700" height="598" alt="frametools" src="https://github.com/user-attachments/assets/4abdc68f-b878-47b6-b185-2e39ace1ba1a" />
  <br>
  <em>(FPS / Upscale Enhancer Tab)</em>
</p>

- **RIFE interpolation (ONNX)**: 2× / 4× / 8× FPS generation with GPU acceleration
- **Real-ESRGAN upscaling (ONNX)**: high-quality super-resolution with optional FP16
- **Two processing pipelines**:
  - **Merged** (stable, low memory)
  - **Threaded** (higher throughput, better utilization)
- **Full video workflow support**:
  - Optional scene splitting for long videos
  - Rebuild output with correct resolution, FPS, and encoding settings
- **Render feedback**: progress, FPS, ETA, logs, and safe cancel handling

**Result:** Turn low-res or low-FPS sources into clean, smooth outputs built for VR playback and high refresh displays.

---

# Depth Blender (Multi-Source Depth Fusion)

<p align="center">
  <img width="700" height="598" alt="DepthBlendTab" src="https://github.com/user-attachments/assets/89c61a02-55e8-4ff6-8ed0-bd3bd739d04e" />
  <br>
  <em>(Depth Blender Tab)</em>
</p>

- **Blend two depth sources** into one cleaner, more stable depth map/video
- **Frames or video mode**:
  - Pair two PNG frame folders
  - Or pair two depth videos
- **Live preview + scrubber**: side-by-side (Base vs Blended) with fast frame navigation
- **Edge-focused blend controls**:
  - White strength injection
  - Feather blur smoothing
  - CLAHE contrast shaping
  - Bilateral edge-preserving denoise
- **Normalization back to base** for consistent depth scale
- **Batch output options**: overwrite base, output to new folder, or export a blended video

**Result:** Cleaner edges, stronger subject separation, and more consistent parallax behavior across full sequences.

---

# Audio Tool (Rip, Attach, Attach + Stitch)

<p align="center">
  <img width="558" height="587" alt="AudioTool" src="https://github.com/user-attachments/assets/bd7775a3-f625-4e77-be53-0f820a5f1b0b" />
  <br>
  <em>(Audio Tool)</em>
</p>

- **Rip**: extract audio tracks from videos (copy or re-encode)
- **Attach**: mux audio back into processed clips (fast copy by default)
- **Attach + Stitch**: batch attach audio per clip, then stitch into one final gapless export
- **Smart matching**: auto-match audio to clips by filename patterns or use one audio track for all
- **Audio offset control** for sync fixes
- **Codec control** for both per-clip muxing and final output encoding
- **Logging + progress** via FFmpeg runner

**Result:** A practical post stage for restoring audio, correcting sync, and finishing multi-clip renders into one clean final movie.

---

# Preview + Format Testing

<p align="center">
  <img width="700" height="587" alt="3Dpreview" src="https://github.com/user-attachments/assets/4ce33583-8db7-40c2-a35d-d5c78efd26d9" />
  <br>
  <em>(Live 3D Preview)</em>
</p>

- Real-time preview modes: **Anaglyph, SBS, Passive Interlaced, Depth and Shift Heatmaps**
- On-frame tuning: convergence and parallax checks without committing to long renders
- Save preview frames for quick comparisons and sharing

---

# Smart GUI + Workflow

<img width="89" height="97" alt="image" src="https://github.com/user-attachments/assets/cb7dc3e9-403a-4e54-af0d-ac44120d1a8c" />
<img width="89" height="97" alt="HelpHotkeys" src="https://github.com/user-attachments/assets/9324a4e9-9f10-4de1-a1e9-0596711410c7" />
<img width="89" height="97" alt="Hotkeys" src="https://github.com/user-attachments/assets/45592879-ec6c-4e59-b9d5-f50144db40d9" />

- Multi-tab interface with persistent settings
- Help menu + hotkeys
- Pause, resume, and cancel for long GPU jobs
- Multi-language UI support (EN, FR, ES, DE, JA)
- Hardware encoding options integrated into export workflow

---

# Output Formats & Aspect Ratios

- Stereo formats: **Half-SBS, Full-SBS, VR180, Anaglyph, Passive Interlaced**
- Aspect ratios: **16:9, 2.39:1, 2.76:1, 4:3, 21:9, 1:1, 2.35:1**
- Containers: **MP4, MKV, AVI**
- Encoders: CPU + FFmpeg hardware options (NVENC/AMF/QSV) when available


## Guide Sheet: Install

### 📌 System Requirements
- ✔️ This program runs on python 3.13
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
   conda create -n VD3D python=3.13
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

## 📘 Documentation

VisionDepth3D includes a full professional user manual with workflows, tuning guides, and advanced features.

👉 Start here: [UserGuide.md](UserGuide.md)

If you're new, begin with:
• Depth Estimation →  
• Depth Blender →  
• 3D Generator →  
• Preview & Clip Testing →  
• Final Render

---

## Dev Notes
This tool is being developed by a solo dev with nightly grind energy (🕐 ~4 hours a night). If you find it helpful, let me know — feedback, bug reports, and feature ideas are always welcome!

## Acknowledgments & Credits

**Thank You!**

A heartfelt thank you to all the researchers, developers, and contributors behind the incredible depth estimation models and open-source tools used in this project. Your dedication, innovation, and generosity have made it possible to explore the frontiers of 3D rendering and video processing. Your work continues to inspire and empower developers like me to build transformative, creative applications.

### **Supported Depth Models**
| Model Name (UI) | Creator / Organization | Model ID / Repository |
|---|---|---|
| **Marigold Depth v1.1 (Diffusers)** | PRS ETH | [`prs-eth/marigold-depth-v1-1`](https://huggingface.co/prs-eth/marigold-depth-v1-1) |
| **Marigold Depth v1.0** | PRS ETH | [`prs-eth/marigold-depth-v1-0`](https://huggingface.co/prs-eth/marigold-depth-v1-0) |
| **Distill-Any-Depth Large (xingyang1)** | xingyang1 | [`xingyang1/Distill-Any-Depth-Large-hf`](https://huggingface.co/xingyang1/Distill-Any-Depth-Large-hf) |
| **Distill-Any-Depth Small (xingyang1)** | xingyang1 | [`xingyang1/Distill-Any-Depth-Small-hf`](https://huggingface.co/xingyang1/Distill-Any-Depth-Small-hf) |
| **Video Depth Anything Large** | Depth Anything Team | [`depth-anything/Video-Depth-Anything-Large`](https://huggingface.co/depth-anything/Video-Depth-Anything-Large) |
| **Video Depth Anything Small** | Depth Anything Team | [`depth-anything/Video-Depth-Anything-Small`](https://huggingface.co/depth-anything/Video-Depth-Anything-Small) |
| **Video Depth Anything (ONNX)** | Depth Anything Team | Bundled ONNX backend (`onnx:VideoDepthAnything`) |
| **Distill-Any-Depth Large (ONNX)** | xingyang1 | Bundled ONNX backend (`onnx:DistillAnyDepthLarge`) |
| **Distill-Any-Depth Base (ONNX)** | xingyang1 | Bundled ONNX backend (`onnx:DistillAnyDepthBase`) |
| **Distill-Any-Depth Small (ONNX)** | xingyang1 | Bundled ONNX backend (`onnx:DistillAnyDepthSmall`) |
| **DA3METRIC-LARGE** | Depth Anything Team | [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE) |
| **DA3MONO-LARGE** | Depth Anything Team | [`depth-anything/DA3MONO-LARGE`](https://huggingface.co/depth-anything/DA3MONO-LARGE) |
| **DA3-LARGE** | Depth Anything Team | [`depth-anything/DA3-LARGE`](https://huggingface.co/depth-anything/DA3-LARGE) |
| **DA3-LARGE-1.1** | Depth Anything Team | [`depth-anything/DA3-LARGE-1.1`](https://huggingface.co/depth-anything/DA3-LARGE-1.1) |
| **DA3-BASE** | Depth Anything Team | [`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE) |
| **DA3-SMALL** | Depth Anything Team | [`depth-anything/DA3-SMALL`](https://huggingface.co/depth-anything/DA3-SMALL) |
| **DA3-GIANT** | Depth Anything Team | [`depth-anything/DA3-GIANT`](https://huggingface.co/depth-anything/DA3-GIANT) |
| **DA3-GIANT-1.1** | Depth Anything Team | [`depth-anything/DA3-GIANT-1.1`](https://huggingface.co/depth-anything/DA3-GIANT-1.1) |
| **DA3NESTED-GIANT-LARGE** | Depth Anything Team | [`depth-anything/DA3NESTED-GIANT-LARGE`](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) |
| **DA3NESTED-GIANT-LARGE-1.1** | Depth Anything Team | [`depth-anything/DA3NESTED-GIANT-LARGE-1.1`](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) |
| **Depth Anything v2 Large** | Depth Anything Team | [`depth-anything/Depth-Anything-V2-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) |
| **Depth Anything v2 Base** | Depth Anything Team | [`depth-anything/Depth-Anything-V2-Base-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) |
| **Depth Anything v2 Small** | Depth Anything Team | [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| **Depth Anything v2 Metric Indoor (Large)** | Depth Anything Team | [`depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf) |
| **Depth Anything v2 Metric Outdoor (Large)** | Depth Anything Team | [`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf) |
| **Depth Anything v2 Giant (safetensors)** | Depth Anything Team | Local weights (`dav2:vitg_fp32`) |
| **Depth Anything v1 Large** | LiheYoung | [`LiheYoung/depth-anything-large-hf`](https://huggingface.co/LiheYoung/depth-anything-large-hf) |
| **Depth Anything v1 Base** | LiheYoung | [`LiheYoung/depth-anything-base-hf`](https://huggingface.co/LiheYoung/depth-anything-base-hf) |
| **Depth Anything v1 Small** | LiheYoung | [`LiheYoung/depth-anything-small-hf`](https://huggingface.co/LiheYoung/depth-anything-small-hf) |
| **Prompt Depth Anything VITS Transparent** | Depth Anything Team | [`depth-anything/prompt-depth-anything-vits-transparent-hf`](https://huggingface.co/depth-anything/prompt-depth-anything-vits-transparent-hf) |
| **LBM Depth** | Jasper | [`jasperai/LBM_depth`](https://huggingface.co/jasperai/LBM_depth) |
| **DepthPro (Apple)** | Apple | [`apple/DepthPro-hf`](https://huggingface.co/apple/DepthPro-hf) |
| **ZoeDepth (NYU+KITTI)** | Intel | [`Intel/zoedepth-nyu-kitti`](https://huggingface.co/Intel/zoedepth-nyu-kitti) |
| **MiDaS 3.0 (DPT-Hybrid)** | Intel | [`Intel/dpt-hybrid-midas`](https://huggingface.co/Intel/dpt-hybrid-midas) |
| **DPT Large (Intel)** | Intel | [`Intel/dpt-large`](https://huggingface.co/Intel/dpt-large) |
| **DPT Large (Manojb)** | Manojb | [`Manojb/dpt-large`](https://huggingface.co/Manojb/dpt-large) |
| **DPT BEiT Large 512** | Intel | [`Intel/dpt-beit-large-512`](https://huggingface.co/Intel/dpt-beit-large-512) |
| **MiDaS v2 (Qualcomm)** | Qualcomm | [`qualcomm/Midas-V2`](https://huggingface.co/qualcomm/Midas-V2) |



### **Multimedia Framework**
This project utilizes the FFmpeg multimedia framework for video/audio processing via subprocess invocation.
FFmpeg is licensed under the GNU GPL v3 or LGPL, depending on how it was built. No modifications were made to the FFmpeg source or binaries — the software simply executes FFmpeg as an external process.

You may obtain a copy of the FFmpeg license at:
https://www.gnu.org/licenses/

VisionDepth3D calls FFmpeg strictly for encoding, muxing, audio extraction, and frame rendering operations, in accordance with license requirements.
