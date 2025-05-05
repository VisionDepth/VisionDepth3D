<h1 align="center">VisionDepth3D</h1>

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
</h3>

<p align="center">
  <a href="https://visiondepth3d.itch.io/visiondepth3d" target="_blank">
    <img src="assets/widget-preview.png" alt="Download VisionDepth3D on Itch.io" width="300" style="border-radius: 8px;">
  </a>
  <br>
  <em style="font-size: 14px; color: #888;">Click to download or support the project 💙</em>
</p>



---

## Table of Contents
- [Key Features](#key-features--visiondepth3d-all-in-one-3d-suite)
- [Guide Sheet: Install](#guide-sheet-install)
- [Guide Sheet: GUI Inputs](#guide-sheet-gui-inputs)
- [Troubleshooting](#troubleshooting)
- [Dev Notes](#dev-notes)
- [Acknowledgments & Credits](#acknowledgments--credits)


# Key Features – VisionDepth3D All-in-One 3D Suite

## AI-Powered Depth Estimation (GPU Accelerated)
- Seamless integration with 20+ transformer-based depth models: **ZoeDepth, Depth Anything, MiDaS, DPT, DepthPro, DinoV2**, and more
- One-click model selection with **automatic downloads**, no CLI setup or config files
- **PyTorch GPU acceleration** (no OpenCV recompile needed)
- Batch support for both **video sequences and image folders**
- **Temporal smoothing**, intelligent scene-adaptive normalization
- Built-in **color inversion**, customizable **colormaps** (Viridis, Inferno, Magma)
- Real-time frame-by-frame **progress bar, FPS display, and ETA tracking**
- Auto-resizing, smart batching, and graceful handling of large resolutions
- Pause/resume/cancel supported during all GPU operations

## Advanced 3D Rendering Engine (Real-Time Stereo Composer)
- **Pixel-accurate depth parallax shifting** using CUDA + PyTorch
- Full control over **foreground(pop) / midground(balance) / background(pull) parallax**
- **Half-SBS, Full-SBS, VR, Passive Interlaced, Anaglyph, and Dynamic Floating Window** formats
- Dynamic floating window with **cinema-style masking that slides and eases smoothly**
- Built-in **Pulfrich effect renderer** (motion delay-based left-eye blending)
- **Feathered shift masking**, sharpening, and edge-aware smoothing
- **Subject tracking-based convergence** for natural stereo alignment
- GPU-accelerated real-time processing with live GUI stats (FPS, elapsed time, %)
- Output is compatible with **Quest VR, YouTube 3D, and most stereo players**

## RIFE Frame Interpolation (ONNX Runtime)
- Integrated **RIFE ONNX** model (no PyTorch required) for real-time frame doubling
- Interpolation modes: **2x, 4x, 8x FPS** with smooth motion blending
- Folder-based processing of raw frames + automatic **video reassembly**
- Preserves **frame resolution, count, audio sync**, and aspect ratio
- Supports preview and export at high quality using FFmpeg codecs
- Real-time progress tracking + FPS + ETA built into GUI

## 4x Super Resolution Upscaling (Real-ESRGAN Powered – ONNX GPU)
- Integrated **Real-ESRGAN (x4) super-resolution model**, exported to ONNX with full GPU support
- Batch upscaling with intelligent **VRAM-aware batching** (1–8 frames)
- Supports **720p ➜ 1080p**, **1080p ➜ 4K**, or any custom resolution
- Automatically resizes final frames to match output format and target resolution
- Lightning-fast **CUDA-accelerated ONNX runtime** (no PyTorch required)
- Full integration with frame renderer: upscales after 3D rendering or interpolation
- Clean, artifact-free outputs using enhanced fp16 inference for visual clarity
- Progress bar, FPS counter, ETA timer integrated into the GUI
- Fully exportable to video with codec support: **MP4V, XVID, MJPG, FFmpeg NVENC**

## Smart Depth-Aware Effects
- **Gradient-aware artifact suppression** near depth edges (limbs, hair)
- **Feathered transition masks** to avoid ghosting and popping
- Depth-aware **sharpening and blending** for polished 3D output
- **Dynamic bar generation** for floating window masking that **eases smoothly** like theatrical films
- Real-time **zero parallax estimation and smoothing** per-frame

## Audio + Video Re-Integration
- Audio button to directly rip and attach audio from source video using **FFmpeg**
- Format choices: **AAC, MP3, WAV** with adjustable bitrate
- Built-in tools, no shell commands needed – fully GUI-based

## Preview System with Format Testing
- Choose preview format: **Passive Interlaced**, **HSBS**, **Shift Heatmap**
- Live preview on frame for quick tuning
- Auto-exports as image preview file, no temp videos needed
- Toggle convergence depth and parallax before full render

## Real-Time 3D Player (VDPlayer)
- Lightweight player built for **Half-SBS, Full-SBS**, and **VR** output
- Fast seeking, play/pause/fullscreen toggles
- Timestamp scrubber + resolution-aware display
- Designed to instantly preview 3D results without leaving the app

## Smart GUI & Workflow Features
- Multi-tab **Tkinter interface**, responsive and persistent settings
- Pause, resume, and cancel buttons for all rendering threads
- Codec selector with **GPU NVENC options** (H.264, HEVC, AV1-ready)
- **One-click launcher**, no pip/CLI scripting needed
- slider recall, and auto-cropping for black bars

## Supported Output Formats & Aspect Ratios
- Formats: **Half-SBS**, **Full-SBS**, **VR Mode**, **Red-Cyan Anaglyph**, **Passive Interlaced**
- Ratios: **16:9**, **CinemaScope (2.39:1)**, **2.76:1**, **4:3**, **21:9**, **Square 1:1**, **Classic 2.35:1**
- Supports export in **MP4, MKV, AVI** with codecs: **XVID, MP4V, MJPG, DIVX**, and **FFmpeg NVENC**

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

### 📌 Step 2: Create Env and Install Required Dependencies 

### 🟢 Option 1: Install via pip (Standard CMD Method)
- **1️. press (Win + R), type cmd, and hit Enter.**
- **2. Clone the Repository (Skip the git clone if you downloaded the ZIP and start from cd)**
  ```
  git clone https://github.com/VisionDepth/VisionDepth3D.git
  cd C:\VisionDepth3D-main
  pip install -r requirements.txt
  ```
  - continue to installing pytorch with cuda and then run VisionDepth3D.bat 
  - 
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

- Once Pytorch and all dependancies are installed update the batch script for system you are running and run the following command:
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

I want to express my gratitude to the amazing creators and contributors behind the depth estimation models used in this project. Your work has made it possible to push the boundaries of 3D rendering and video processing. 🙌

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

**🙏 Thank You!**
A huge thank you to all the researchers, developers, and contributors who created and shared these models. Your work is inspiring and enables developers like me to build exciting and innovative applications! 🚀💙


