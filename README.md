# <h1 align="center">VisionDepth3D</h1>  
## <h2 align="center">VisionDepth3D: The All-in-One 3D Suite for Creators
From depth maps to final render — everything you need to turn flat 2D footage into cinematic 3D.
Powered by AI. Optimized for speed. Designed for creators. </h2> 
### <h3 align="center">[![Github All Releases](https://img.shields.io/github/downloads/VisionDepth/VisionDepth3D/total.svg)]() ![Python Version](https://img.shields.io/badge/python-3.12-blue)</h3>

---

## **GUI Layout**
![GUITabsSBS](https://github.com/user-attachments/assets/337a6bd3-43ad-4f25-ab63-4563387305d6)

---

## Table of Contents
- [Key Features](#key-features)
- [Advanced Depth Estimation & 3D Processing](#advanced-depth-estimation--3d-video-processing)
- [GUI Layout](#gui-layout)
- [System Requirements](#-system-requirements)
- [Installation](#guide-sheet-install)
- [GUI Settings & Adjustments](#guide-sheet-gui-inputs)
- [Pulfrich Effect Explained](#pulfrich-effect-explained)
- [Troubleshooting](#troubleshooting)
- [Dev Notes](#notes)
- [Acknowledgments & Credits](#acknowledgments--credits)

## Key Features – VisionDepth3D All-in-One 3D Suite

### AI-Powered Depth Estimation (GPU Accelerated)
- Multi-model support: Depth Anything V2, MiDaS 3.0, ZoeDepth, DPT, DinoV2, and more
- One-click model downloads (cached locally for fast reuse)
- Fully GPU-accelerated via PyTorch (no OpenCV CUDA builds required)
- Adaptive depth smoothing for clean, flicker-free results

### Advanced 3D Video Rendering Engine
- Real-time pixel shifting for HSBS, Anaglyph, and Over/Under 3D formats
- Depth-based stereo parallax generation (foreground/midground/background tuning)
- Pulfrich Effect blending for motion-driven depth enhancements
- Adjustable convergence, divergence, and depth transition settings
- Optional sharpening and output aspect ratio correction

### RIFE Frame Interpolation (ONNX Runtime)
- Built-in RIFE ONNX model for smooth FPS upscaling (2x, 4x, 8x)
- Full GPU acceleration with ETA, FPS counter, and progress tracking
- Real-time interpolation of both video and depth maps
- Output matches frame count and resolution for perfect sync in stereo rendering

### Audio & Video Sync Support
- Automatically reattaches original audio after SBS rendering
- Clean muxing via ffmpeg-python (no external dependency needed)
- Ensures final 3D output is playable and shareable immediately

### Built-in Video Preview Player
- VDPlayer tab with load, play, pause, stop, and fullscreen
- Stereo-friendly layout for previewing your SBS outputs
- Timeline scrubber and timestamp display

### Smart UI & Workflow Enhancements
- Multi-tab GUI built in Tkinter (clean layout, responsive controls)
- Real-time progress bars and FPS/ETA indicators during all operations
- Black bar removal for full-frame rendering
- Persistent settings saved between sessions
- One-click launcher for Conda environments (no install hassle)

### Formats & Aspect Ratio Support
- Output formats: Full-SBS, Half-SBS, Red/Cyan Anaglyph
- Aspect Ratios: 16:9, 2.39:1, 21:9, 4:3, 1:1, 2.35:1, 2.76:1
- Codec selection: MP4V, XVID, MJPG, DIVX


---

## Guide Sheet: Install

### 📌 System Requirements
- ✔️ This program runs on python 3.12
- ✔️ This program has been tested on cuda 12.8
- ✔️ Conda (Optional, Recommended for Simplicity)
- ❌ Linux/macOS is not officially supported until a more stable solution is found

### 📌 Step 1: Download the VisionDepth3D Program
- 1️⃣ Download the VisionDepth3D zip file from the official download source. (green button)
- 2️⃣ Extract the zip file to your desired folder (e.g., c:\user\VisionDepth3D).
- 3️⃣ Download RIFE model [Here](https://drive.google.com/file/d/16SLYOgHw5VSBp1UgmGRLkBBKJQD-hGZW/view?usp=sharing) and put in weights folder


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

if you are running Cuda 12.8 install Pytorch(nightly)-Cuda 12.8, if that doesnt work use 12.6 version

- Once Pytorch and all dependancies are installed run the following command:
```
VisionDepth3D.bat
```
Congrats you have successfully downloaded VisionDepth3D! 
This snippet guides users through cloning the repo, creating and activating the environment, and running the app—all in a few simple steps.

---

### Guide Sheet: GUI Inputs
Below is a guide to help you understand and adjust each parameter in the GUI.

### 1. Codec
- **Description**: Specifies the codec used for encoding the output video.
- **Default**: `mp4v`
- **Options**: 
  - `mp4v` (MP4 format)
  - `XVID` (MKV format)
  - Others supported by OpenCV.


### 2. Divergence Shift
- **Description**: Controls the amount of pixel shift for objects in the foreground.
- **Default**: `4.8`
- **Recommended Range**: `3.0` to `8.0`
- **Effect**: Higher values create a stronger 3D effect for objects closest to the viewer.


### 3. Depth Transition 
- **Description**: Controls the amount of pixel shift for midground objects.
- **Default**: `1.9`
- **Recommended Range**: `1.0` to `5.0`
- **Effect**: Fine-tune this value to balance the depth effect between divergence and convergence.

### 4. Convergence Shift
- **Description**: Controls the amount of pixel shift for Depth.
- **Default**: `-2.8`
- **Recommended Range**: `-5.0` to `0.0`
- **Effect**: Use Negative values to push objects farther back, creating more depth.

### 5. Sharpness Factor
- **Description**: Adjusts the sharpness of the final output video.
- **Default**: `0`
- **Range**: `-1.0` (softer) to `1.0` (sharper)
- **Effect**: Higher values make edges more distinct, but excessive sharpness can introduce artifacts like over brightness.

### 6. Blend Factor (Pulfrich Effect)
- **Description**: Controls the blending ratio between delayed and current frames for the Pulfrich effect.
- **Default**: `0.5`
- **Recommended Range**: `0.3` (subtle) to `0.7` (strong effect)
- **Effect**: Higher values enhance the Pulfrich effect but may make scene transitions harder on the eyes.

### 7. Delay Time (Pulfrich Effect)
- **Description**: Specifies the temporal delay (in seconds) to create the Pulfrich effect.
- **Default**: `1/30`
- **Recommended Range**: `1/50` to `1/20`
- **Effect**: Lower values (e.g., `1/50`) reduce the delay, creating a more subtle effect.


## Depth Map File Requirements
### 1. Create a Depth map from Depth Generator Tab or use an existing
**Lots of models to choose from!** 

Some Models Inverse the depth map, check a frame before rendering a whole clip
and toggle inverse checkbox to generate the Depthmap as higher contrast/white(close object) and lower contrast/black (Far objects) 
- *Distil Any Depth Large*
- *Depth Anything V2 Large*
- *Depth Anything V2 Base*
- *Depth Anything V2 Small*
- *Depth Anything V1 Large*
- *Depth Anything V1 Base*
- *Depth Anything V1 Small*
- *Depth-Anything-V2-Metric-Indoor-Large*
- *Depth-Anything-V2-Metric Outdoor-Large*
- *depth_anything_vitl14*
- *depth_anything_vits14*
- *DepthPro*
- *ZoeDepth*
- *MiDaS 3.0*
- *DPT-Large*
- *dpt-dinov2-small-kitti*
- *dpt-beit-large-512*


## Processing Times
- **Estimated Times**:
  - A 30-second clip: ~1-4mins.
  - Full-length videos: ~5-24 hours+.
plus 3D render time	

---

## Example Workflow
1. Select your input video (`.mp4`, `.avi`, `.mkv`) and output file path.
2. Select your Depth map Video, make sure both video files are same width and  height and FPS 
3. Adjust rendering parameters for the desired 3D effect.
4. Click "Generate 3D SBS Video" to process.

## Troubleshooting
- **Black Screens or Artifacts**:
  - Ensure the depth map matches the input video's resolution and frame rate.
  - Adjust `blend_factor` and `delay_time` for smoother transitions between scenes. this effect is supposed to 

---

## Notes
- Active Development: This project is constantly evolving. If you encounter any issues, have questions, or suggestions, please feel free to start a conversation in the Discussions tab. Your feedback is always appreciated!
  
- Solo Developer Notice: As a solo developer working on this project during my limited free time (~4 hours per night), I truly appreciate your patience and understanding as I continue to improve the software, squash bugs, and fine-tune features to deliver the best 3D rendering experience possible.

## Pulfrich Effect Explained
- **How to Use the Pulfrich Effect in Your Program**
-- The Pulfrich effect in your program creates a dynamic 3D experience by introducing a temporal delay between the left and right views, simulating depth perception based on motion. Here's how to use it effectively:

- **Enable Pulfrich Effect**: The effect is automatically applied when generating Half-SBS 3D videos.
It works by blending current and delayed frames to enhance depth perception during motion.

- **Adjust Blend Factor**: Use the Blend Factor slider or parameter to control the intensity of the Pulfrich effect.
Higher values increase the blending between delayed and current frames, while lower values reduce it.

- **Scene Change Handling**: the program detects scene changes automatically and dynamically reduces blending to avoid artifacts in abrupt transitions.
No manual intervention is required for smooth scene transitions.

- **Delay Time Control**: Modify the Delay Time parameter to fine-tune the temporal offset.
A smaller delay creates subtle depth, while a larger delay produces more pronounced effects.

## Acknowledgments & Credits

I want to express my gratitude to the amazing creators and contributors behind the depth estimation models used in this project. Your work has made it possible to push the boundaries of 3D rendering and video processing. 🙌

### **Supported Depth Models**
| Model Name | Creator / Organization | Hugging Face Repository |
|------------|------------------------|-------------------------|
| **Distil-Any-Depth-Large** | xingyang1 | [Distil-Any-Depth-Large-hf](xingyang1/Distill-Any-Depth-Large-hf) |
| **Distil-Any-Depth-Small** | xingyang1 | [Distil-Any-Depth-Small-hf](xingyang1/Distill-Any-Depth-Small-hf) |
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


