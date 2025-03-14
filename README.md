# <h1 align="center">VisionDepth3D</h1>  
## <h2 align="center">3D Video Converter and Depth map Generator </h2> 
### <h3 align="center">[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVisionDepth%2FVisionDepth3D&count_bg=%23AA1400&title_bg=%235E5E5E&icon=&icon_color=%23ACAAAA&title=Page+Views&edge_flat=false)](https://hits.seeyoufarm.com) [![Github All Releases](https://img.shields.io/github/downloads/VisionDepth/VisionDepth3D/total.svg)]() ![Python Version](https://img.shields.io/badge/python-3.12-blue)</h3>

---

## **GUI Layout**
![GUITabsSBS](https://github.com/user-attachments/assets/337a6bd3-43ad-4f25-ab63-4563387305d6)

---

## Table of Contents
- [Key Features](#key-features)
- [Advanced Depth Estimation & 3D Processing](#advanced-depth-estimation--3d-video-processing)
- [GUI Layout](#gui-layout)
- [System Requirements](#-system-requirements)
- [Installation](#guide-sheet-installation)
- [GUI Settings & Adjustments](#guide-sheet-gui-inputs)
- [Pulfrich Effect Explained](#pulfrich-effect-explained)
- [Troubleshooting](#troubleshooting)
- [Dev Notes](#notes)
- [Acknowledgments & Credits](#acknowledgments--credits)

## Key Features

### Depth Estimation Models via Transformers
 - Multi-Model AI Support – Choose from cutting-edge depth estimation models like Depth Anything V2, MiDaS 3.0, ZoeDepth, DinoV2, and more.
 - Real-Time Depth Processing – GPU-accelerated estimation with dynamic scaling for enhanced efficiency.
 - Simple pick and download Depth model and cached for future uses
 - Adaptive Depth Smoothing – Intelligent filtering reduces noise while preserving sharp depth details.
 - Customizable Depth Formats – Export in Full-SBS, Half-SBS, Full-OU, Half-OU, Anaglyph 3D, or VR-optimized formats.
 - Precision Depth Convergence – Advanced background isolation and convergence shift correction for realistic 3D results.
 - Batch Video Processing – Accelerate video depth conversion with optimized batch inference.

### Advanced Depth Estimation & 3D Video Processing
 - AI-Powered Depth Shifting – Generate precise depth-based parallax effects for immersive 3D visuals.
 - Customizable Depth Mapping – Fine-tune foreground, midground, and background shifts for accurate depth perception.
 - Pulfrich Effect Blending – Motion-aware depth enhancement for fluid cinematic depth transitions.
 - Frame-Accurate Depth Tracking – Consistent per-frame depth mapping with smart scene correction for precise rendering.
  
### AI-Powered Convergence Correction
 - Deep-Learning Warp Model – Auto-corrects divergence for natural 3D separation
 - Smart Depth Normalization – Dynamic depth scaling per frame
 - Bilateral Filtering – Sharpens depth maps & reduces noise artifacts

### Aspect Ratio Support
**Select from a variety of aspect ratios for cinematic and immersive experiences:**
 - 16:9 (Default) – Standard HD/UHD format
 - 2.39:1 (CinemaScope) – Widescreen cinematic experience
 - 21:9 (UltraWide) – Perfect for ultrawide monitors
 - 4:3 (Classic Films) – Retro 3D format
 - 1:1 (Square) – Social media-friendly format
 - 2.35:1 & 2.76:1 – Cinematic widescreen options

### Smart Pre-Processing
- Automatic Black Bar Detection & Removal (Removes letterboxing for true full-frame 3D!)
- White Edge Correction (Blends edges seamlessly with median blur!)

### Real-Time Performance Insights
**Monitor your rendering performance in real-time with intuitive feedback tools:**
- FPS Tracker (Displays real-time frames-per-second speed!)
- Interactive Progress Indicators (Live tracking of render progress!)

### Persistent User Settings
- Auto-Save Preferences (Restores previous depth settings on relaunch!)

### Interactive Tkinter GUI
- Slider Controls for Divergence shift, Depth Transition, Convergence shift, Pulfrich effect and Frame blending
- Live Controls (Pause, resume, or cancel rendering anytime!)

---

#  Guide Sheet: Installation

## 🔹 System Requirements
- ✔️ **Python 3.12** (Required)
- ✔️ **CUDA 12.8** (Tested & Recommended)
- ✔️ **pip** (Required to install dependencies)
- ✔️ **Conda** (Recommended for Simplicity)
- ❌ **Linux/macOS is not officially supported** (until a more stable solution is found)

---

## 🔹 Step 1: Download the VisionDepth3D Program
### ⚠️ Make Sure to not install opencv-python, this will overwrite GPU Support
### If you Have VD3D environment set up already delete it and start fresh

   ```
 conda remove -n VD3D --all
   ```

1️⃣ **Download the VisionDepth3D ZIP file** from the official GitHub repository (Green button).  
2️⃣ **Extract** the ZIP file to your desired folder, e.g., `C:User\VisionDepth3D-Main`.  
3️⃣ **Download the Backwards Warp Model**  
   - [Click Here to Download](https://drive.google.com/file/d/1BbcWwWR0IrQEIIuVFiTB0biosYn5DcOz/view?usp=sharing)  
   - Move the downloaded file into the **`weights`** folder inside VisionDepth3D.

4️⃣ **Create Environment**
   - This will **create the `VD3D` Conda environment** and install all dependencies

   ```
conda create -n VD3D python=3.12
conda activate VD3D
cd C:\User\VisionDepth3D-Main # Change to Path of main script
pip install -r requirements.txt
   ```

---

## 🔹 Step 2: Install OpenCV with GPU Acceleration

### **Download the OpenCV GPU Package**
 **Download the ZIP file from GitHub Releases:**  
🔗 [OpenCV GPU Package v4.12.0](https://github.com/VisionDepth/openCV-GPU/releases/download/v4.12.0/opencv_gpu_custom-4.12.0.zip)

### **Extract & Install OpenCV**
1️⃣ **Extract** `opencv_gpu_custom-4.12.0.zip` to VisionDepth3D-Main Folder.  
2️⃣ **Run `install_cv2.bat`** (In open environment)  

   ```
cd C:\User\VisionDepth3D-Main\opencv_gpu_release
install_cv2.bat
   ``` 

   - This will **install OpenCV GPU support into the environment**.

✅ **Now, OpenCV GPU is installed!**

##🔹 Step 3: Install PyTorch with the Correct CUDA Version  
Go to the official PyTorch website to find the best install command for your setup:
🔗 [ https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
and install inside created VD3D environment

⚠️ sometimes "pip3" doesnt work just change it to "pip"

if you are running Cuda 12.8 you can install Pytorch Cuda 12.6 and it will work

## 🔹 Step 4: **Open VisionDepth3D**

   ```
cd C:\User\VisionDepth3D-Main
python VisionDepth3D.py
   ``` 

*This snippet guides users through cloning the repo, creating and activating the environment, and running the app—all in a few simple steps.*

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
| **Distil-Any-Depth-Large** | Keetrap | [Distil-Any-Depth-Large-hf](https://huggingface.co/keetrap/Distil-Any-Depth-Large-hf) |
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


