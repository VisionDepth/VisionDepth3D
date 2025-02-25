# VisionDepth3Dv2 - AI-Powered 3D Video Conversion with Dynamic Depth Mapping & Pulfrich Effect
VisionDepth3Dv2 transforms 2D videos into immersive 3D experiences using AI-driven depth mapping, advanced Pulfrich effect blending, and multi-format 3D rendering including Half-SBS, Full-SBS, Over-Under, VR, and Red-Cyan Anaglyph. Perfect for VR headsets, 3D displays, and cinematic presentations, VisionDepth3Dv2 offers real-time rendering controls, AI-powered convergence correction, and customizable depth effects — all within an intuitive GUI.

With real-time depth adjustments, the system dynamically modifies foreground divergence, midground depth transition, and background convergence, ensuring a cinematic and immersive 3D experience. The integration of a deep-learning warp model refines depth shifts, improving background separation for more natural parallax effects.

### "Transform your 2D moments into stunning 3D realities — with AI precision and cinematic depth."


## 🚀 Key Features

### 🔄 Multi-Format 3D Output 
- Full-SBS (Side-by-Side) – Best for high-quality 3D displays (7680x1608 for 8K resolution).
- Half-SBS (HSBS) – Optimized for 1080p displays (1920x1080 total).
- Full-OU (Over-Under) – Suitable for vertical 3D viewing (3840x3216).
- Half-OU – Optimized for lower bandwidth with halved vertical resolution.
- Red-Cyan Anaglyph – True balanced anaglyph for standard 3D glasses.
- VR Format – Supports Oculus Quest 2 and similar VR headsets (2880x1600 resolution).

## 🎚 Advanced Depth-Based Rendering
- Dynamic depth-based pixel shifting for realistic parallax effects.
- Foreground, midground, and background shift customization for precise depth control.
- Adjustable Pulfrich effect blending for smooth motion-based depth illusions.

## 🎨 AI-Powered Convergence Correction
- Utilizes a TensorFlow deep learning model to correct divergence for comfortable 3D viewing.
- Automatic depth normalization and bilateral filtering for cleaner depth maps.

## 🖼 Aspect Ratio Support
-Select from cinematic aspect ratios including:
- 16:9 (Default)
- 2.39:1 (CinemaScope)
- 21:9 (UltraWide)
- 4:3 (Classic Films)
- 1:1 (Square)
- 2.35:1 (Classic Cinematic)
- 2.76:1 (Ultra-Panavision)

## 🛠 Smart Pre-Processing
- Automatic black bar detection and removal to ensure full-frame 3D rendering.
- White edge correction using median blur for professional visual consistency.

## ⚡ Real-Time Performance Insights
- Real-time FPS tracking with adaptive smoothing for accurate performance feedback.
- Interactive progress bar and percentage completion updates during rendering.

## 💾 Persistent User Settings
- All rendering preferences, including depth shifts, sharpness, and delay times, are automatically saved and restored on - - relaunch via JSON settings.

## 🎛 Flexible Codec Support
- Choose from MP4 (mp4v, H264), AVI (XVID, DIVX), and MKV for wide playback compatibility.

## 🖱 Interactive Tkinter GUI
- Intuitive drag-and-drop interface with live video preview thumbnails.
- Real-time pause, resume, and cancel controls for rendering operations.
- Quick GitHub access for updates and support.

## 🧩 Optimized for Efficiency
- Threaded rendering with pause/resume functionality for optimized system resource management.

Designed for VR enthusiasts, stereoscopic filmmakers, and 3D content creators, VisionDepth3D provides fine-tuned control over depth parameters, delivering professional-quality 3D video conversions with precision and efficiency.

GUI Layout
--
![GUILayoutV2](https://github.com/user-attachments/assets/7576866f-e655-48b8-ab15-bf34d9156825)


## Guide Sheet: Install

Installation Steps
This program runs on python 3.9 - 3.10
pip required to install dependancies

### Step 1: Download the VisionDepth3Dv2 Program
- Download the VisionDepth3D zip file from the official download source. (green button)
- Extract the zip file to your desired folder (e.g., c:\user\VisionDepth3D).
- Download Backwards warp model [Here](https://drive.google.com/file/d/1x2JApPfOcUA9EGLGEZK-Bzgur7KkGrTR/view?usp=sharing) and put in weights folder

### Step 2: Install Required Dependencies 
1. **press (Win + R), type cmd, and hit Enter.**

2. **Navigate to the Program Directory:**
```
cd C:\user\VisionDepth3D
```
3. **Install Dependencies:**
```
pip install -r requirements.txt
```

### Conda Install (easiest method) ###
1. **Clone the Repository:**
   ```bash
   git clonehttps://github.com/VisionDepth/VisionDepth3D.git
   cd VisionDepth3D
2. **Create the Conda Environment:**
   We provide an environment.yml file that installs all required dependencies. To create the environment, run:
    ```bash
    conda env create -f environment.yml
4. **Activate the Environment:**
   ```bash
   conda activate visiondepth3d
5. **Run VisionDepth3D:**
   ```bash
   python VisionDepth3Dv2.1.py

This snippet guides users through cloning the repo, creating and activating the environment, and running the app—all in a few simple steps.


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
### 1. Just Have a Depth map Generated I suggest looking at
- **Depth Anything V2
- **Midas Models
- **DPT Models

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

## 📝 Notes
- 🚀 Active Development: This project is constantly evolving. If you encounter any issues, have questions, or suggestions, please feel free to start a conversation in the Discussions tab. Your feedback is always appreciated!
  
- 👨‍💻 Solo Developer Notice: As a solo developer working on this project during my limited free time (~4 hours per night), I truly appreciate your patience and understanding as I continue to improve the software, squash bugs, and fine-tune features to deliver the best 3D rendering experience possible.
  
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
