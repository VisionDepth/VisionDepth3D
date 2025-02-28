# VisionDepth3D v3 - AI-Powered 3D Video Converter
## 🎥 Convert Any 2D Video into Immersive 3D!

VisionDepth3D v3 uses AI-powered depth estimation to generate stunning, multi-format 3D videos, optimized for:
- ✅ VR Headsets (Quest, SteamVR, etc.)
- ✅ 3D TVs & Projectors (Full-SBS, Half-SBS, Over-Under)
- ✅ Red-Cyan Glasses (Anaglyph)
- ✅ Cinematic Experiences (2.39:1, 21:9, 16:9 & more)
  
- 🔹 Supports GPU acceleration (CUDA) for faster processing
- 🔹 Advanced Pulfrich effect blending for motion-based 3D depth
- 🔹 AI-powered convergence correction for natural 3D separation
- 🔹 Simple drag-and-drop GUI with real-time rendering controls
  
### 🎯 "Turn your 2D moments into immersive 3D realities—AI precision, cinematic depth, and VR-ready output!" 🚀"

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVisionDepth%2FVisionDepth3D&count_bg=%23AA1400&title_bg=%235E5E5E&icon=&icon_color=%23ACAAAA&title=Page+Views&edge_flat=false)](https://hits.seeyoufarm.com)

## 🚀 Key Features
### 🔄 Multi-Format 3D Output
Convert 2D videos into multiple industry-standard 3D formats, compatible with VR headsets, 3D displays, and anaglyph glasses:

- 🖥️ Full-SBS (Side-by-Side) – Highest quality for 8K/4K projectors & 3D displays
- 🎬 Half-SBS (HSBS) – Optimized for 1080p 3D TVs & monitors
- 📺 Full-OU (Over-Under) – Perfect for vertical 3D setups
- 📉 Half-OU – Lower bandwidth streaming & mobile VR compatibility
- 🎨 Red-Cyan Anaglyph – Viewable on any standard screen with 3D glasses
- 🕶️ VR Format – Optimized for Oculus Quest, SteamVR & more
  
## 🎚 Advanced Depth-Based Rendering
- 🌌 AI-Powered Depth Pixel Shifting – Dynamic depth-based parallax effects
- 🎛 Customizable Depth Controls – Fine-tune foreground, midground, and background shifts
- 🌊 Pulfrich Effect Blending – Motion-based depth enhancement for smoother cinematic transitions
- 🏃 Frame-Accurate Depth Mapping – Per-frame depth consistency & scene correction
  
## 🎨 AI-Powered Convergence Correction
- ⚡ Deep-Learning Warp Model – Auto-corrects divergence for natural 3D separation
- 🧪 Smart Depth Normalization – Dynamic depth scaling per frame
- 🎛 Bilateral Filtering – Sharpens depth maps & reduces noise artifacts

## 🖼 Aspect Ratio Support
Select from a variety of aspect ratios for cinematic and immersive experiences:
- 🎞️ 16:9 (Default) – Standard HD/UHD format
- 🍿 2.39:1 (CinemaScope) – Widescreen cinematic experience
- 🖥️ 21:9 (UltraWide) – Perfect for ultrawide monitors
- 🎥 4:3 (Classic Films) – Retro 3D format
- 🔲 1:1 (Square) – Social media-friendly format
- 🎬 2.35:1 & 2.76:1 – Cinematic widescreen options

## 🛠 Smart Pre-Processing
- ✔ 🎯 Automatic Black Bar Detection & Removal (Removes letterboxing for true full-frame 3D!)
- ✔ 🎨 White Edge Correction (Blends edges seamlessly with median blur!)

## ⚡ Real-Time Performance Insights
Monitor your rendering performance in real-time with intuitive feedback tools:
- ✔ ⏱️ FPS Tracker (Displays real-time frames-per-second speed!)
- ✔ 📊 Interactive Progress Indicators (Live tracking of render progress!)

## 💾 Persistent User Settings
- ✔ 🔄 Auto-Save Preferences (Restores previous depth settings on relaunch!)

## 🖱 Interactive Tkinter GUI
- ✔ Drag-and-Drop Simplicity (Easily load videos with real-time thumbnails!)
- ✔ ⏸ Live Controls (Pause, resume, or cancel rendering anytime!)

GUI Layout
--

![Tab1](https://github.com/user-attachments/assets/259a169d-fd99-4098-b08b-554dd4ea705f)
![Tab2](https://github.com/user-attachments/assets/80296073-2f6b-4d00-a90d-69ed6c687368)


## ✅ VisionDepth3D Installation Guide

Installation Steps

### 📌 System Requirements
- ✔ Python 3.9 - 3.10 (Required)
- ✔ pip (Required for dependency installation)
- ✔ Conda (Optional, Recommended for Simplicity)
- ✔ Linux/macOS users may need additional dependencies (see Step 3).

### 📌 Step 1: Download & Extract VisionDepth3D
- 1️⃣ Download the latest VisionDepth3D ZIP file from the official repository (green "Download" button).
- 2️⃣ Extract the ZIP file to a folder of your choice (e.g., C:\VisionDepth3D).
- 3️⃣ Download the Backwards Warp Model and place it in the weights folder: [Here](https://drive.google.com/file/d/1x2JApPfOcUA9EGLGEZK-Bzgur7KkGrTR/view?usp=sharing)


### 📌 Step 2: Install PyTorch with CUDA Support 
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
  
### 📌 Install PyTorch with the Correct CUDA Version ### 
Go to the official PyTorch website to find the best install command for your setup:
🔗 [ https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Step 3: Install Required Dependencies 
🟢 Option 1: Install via pip (Standard CMD Method)
- 1️⃣ **press (Win + R), type cmd, and hit Enter.**
- 2️⃣ **Navigate to the Program Directory:**
```
cd C:\VisionDepth3D
```
Linux/macOS
```
cd ~/VisionDepth3D 
```

- 3️⃣ **Install All Dependencies:**
```
pip install -r requirements.txt
```
- 🔵 Linux/macOS users may need to install additional dependencies:
```
sudo apt-get install python3-tk ffmpeg   # Ubuntu/Debian
brew install python-tk ffmpeg            # macOS (requires Homebrew)
```

## 🔵 Option 2: Install via Conda (Recommended)

(Automatically manages dependencies & isolates environment.)
- **1️⃣ Clone the Repository (Skip this if you downloaded the ZIP)**
   ```bash
   git clone https://github.com/VisionDepth/VisionDepth3D.git
   cd VisionDepth3D
   ```
- **2️⃣ Create the Conda Environment (Automatically installs dependencies)**
   We provide an environment.yml file that installs all required dependencies. To create the environment, run:
    ```bash
    conda env create -f environment.yml
    ```
- **3️⃣ Activate the Conda Environment**
   ```bash
   conda activate visiondepth3d
   ```
- **4️⃣ Run VisionDepth3D**
   ```bash
   python VisionDepth3Dv3.py
   ```

### 📌 Step 3: One-Click Launch (Recommended)
** Windows:**
To make launching easier, a one-click .bat file is included:
- ✔ Double-click start_visiondepth3d.bat inside the VisionDepth3D folder

***📌 What the .bat file does:***
- Automatically detects if Conda is installed
- Activates Conda environment if available
- Runs VisionDepth3D using either Conda or standard Python

** Linux/macOS:**
- ✔ Use the included .sh script for one-click launch:
- 1️⃣ Give execution permission (only needed once):
```
chmod +x start_visiondepth3d.sh
```
**  ⃣ Run VisionDepth3D:**
```
./start_visiondepth3d.sh
```

### 📌 Step 4: (Optional) Create a Desktop Shortcut
** Windows:** 
- ✅ Right-click start_visiondepth3d.bat → Create Shortcut
- ✅ Move the shortcut to your Desktop
- ✅ (Optional) Right-click → Properties → Change "Run" to "Minimized" to hide the CMD window.

** Linux/macOS:**
- 1️⃣ Open Terminal and create a shortcut:
```
ln -s ~/VisionDepth3D/start_visiondepth3d.sh ~/Desktop/VisionDepth3D
```
- 2️⃣ (Optional) Make a .desktop shortcut:
```
echo '[Desktop Entry]
Name=VisionDepth3D
Exec=/home/user/VisionDepth3D/start_visiondepth3d.sh
Type=Application
Terminal=true' > ~/Desktop/VisionDepth3D.desktop
chmod +x ~/Desktop/VisionDepth3D.desktop
```

🔥 Now you can launch VisionDepth3D in one click from your Desktop!

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
