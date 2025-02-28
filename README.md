# VisionDepth3Dv2 - AI-Powered 3D Video Conversion with Dynamic Depth Mapping & Pulfrich Effect
VisionDepth3Dv2 transforms 2D videos into immersive 3D experiences using AI-driven depth mapping, advanced Pulfrich effect blending, and multi-format 3D rendering including Half-SBS, Full-SBS, Over-Under, VR, and Red-Cyan Anaglyph. Perfect for VR headsets, 3D displays, and cinematic presentations, VisionDepth3Dv2 offers real-time rendering controls, AI-powered convergence correction, and customizable depth effects — all within an intuitive GUI.

With real-time depth adjustments, the system dynamically modifies foreground divergence, midground depth transition, and background convergence, ensuring a cinematic and immersive 3D experience. The integration of a deep-learning warp model refines depth shifts, improving background separation for more natural parallax effects.

### "Transform your 2D moments into stunning 3D realities — with AI precision and cinematic depth."

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVisionDepth%2FVisionDepth3D&count_bg=%23AA1400&title_bg=%235E5E5E&icon=&icon_color=%23ACAAAA&title=Page+Views&edge_flat=false)](https://hits.seeyoufarm.com)

## 🚀 Key Features
### 🔄 Multi-Format 3D Output
Seamlessly generate 3D videos in multiple industry-standard formats, tailored for a variety of 3D displays, VR headsets, and anaglyph glasses:

- 🖥️ Full-SBS (Side-by-Side) – Ideal for high-quality 3D displays and projectors, providing the highest resolution per 
  eye. (7680x1608 for 8K output)
- 🎬 Half-SBS (HSBS) – Optimized for 1080p displays like 3D TVs and monitors. Reduces bandwidth while maintaining crisp 
  image quality.
- 📺 Full-OU (Over-Under) – Perfect for vertical 3D display setups, ensuring full-resolution per eye. (3840x3216 total)
- 📉 Half-OU – Designed for lower bandwidth streaming and mobile VR with halved vertical resolution, preserving visual 
  quality while reducing file size.
- 🎨 Red-Cyan Anaglyph – Generates true balanced anaglyph 3D compatible with standard Red-Cyan glasses, perfect for 
  classic 3D viewing experiences.
- 🕶️ VR Format – Optimized for VR headsets like Oculus Quest 2, providing an immersive experience with per-eye resolution 
  of (1440x1600) for crystal-clear virtual environments. (2880x1600 total)

## 🎚 Advanced Depth-Based Rendering
- 🌌 Dynamic Depth-Based Pixel Shifting: Generates realistic parallax effects by shifting pixels based on depth data, 
  creating a convincing 3D experience with natural scene separation.
- 🎛 Customizable Depth Layers: Fine-tune foreground, midground, and background shifts to control the intensity and depth 
  perception, offering precision 3D rendering tailored to your visual preferences.
- 🌊 Adjustable Pulfrich Effect Blending: Incorporates the Pulfrich effect with customizable blending, delivering smooth 
  motion-based depth illusions that enhance the 3D impact during fast-moving scenes.
- 🏃 Frame-Accurate Depth Mapping: Processes depth data per frame for smoother transitions and consistent depth 
  perception, even in complex, high-motion videos.

## 🎨 AI-Powered Convergence Correction
- ⚡ TensorFlow-Driven Precision: Leverages a deep learning model to automatically correct divergence issues, ensuring 
  comfortable, eye-strain-free 3D viewing with natural depth perception.
- 🧪 Smart Depth Normalization: Dynamically normalizes depth maps for consistent depth scaling across frames, delivering 
  smoother 3D transitions.
- 🎛 Bilateral Filtering for Clarity: Applies advanced bilateral filtering to eliminate noise while preserving important 
  edges, resulting in sharper, cleaner 3D visuals.
- 🎯 Adaptive Convergence Shifts: Adjusts convergence points based on scene depth, enhancing realism by ensuring objects 
  at varying depths appear comfortably aligned.

## 🖼 Aspect Ratio Support
Achieve the perfect cinematic look with a variety of predefined aspect ratios, tailored for different viewing experiences and display formats:

- 🎞️ 16:9 (Default) – Standard widescreen format, ideal for HD and UHD displays, including most modern TVs, monitors, and 
  online video platforms.
- 🍿 2.39:1 (CinemaScope) – Epic cinematic aspect ratio, commonly used in blockbuster movies for a widescreen, immersive 
  experience.
- 🖥️ 21:9 (UltraWide) – Perfect for ultrawide monitors and theater-like experiences, offering extra horizontal space for 
  enhanced immersion.
- 🎥 4:3 (Classic Films) – The classic TV and film format, bringing a retro aesthetic to your 3D videos, ideal for 
  archival and artistic projects.
- 🔲 1:1 (Square) – Balanced and modern, perfect for social media platforms like Instagram and TikTok, offering a 
  versatile framing option.
- 🎬 2.35:1 (Classic Cinematic) – A traditional cinematic ratio that offers a widescreen look with subtle letterboxing, common in classic Hollywood films.
- 🎞 2.76:1 (Ultra-Panavision) – Ultra-wide format used in iconic epic films, providing a dramatic and expansive field of 
view for cinematic storytelling.

## 🛠 Smart Pre-Processing
Ensure flawless 3D rendering with intelligent pre-processing features designed for professional-grade visual consistency:

- 🎯 Automatic Black Bar Detection & Removal – Automatically detects and removes letterboxing (black bars) from input videos, ensuring that every frame is fully utilized for a true full-frame 3D experience without unwanted borders.
- 🎨 White Edge Correction with Median Blur – Detects and corrects white edges often caused by cropping or aspect ratio adjustments. Median blur filtering is applied to seamlessly blend edges, resulting in cleaner frames and a polished, professional look.

## ⚡ Real-Time Performance Insights
Monitor your rendering performance in real-time with intuitive feedback tools:

- ⏱️ Real-Time FPS Tracking – Displays live frames-per-second (FPS) data with adaptive smoothing for accurate and stable performance feedback.
- 📊 Interactive Progress Indicators – Track rendering progress effortlessly with a dynamic progress bar and percentage completion updates, keeping you informed at every stage.

## 💾 Persistent User Settings
Save time with automatically retained preferences, ensuring a consistent user experience across sessions:

- 💡 Auto-Save Settings – All rendering preferences, including depth shifts, sharpness levels, blend factors, and delay times, are saved to a JSON file and restored on relaunch for hassle-free continuity.

## 🎛 Flexible Codec Support
Enjoy broad playback compatibility with support for popular video codecs:

- 🎞️ MP4 – Supports mp4v and H264 for high-quality compression and widespread compatibility.
- 🎬 AVI – Offers XVID and DIVX options for classic playback and legacy systems.
- 🎥 MKV – Ideal for high-definition video with rich metadata support and multi-track capabilities.

## 🖱 Interactive Tkinter GUI
A modern, intuitive interface designed for effortless 3D video generation:

- 🖱️ Drag-and-Drop Simplicity – Easily load videos with drag-and-drop support and real-time thumbnail previews.
- ⏸ Interactive Controls – Pause, resume, or cancel rendering in real time, offering full control over the process.
- 🌐 GitHub Integration – One-click GitHub access for the latest updates, documentation, and community support.

## 🧩 Optimized for Efficiency
- 🔀 Threaded Rendering Architecture – Utilizes multi-threading to ensure smooth performance, allowing you to pause and resume rendering without restarting the entire process.

Designed for VR enthusiasts, stereoscopic filmmakers, and 3D content creators, VisionDepth3D provides fine-tuned control over depth parameters, delivering professional-quality 3D video conversions with precision and efficiency.

GUI Layout
--
![GUILayoutV2](https://github.com/user-attachments/assets/7576866f-e655-48b8-ab15-bf34d9156825)


## ✅ VisionDepth3D Installation Guide

Installation Steps

### 📌 System Requirements
- ✔ Python 3.9 - 3.10 (Required)
- ✔ pip (Required for dependency installation)
- ✔ Conda (Optional, Recommended for Simplicity)

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
- 
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
- 3️⃣ **Install All Dependencies:**
```
pip install -r requirements.txt
```

## 🔵 Option 2: Install via Conda (Recommended)
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
To make launching easier, a one-click .bat file is included:
- ✔ Instead of manually running commands, just double-click:
- 📂 start_visiondepth3d.bat inside the VisionDepth3D folder.

***📌 What the .bat file does:***
- Automatically detects if Conda is installed
- Activates Conda environment if available
- Runs VisionDepth3D using either Conda or standard Python

### 📌 Step 4: (Optional) Create a Desktop Shortcut
- ✅ Right-click start_visiondepth3d.bat → Create Shortcut
- ✅ Move the shortcut to your Desktop
- ✅ (Optional) Right-click → Properties → Change "Run" to "Minimized" to hide the CMD window.

### 📌 Step 5: (Optional) Create a Desktop Shortcut
- 1️⃣ Right-click start_visiondepth3d.bat → Create Shortcut
- 2️⃣ Move the shortcut to your Desktop
- 3️⃣ (Optional) Right-click → Properties → Change "Run" to "Minimized" to hide the CMD window.

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
