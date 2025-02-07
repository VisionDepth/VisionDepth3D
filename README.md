# VisionDepth3D - 3D Video Conversion with AI Depth Mapping & Pulfrich Effect
VisionDepth3D AI-powered 3D converter is a high-performance tool for creating Half Side-by-Side (H-SBS) and Over-Under (OU) 3D videos with advanced depth-based rendering, Pulfrich effect simulation, it intelligently enhances depth perception by reducing halo artifacts and preserving object integrity.

With real-time depth adjustments, the system dynamically modifies foreground divergence, midground depth transition, and background convergence, ensuring a cinematic and immersive 3D experience. The integration of a deep-learning warp model refines depth shifts, improving background separation for more natural parallax effects. Runs on mid-range hardware, this open-source 3D tool is optimized for efficiency!

Key features include:
- ✅ Real-time motion-based depth enhancements for pixel-accurate shifting
- ✅ AI-powered convergence correction to refine depth shifts dynamically
- ✅ Pulfrich Effect simulation for enhanced motion-based depth perception
- ✅ Intelligent edge blending for smoother object transitions
- ✅ Cinematic aspect ratio support (2.39:1, 21:9, 4:3, 1:1, etc.)
- ✅ VR-ready formats (optimized for Oculus, SteamVR, and Meta Quest)
- ✅ Advanced processing optimizations for mid-range and high-end GPUs
- ✅ Full audio synchronization with seamless 3D conversion


Designed for VR enthusiasts, stereoscopic filmmakers, and 3D content creators, VisionDepth3D provides fine-tuned control over depth parameters, delivering professional-quality 3D video conversions with precision and efficiency.

GUI Layout

![GuiSetupv3](https://github.com/user-attachments/assets/7186246a-917f-4e3d-b54b-d8a145fb30b3)


## Guide Sheet: Install
Installation Steps
This program runs on python 3.9 - 3.10

### Step 1: Download the VisionDepth3D Program
- Download the VisionDepth3D.zipfile from the official download source.
- Extract the zip file to your desired folder (e.g., C:\VisionDepth3D).
- Download backwards warp model [👉 here](https://drive.google.com/file/d/1Ff0py6EpTG7IcLDQE9Brl9d3002Hd3JO/view?usp=sharing)

### Step 2: Install Required Dependencies

### Open a terminal or command prompt and enter:
- git clone https://github.com/VisionDepth/VisionDepth3D.git
- cd VisionDepth3D
- install_visiondepth3d.bat

### or alternatively you can use pip**
- pip install git+https://github.com/VisionDepth/VisionDepth3D.git
- pip install -r requirements.txt

### use conda by setting up an environment (best method)
- open conda
- enter cd Path/To/Visiondepth3D
- enter conda create --name VisionDepth3D python=3.10
- enter conda activate VisionDepth3D
- pip install -r requirements.txt
- enter python VisionDepth3D.py



### Guide Sheet: GUI Inputs
Below is a guide to help you understand and adjust each parameter in the GUI.

### 1. Codec
- **Description**: Specifies the codec used for encoding the output video.
- **Default**: `mp4v`
- **Options**: 
  - `mp4v` (MP4 format)
  - `XVID` (MKV format)
  - Others supported by OpenCV.

### 2. 3D Output Formats
- **Description**: Specifies the output format used for encoding the output video.
- **Options**: 
  - `Half-SBS / Full-SBS` (For 3D TVs, YouTube 3D)
  - `Half-OU / Full-OU` (For 3D projectors & VR headsets)
  - Others supported by OpenCV.

### 3. Aspect Ratio Selection
- **Description**: Specifies the Aspect Ratio used for encoding the output video.
- **Default**: `Default (16:9)`
- **Options**: 
  - `CinemaScope (2.39:1)` (True cinematic widescreen)
  - `21:9 UltraWide` (Compatible with UltraWide monitors)
  - `4:3 (Classic Films` (Retro-style)
  - `1:1 (Square)` (Social media content)
  - `2.35:1 & 2.76:1` (Advanced widescreen format

### 4. Divergence Shift 
- **Description**: Controls the amount of pixel shift for objects in the foreground.
- **Default**: `4.8`
- **Recommended Range**: `3.0` to `8.0`
- **Effect**: Higher values create a stronger 3D effect for objects closest to the viewer.

### 5. Depth Transition 
- **Description**: Controls the amount of pixel shift for midground objects.
- **Default**: `1.9`
- **Recommended Range**: `1.0` to `5.0`
- **Effect**: Fine-tune this value to balance the depth effect between divergence and convergence.

### 6. Convergence Shift
- **Description**: Controls the amount of pixel shift for Depth.
- **Default**: `-2.8`
- **Recommended Range**: `-5.0` to `0.0`
- **Effect**: Use Negative values to push objects farther back, creating more depth.

### 7. Sharpness Factor
- **Description**: Adjusts the sharpness of the final output video.
- **Default**: `0`
- **Range**: `-1.0` (softer) to `1.0` (sharper)
- **Effect**: Higher values make edges more distinct, but excessive sharpness can introduce artifacts like over brightness.

### 8. Blend Factor (Pulfrich Effect)
- **Description**: Controls the blending ratio between delayed and current frames for the Pulfrich effect.
- **Default**: `0.5`
- **Recommended Range**: `0.3` (subtle) to `0.7` (strong effect)
- **Effect**: Higher values enhance the Pulfrich effect but may make scene transitions harder on the eyes.

### 9. Delay Time (Pulfrich Effect)
- **Description**: Specifies the temporal delay (in seconds) to create the Pulfrich effect.
- **Default**: `1/30`
- **Recommended Range**: `1/50` to `1/20`
- **Effect**: Lower values (e.g., `1/50`) reduce the delay, creating a more subtle effect.

## 📁 Depth Map File Requirements
### 1. Just Have a Depth map Generated I suggest looking at
- **Depth Anything V2
- **Midas Models
- **DPT Models

## Processing Times
- **Estimated Times**:
  - A 30-second clip: ~15-30 minutes.
  - Full-length videos: ~5-24 hours+.
plus 3D render time	

---

## Example Workflow
1. Select your input video (`.mp4`, `.avi`, `.mkv`) and output file path.
2. Select your Depth map Video, make sure both video files are same width and  height and FPS 
3. Adjust rendering parameters for the desired 3D effect.
4. Click "Generate 3D Video" to process.

## Troubleshooting
- **Black Screens or Artifacts**:
  - Ensure the depth map matches the input video's resolution and frame rate.
  - Adjust `blend_factor` and `delay_time` for smoother transitions between scenes. this effect is supposed to 
- **Audio Issues**:
  - Verify that the source video contains an audio stream, if not video will be generated with no audio

## Notes
- Ensure `ffmpeg` is installed and available in your system's PATH for audioprocessing. or put ffmpeg.exe in assets folder 
- Depth maps must match the input video dimensions and frame rate.

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
