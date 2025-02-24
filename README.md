# VisionDepth3Dv2 - 3D Video Conversion with AI Depth Mapping & Pulfrich Effect
VisionDepth3D AI-powered 3D converter is a high-performance tool for creating Half Side-by-Side (H-SBS) and Over-Under (OU) 3D videos with advanced depth-based rendering, Pulfrich effect simulation, and adaptive scene transitions. Leveraging AI-driven backward warping correction, it intelligently enhances depth perception by reducing halo artifacts and preserving object integrity.

With real-time depth adjustments, the system dynamically modifies foreground divergence, midground depth transition, and background convergence, ensuring a cinematic and immersive 3D experience. The integration of a deep-learning warp model refines depth shifts, improving background separation for more natural parallax effects. Runs on mid-range hardware.

🚀 Key Features:
📚 Multi-Format 3D Output:
= Side-by-Side (SBS)
- Half Over-Under (Half-OU)
- Full Over-Under (Full-OU)
- Interlaced 3D
  
🎛️ Advanced Depth Processing:
- Dynamic foreground, midground, and background shifting for accurate 3D depth perception
- Depth map normalization and bilateral filtering for enhanced accuracy
- Machine Learning Model (TensorFlow/Keras) for convergence correction via backward warping
  
🎞️ Video Processing & Effects:
- Real-time Pulfrich effect simulation
- Cinemascope cropping for cinematic 2.39:1 aspect ratio
- Automatic black bar removal and white edge correction
- Dynamic sharpness control for enhanced clarity
  
🕹 Interactive GUI with Tkinter:
- Live progress bar with FPS monitoring
- Pause, Resume, and Cancel processing on the fly
- Drag-and-drop file selection for input video, depth map, and output path
- Customizable settings with save/load functionality
  
⚡ Optimized Performance:
- Threaded rendering for non-blocking UI
- FPS smoothing using a moving average for stable performance feedback
- Adaptive scene change detection with dynamic depth adjustments
  
🌐 Extras:
- One-click GitHub repository access via UI
- Settings persistence in JSON format
- Supports multiple codecs: mp4v, H264, XVID, and DIVX

Designed for VR enthusiasts, stereoscopic filmmakers, and 3D content creators, VisionDepth3D provides fine-tuned control over depth parameters, delivering professional-quality 3D video conversions with precision and efficiency.

GUI Layout
--
![GUILayout](https://github.com/user-attachments/assets/c77bd88d-4dd1-40ae-8e4e-2fd89400bfe4)

## Guide Sheet: Install

Installation Steps
This program runs on python 3.9 - 3.10
pip required to install dependancies

### Step 1: Download the VisionDepth3Dv2 Program
- Download the VisionDepth3D zip file from the official download source. (green button)
- Extract the zip file to your desired folder (e.g., c:\user\VisionDepth3D).
- Download Backwards warp model [Here](https://drive.google.com/file/d/1x2JApPfOcUA9EGLGEZK-Bzgur7KkGrTR/view?usp=sharing) and put in weights folder

### Step 2: Install Required Dependencies
- press (Win + R), type cmd, and hit Enter.
- 
- Navigate to the Program Directory:
```
cd C:\user\VisionDepth3D
```
- Install Dependencies:
```
pip install -r requirements.txt
```

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
