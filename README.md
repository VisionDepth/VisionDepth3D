# 3D HSBS Video Generator
This program generates Half Side-by-Side (H-SBS) 3D videos with advanced visual enhancements. It features the Pulfrich effect, black bar removal, and adaptive scene change handling. By processing input video alongside depth map data, it dynamically adjusts depth-based shifts and convergence for a more immersive 3D experience. The backward warping model corrects divergence shifts, and an inpainting process fills black regions left by warping to ensure seamless visuals. User-controlled parameters allow fine-tuning of VRAM usage, batch processing size, and depth effects for optimized performance and quality.

## Guide Sheet: Install
Installation Steps

### Step 1: Download the VisionDepth3D Program
- Download the VisionDepth3D.zipfile from the official download source.
- Extract the zip file to your desired folder (e.g., C:\VisionDepth3D).

### Step 2: Install Required Dependencies

If Using the Standalone Executable:
- No additional dependencies are needed. Skip to Step 3.
  
If Running from Source Code:

- Open a terminal or command prompt and enter:

- git clone https://github.com/VisionDepth/VisionDepth3D.git
- cd VisionDepth3D
- install_visiondepth3d.bat

- or alternatively you can use pip
- pip install git+https://github.com/VisionDepth/VisionDepth3D.git
- pip install -r requirements.txt

### Step 3: Download Backwards Warp Model
here you can download Backwards warp model and put it in ".\weights\" folder 
https://drive.google.com/file/d/1Ff0py6EpTG7IcLDQE9Brl9d3002Hd3JO/view?usp=sharing

### Guide Sheet: GUI Inputs
Below is a guide to help you understand and adjust each parameter in the GUI.

### 1. Codec
- **Description**: Specifies the codec used for encoding the output video.
- **Default**: `mp4v`
- **Options**: 
  - `mp4v` (MP4 format)
  - `XVID` (MKV format)
  - Others supported by OpenCV.


### 2. Foreground Shift
- **Description**: Controls the amount of pixel shift for objects in the foreground.
- **Default**: `4.8`
- **Recommended Range**: `3.0` to `8.0`
- **Effect**: Higher values create a stronger 3D effect for objects closest to the viewer.


### 3. Midground Shift
- **Description**: Controls the amount of pixel shift for midground objects.
- **Default**: `1.9`
- **Recommended Range**: `1.0` to `5.0`
- **Effect**: Fine-tune this value to balance the depth effect between foreground and background.

### 4. Background Shift
- **Description**: Controls the amount of pixel shift for background objects.
- **Default**: `-2.8`
- **Recommended Range**: `-5.0` to `0.0`
- **Effect**: Negative values pull background objects farther back, creating more depth.

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

### 8. Convergence Shift
- **Description: Adjusts the pixel shift to converge the left and right images for better depth perception.
- **Default: 0.0
- **Effect: Positive values will bring the images closer, enhancing depth perception.

### 9. Divergence Shift
- **Description: Adjusts the pixel shift to diverge the left and right images.
- **Default: 0.0
- **Effect: Positive values will push the images apart, creating a wider perspective.

## 10. VRAM Limit
- Description: Sets the maximum GPU memory usage to optimize performance and prevent out-of-memore errors. 
- Default: 0.0
- Effect: Controls batch processing size to balance speed and stability during rendering.

## 11. Batch Size
- Description: Specifies the number of frames processed in each batch.
- Default: 10
- Effect: Larger batch sizes may improve performance but require more VRAM.

### Backward Warping Model
- Description: A trained deep learning model that corrects divergence shifts in stereoscopic 3D rendering. It predicts and applies warp transformations to align left and right frames, ensuring a more natural depth perception.
- Default: Enabled
- Effect: Reduces visual artifacts caused by divergence shifts, improving overall 3D depth consistency. Uses inpainting to fill black regions after warping for a seamless result.

## Depth Map File Requirements
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
4. Click "Generate 3D SBS Video" to process.

## Troubleshooting
- **Black Screens or Artifacts**:
  - Ensure the depth map matches the input video's resolution and frame rate.
  - Adjust `blend_factor` and `delay_time` for smoother transitions between scenes. this effect is supposed to 
- **Audio Issues**:
  - Verify that the source video contains an audio stream, if not video will be generated with no audio
- **GPU Memory Errors**:
 - Reduce the batch_size or vram_limit to avoid exceeding GPU memory limits.
---

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
