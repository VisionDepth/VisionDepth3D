# 3D SBS Video Generator
This program generates Side-by-Side (SBS) 3D videos with enhanced visual effects such as the Pulfrich effect, black bar removal, and scene change handling. It processes input video and depth map data to create immersive 3D content, with options for user-controlled parameters.

--

## Guide Sheet: Install
Installation Steps

### Step 1: Download the VisionDepth3D Program
- Download the VisionDepth3D.zipfile from the official download source.
- Extract the zip file to your desired folder (e.g., C:\VisionDepth3D).

### Step 2: Install Required Dependencies

If Using the Standalone Executable:
- No additional dependencies are needed. Skip to Step 3.
  
If Running from Source Code:
- 

Open a terminal or command prompt.
Navigate to the program directory:

cd path\to\VisionDepth3D

Install required dependencies using pip:

pip install -r requirements.txt

## Guide Sheet: GUI Inputs
Below is a guide to help you understand and adjust each parameter in the GUI.

### 1. Codec
- **Description**: Specifies the codec used for encoding the output video.
- **Default**: `mp4v`
- **Options**: 
  - `mp4v` (MP4 format)
  - `XVID` (AVI format)
  - Others supported by OpenCV.


### 2. Foreground Shift (fg_shift)
- **Description**: Controls the amount of pixel shift for objects in the foreground.
- **Default**: `4.8`
- **Recommended Range**: `3.0` to `8.0`
- **Effect**: Higher values create a stronger 3D effect for objects closest to the viewer.


### 3. Midground Shift (mg_shift)
- **Description**: Controls the amount of pixel shift for midground objects.
- **Default**: `1.9`
- **Recommended Range**: `1.0` to `5.0`
- **Effect**: Fine-tune this value to balance the depth effect between foreground and background.

### 4. Background Shift (bg_shift)
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
- Description: Adjusts the pixel shift to diverge the left and right images. 
- Default: 0.0
- Effect: Helps manage GPU memory usage for smoother processing.

## 11. Batch Size
- Description: Specifies the number of frames processed in each batch.
- Default: 10
- Effect: Larger batch sizes may improve performance but require more VRAM.


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
- **GPU Memory Errors
 - Reduce the batch_size or vram_limit to avoid exceeding GPU memory limits.
---

## Notes
- Ensure `ffmpeg` is installed and available in your system's PATH for audioprocessing. or put ffmpeg.exe in assets folder 
- Depth maps must match the input video dimensions and frame rate.
