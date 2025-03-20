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

### To use this software users must have the following requirements
- Conda prompt ([Anaconda](https://www.anaconda.com/download/success))(Miniconda Recommended)
- openCV built with CUDA
- Pytorch-cu+12.6 ([PyTorch link](https://pytorch.org/get-started/locally/))
- Python 3.12
- CUDA 12.8 (Tested & Recommended)
- cuDNN 9.8.0
- VisionDepth3D-Package

## -- installation --
Remove any old visiondepth you may have already downloaded
```
conda env list
```
remove with
```
conda remove -n anyoldVD3D --all #change -n to one you want to remove
```

**Build openCV+CUDA**
1. First thing to do is download the VisionDepth3D-Package and unzip the folder to a directory you prefer. (C:/users/VisionDepth3D-Package etc.)
2. Next thing to do is create an environment
```
conda create -n VisionDepth3D python=3.12 
conda activate VisionDepth3D
cd VisionDepth3D-Package
```
3. After that download the latest versions of openCV Source codes
[openCV](https://github.com/opencv/opencv) and [openCV_contrib](https://github.com/opencv/opencv_contrib) you can either download the zip file they provide ![image](https://github.com/user-attachments/assets/285b5b5b-7c12-4ac0-8edb-f6fccb437d87)
![image](https://github.com/user-attachments/assets/577123ef-21ef-4641-91c1-68f55d3f9782)


or clone their repo into VisionDepth3D-Package file folder ("cd path/to/VisionDepth3D-Package" first if not in environment):
```
git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
```

4. Download [CMake](https://cmake.org/download/)  from the Official Website and open GUI Application, once in the gui set "where is source code" to 
![image](https://github.com/user-attachments/assets/4021ce84-1b5b-4cfc-9fb6-e0f1a6e77781)
![image](https://github.com/user-attachments/assets/0a97e437-b4d1-461e-892e-9e9170186871)


5. After setting the source code set "where to build the binaries" to the cmake folder inside Vision Depth3D-Package 
![image](https://github.com/user-attachments/assets/8cc1bc62-b8c3-4b1d-8a87-03acf84923e3)
![image](https://github.com/user-attachments/assets/131e0445-0ddc-400e-866f-7449db68d76f)

6. Next click configure 
![image](https://github.com/user-attachments/assets/a1478ab7-8adf-4cb5-b2a1-f19028e61ef7)
and a window will pop up asking you to specify the generator for this project, I use Visual Studio 16 2019 Havent Tested on vs17, Choose a platform for the generator, i chose x64 because that is what I am running, hit finish and the prompt will configure the files
![image](https://github.com/user-attachments/assets/95dc8c8a-0e5d-4985-904c-1afa41b7dbb7)

7. Once complete you should have a window of generated files
![image](https://github.com/user-attachments/assets/4b25ea13-3c90-4219-8823-f1abbf5a33ee)

8. toggle these one by one. 
- WITH_CUDA ✅
![image](https://github.com/user-attachments/assets/6ca661bf-dad8-4f6c-98d1-13357d094b40)
- BUILD_opencv_world ✅
![image](https://github.com/user-attachments/assets/a7f201e4-06b9-472c-a213-a9a03793cecb)
- ENABLE_FAST_MATH ✅
![image](https://github.com/user-attachments/assets/e5f14535-0e7f-4c97-ac1a-43557ec2ad53)
- OPENCV_EXTRA_MODULES_PATH to 📂 C:\Users\VisionDepth3D-Package\opencv_contrib-4.x\modules
![image](https://github.com/user-attachments/assets/7a6d0b9a-9a64-4436-979d-73a09c0c9036)
![image](https://github.com/user-attachments/assets/ffc280a0-af44-43cb-bb34-3a1701d2ace6)
![image](https://github.com/user-attachments/assets/06d6f473-4792-4b42-88c3-7b8b9fc77620)

9. toggle these three off from test we don't need em
![image](https://github.com/user-attachments/assets/16196b81-a450-41ce-af48-e170bf375cac)

10. Search CUDA And toggle OPENCV_DNN_CUDA
![image](https://github.com/user-attachments/assets/38cfb8b1-95ea-425a-a9f2-cd143cd67a64)

11.  ### ⚠️Important⚠️
Make sure you specify the paths to VisionDepth3D environment python we created
![image](https://github.com/user-attachments/assets/f9212ed5-c569-466f-bdaf-6c25e41df54a)

when you created the conda environment you installed python=3.12 as well, when setting these inputs go to the miniconda3 or anaconda folder to find your environment folder , these are snaps of my miniconda3 env  folder for example if you see VD3D in the snaps this is just my original VisionDepth3D environment, ignore and just make sure you are in the created VisionDepth3D environment, the first snap here is just how the env folder looks like, 
![image](https://github.com/user-attachments/assets/791afea9-52a8-44f7-bc5b-fcd81fd1476b)

- PYTHON3_EXECUTABLE: when you created the environment you installed python 3.12 you can find the .exe in the VisionDepth env folder from conda should be in an file folder like this 
![image](https://github.com/user-attachments/assets/c8646a46-a9b0-4b89-85a0-8ec753aa6180)

-  PYTHON3_INCLUDE_DIR: set to main Miniconda or Anaconda include folder 
![image](https://github.com/user-attachments/assets/46d9d01c-0183-42e2-850a-944bd9a3461c)

- PYTHON3_LIBRARY: you can set this to the python312.lib in your Miniconda or Anaconda folder 
![image](https://github.com/user-attachments/assets/031837a4-5ca7-497d-bb64-f505be8e707c)

- PYTHON3_NUMPY_INCLUDE_DIRS: you can set it to numpy installed in environement miniconda3/envs/VisionDepth3D/Lib/site-packages/numpy/_core/include if numpy folder is missing "pip install numpy"
![image](https://github.com/user-attachments/assets/c862b1c9-5005-478d-82cb-41a09133eedb)

- PYTHON3_PACKAGES_PATH: set this to your environments site-packages folder inside Lib folder
![image](https://github.com/user-attachments/assets/e81a6194-a555-4717-844a-ba7be9499384)

12. change CMAKE_INSTALL_PREFIX to the build folder we created 
![image](https://github.com/user-attachments/assets/4a21f40b-f0ab-4e9b-bba2-9ee556f76180)

Hit configure again to generate more options and update files, An error will pop up but that is ok we still have more to toggle, 

13. Set CUDA_FAST_MATH 
![image](https://github.com/user-attachments/assets/461a5656-ce0e-45d9-b012-91f9994a2a7a)

14. Enter your CUDA ARCH BIN and PTX for your required system 
![image](https://github.com/user-attachments/assets/d113d5c8-e2ac-468a-96d8-6948c10d9f57)

15. Click Configure Again and it it should configure correctly, if not check python paths 
16. Once you get a configure complete click Generate, then click open project to open Visual Studios generator
17. in Visual Studios in the top bar where it says debug, change that to Release
![image](https://github.com/user-attachments/assets/9999a63e-1f1c-4565-9ff4-008ab599b620)
18. Next up is to build solution (Ctrl+Shift+B), and script will start Building openCV together with the files we generated in cmake, this may take 1-3 hours depending on system
![image](https://github.com/user-attachments/assets/fd68cb93-9971-4a6b-950a-7f07ba33dd3c)
19. once that is finished in the solution explorer right click on INSTALL name and click build, this will install openCV+gpu to environment since we set our python paths 
![image](https://github.com/user-attachments/assets/731c7f37-4871-4c33-a95f-3de4f5414dbe)

20. Next Check if opencv is installed in activated environment
```
opencv_version.exe
```
![image](https://github.com/user-attachments/assets/fa08e246-a640-4838-a2c4-002a61956854)

If you Get This you have successfully built openCV with CUDA Support

## Install VisionDepth3D
While in VisionDepth3D Environment and VisionDepth3D-Package directory, install requirements  
```
pip install -r requirements.txt
```
after it finishes install the latest pytorch+cu126 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
After that is finished Launch VisionDepth3D
```
python VisionDepth3D.py
```

Congratulations! you have successfully built opencv with CUDA support and installed VisionDepth3D 
Enjoy!

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


