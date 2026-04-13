# VisionDepth3D v3.9 - Changelog

---

> This release introduces major upgrades to VisionDepth3D’s model loading, depth workflow,
  VR180 output, and pipeline stability, with improved Hugging Face integration, ONNX support, 
  and overall reliability across the app.

---
## 3D Video Generator

### VR180 Equirect Output (Initial Release)

VisionDepth3D now supports native VR180 equirect stereo output.

#### New Output Modes

- Added **VR180 Equirect (Top-Bottom)**
- Added **VR180 Equirect (Side-by-Side)**

#### Dual-Resolution Workflow

Introduced a two-stage VR180 resolution system:

- **Flat Working Resolution (Per Eye)** – Internal render resolution used for stereo warping and depth-based effects.
- **Equirect Per-Eye Resolution (2:1)** – Final projected resolution used for VR headset playback.

This allows users to independently balance performance and final visual sharpness.

#### VR180 Controls

- Added adjustable **HFOV (60°–140°)**
- Added manual **Equirect WxH** configuration
- Added manual **Flat Working WxH** configuration
- Preset-ready structure for common VR180 resolutions

#### SDR & HDR Support

- VR180 output works in both **SDR** and **HDR10** rendering pipelines
- HDR10 VR180 maintains correct 10-bit encoding and color metadata through FFmpeg

#### Projection & Stereo Packing

- Proper hemispherical projection per eye
- Correct stereo packing for both TB and SBS modes
- Consistent output sizing across FFmpeg and OpenCV rendering paths

#### VR Player Naming Compatibility

- Updated auto-naming for VR180 outputs to align with VR players:
  - `_SBS_180`
  - `_TB_180`

Ensures correct automatic detection in players such as DeoVR.

---

## FPS/Upscale Enhancement (formerly FrameTools)

### Renamed & Reworked

- Renamed **FrameTools** to **FPS/Upscale Enhancement** for clearer purpose and usability

### UI Improvements

- Redesigned the FPS/Upscale Enhancement tab layout for a cleaner, more intuitive workflow
- Improved spacing, grouping, and visual hierarchy for faster setup and easier readability
- Introduced Pause, Resume, and Stop buttons

---

### Performance & Stability Improvements

- Fixed the threaded RIFE + ESRGAN processing pipeline that previously failed to run correctly
- Matched threaded pipeline behavior to the merged pipeline (native-resolution RIFE with final-stage upscaling)
- Removed unnecessary resizing and duplicate processing that caused slowdowns and memory spikes
- Improved frame queue handling for smoother throughput and better GPU utilization
- Fixed progress reporting and ETA calculation to prevent incorrect values and UI update spam
- Added safe, bounded frame loading with cancellation support to prevent deadlocks and excessive RAM usage
- Stopped swallowing queue exceptions by handling `Full` and `Empty` normally and surfacing unexpected errors for easier debugging
- Prevented the writer reordering buffer from growing indefinitely if a frame ID goes missing by adding a skip/limit safeguard
- Improved FFmpeg writer robustness by validating frame format/size before writing, reducing crashy edge cases

---

### Additional Pipeline Improvements (Post-Fix Enhancements)

- Implemented dynamic model loading system using Hugging Face repositories
- Removed requirement to ship large model weights with the application
- Models are now downloaded on demand and cached to the local `weights/` folder
- Ensured all runtime model loading resolves correctly regardless of install method
- Refactored model initialization to prevent duplicate loading and improve startup reliability
- Removed redundant frame normalization passes that caused unnecessary CPU overhead
- Simplified threaded pipeline frame flow to reduce queue overhead and unnecessary data copying

---

### Performance Insight

- Identified input resolution as a primary performance bottleneck during RIFE + ESRGAN processing
- Lowering input resolution significantly improves processing speed due to reduced pixel workload
- Input Resolution % setting now serves as a key control for balancing performance vs quality

---

### Speed Improvements

- Additional optimizations reduce CPU overhead and improve GPU utilization during processing

---

## Preview GUI

### Stability Fix

- Fixed issue where the Preview GUI window would not properly close in `.exe` builds
- Ensured proper window cleanup and destruction to prevent hanging UI processes

---

## Depth Engine

- Fixed packaging for DA3 in the `.exe` build
- Fixed `Process Video Folder` in the depth estimation pipeline
- Fixed UI freezing during depth folder processing and restored live progress bar updates
- Refactored depth folder and single-video processing to safely separate UI-side controls from background worker execution
- Fixed depth pipeline argument handling for single video and folder-based processing
- Added support for loading ONNX depth models from either local model folders or Hugging Face repositories
- Restored ONNX-specific warm-up and inference handling for Hugging Face-hosted ONNX models
- Hid diffusion-only controls such as `Inference Steps` and `CPU Offload Mode` unless a compatible diffusion model is selected
- Improved Hugging Face ONNX model detection for Distill-Any-Depth and Video Depth Anything exports
- Cleans up Letterbox.json file after rendering so no random temp file is left behind

## Special Thanks

A big thank you to **AcolyteOfHedone** for contributing fixes and technical improvements that helped strengthen this release, including AMD AMF encoder fixes, ONNX adjustments, and AMD GPU provider compatibility work.

GitHub: [EvolvingProficiency](https://github.com/EvolvingProficiency)

---

> **Upgrade Note**  
> Back up your `weights/` and `presets/` folders before uninstalling v3.8.2 
> Then run **VisionDepth3D_Setup_Downloader** to download the official  
> VisionDepth3D v3.9 Windows installer and required `.bin` files.
