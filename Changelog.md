# VisionDepth3D v3.8 – Changelog

---

## 1) Depth Estimation Tab

### Depth Models

- Fixed ONNX model loading:
  - Distill-Any-Depth (inference resolution 518×518, batch size 8)
  - Video Depth Anything (inference resolution 512×288, batch size 8)
- Implemented LBM depth model (development version). Thanks to Aether for the implementation fix.
- Removed depth models from the dropdown that returned no `d_type`.
- Fixed Hugging Face model downloads and caching so zoo models consistently save inside the app `weights/` directory (no more extra `.cache` downloads).
- Updated Transformers image processor loading to prefer `use_fast=True` when available (with automatic fallback when unsupported).

### Depth Backend

- Implemented temporal smoothing in the depth pipeline to reduce flicker and improve temporal stability of depth map output.
- Packaged VisionDepth3D.exe with Distill-Any-Depth (ONNX), Video Depth Anything (ONNX), and Depth Anything v2 Giant weights.

---

## 2) 3D Render Tab

### UI Fixes

- Added buttons for encoder settings and processing options.
- Implemented multi-language support and tooltips for new dialog boxes.
- Adjusted preview image window size and video info layout to prevent window overflow.
- 3D tab columns now stack correctly when resizing the window on smaller screens.

### 3D Backend

- Reworked Auto Crop Black Bars to use first-frame detection with cached crop reuse.
- Prevents per-frame crop jitter and depth/frame misalignment.
- Improves stability for cinema content with subtle letterboxing.
- Keep Audio checkbox now respects the user-selected output container instead of forcing MP4.

---

## Frametool Backend

- Reworked Frametool backend to support SSResNet models for feature model integration.

---

## Console Improvements

- Standardized startup console messages to clearly reflect which subsystems are initializing (Torch, depth estimation, upscaler, external 3D pipeline, language, settings).
- Unified compute device reporting across pipelines for consistent and clearer console output.
- Suppressed optional xFormers dependency warning on startup.
- Prevented duplicate language loading during settings restore.

---

## Summary

v3.8 focuses on stabilizing depth estimation, improving model compatibility, 
and refining the 3D Render tab UI with better layout behavior, clearer diagnostics, and improved localization support.

> Back up your `weights/` and `presets/` folders before uninstalling v3.7.
> Then run VisionDepth3D_Setup_Downloader to download the official
> VisionDepth3D v3.8 Windows installer and required `.bin` files.

> (Optional but recommended) Clear the Hugging Face cache to free space and
> avoid duplicate model downloads:
> `C:\Users\YOUR_USERNAME\.cache\huggingface`
