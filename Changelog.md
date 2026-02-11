# VisionDepth3D v3.8.2 - Changelog 

---

> This release delivers major performance improvements to both live 3D preview and offline rendering, alongside new depth engines, stability fixes, and encoding reliability upgrades.

---

## 1) Depth Estimation Tab

### UI Depth Tab Labelling

- Renamed the Depth Estimation tab to **Depth Engine** to better reflect multi-backend depth processing.
- Reduced console warning spam related to sequential pipeline usage by suppressing the specific Hugging Face warning message.

### Depth Anything 3 (DA3) Adapter Integration

- Added native Depth Anything 3 backend support via a dedicated DA3 adapter (separate from Hugging Face pipeline models).
- Implemented DA3 model loading through Hugging Face `from_pretrained` with VD3D cache routing into the `weights/` directory.
- Added DA3 model entries to the model selector (DA3-SMALL / BASE / LARGE / GIANT and DA3METRIC variants).
- Wired DA3 inference into the unified depth pipeline so it works with both image and video depth workflows.
- Mapped the UI “Inference Resolution” dropdown into DA3’s `process_res` logic (single max-side target resolution), with a video-friendly cap applied to prevent excessive internal upscaling.
- Normalized DA3 depth outputs into a consistent 0–1 range to match existing VD3D depth handling and export logic.
- Depth polarity handling for DA3 metric models remains user-controlled via the “Invert Depth” toggle.
- Improved DA3 batching compatibility by supporting list-of-PIL inference and ensuring returned depth frame counts match input batch size (with a per-image fallback if needed).
- Added a DA3 warm-up pass during model load to reduce first-frame hitching and confirm the backend is initialized correctly.

### Video Depth Anything (VDA) Adapter Integration

- Added native **Video Depth Anything** backend support via a dedicated VDA adapter for sequence-based video depth inference.
- Implemented VDA model loading directly from Hugging Face repositories (e.g. `depth-anything/Video-Depth-Anything-*`) with automatic checkpoint download and caching.
- Integrated VDA into the unified depth pipeline so it can be selected and used alongside DA3, ONNX, and Hugging Face depth models.
- Enabled sequence-aware inference for video input, allowing VDA to process temporal frame batches instead of independent per-frame depth estimation.
- Added configurable target FPS handling for VDA to reduce inference load on high-FPS sources by running depth inference at a lower temporal rate.
- Ensured VDA output depth frames are normalized into VD3D’s standard 0–1 depth range for compatibility with existing export, blending, and 3D rendering logic.
- Wired VDA output into the same post-processing, temporal normalization, and letterbox-handling pipeline used by other depth engines.
- Added VDA model warm-up during load to verify backend initialization and reduce first-inference latency.
- Depth polarity for VDA models remains user-controlled via the existing “Invert Depth” toggle for consistency across all depth engines.

### ONNX Model Fixes & Stability Improvements

- Fixed Distill-Any-Depth ONNX models (Small / Base / Large) failing to run due to internal tensor shape mismatch.
- Distill-Any-Depth ONNX models now correctly use a fixed 518×518 inference size, matching their exported positional embedding grid.
- Added automatic detection for Distill-Any-Depth ONNX models and enforced fixed input resolution internally.
- Updated ONNX image preprocessing to preserve aspect ratio using padding instead of stretching, improving depth stability and quality on widescreen content.
- ONNX warm-up now succeeds reliably for Distill-Any-Depth models without broadcast or Add-node errors.
- Enabled safe ONNX Runtime graph optimizations to reduce unnecessary memory copies and warning spam.
- Added clearer ONNX model identification output in the console so users can see exactly which ONNX model is being loaded.

### Model List Consistency

- Fixed missing Distill-Any-Depth ONNX models in the depth inference script while still being listed in the UI.
- Ensured ONNX model availability in the UI now correctly matches backend support.

### Video Encoding / Codec Handling

- Fixed CPU and GPU FFmpeg codecs (libx264, libx265, NVENC, AMF, QSV) being incorrectly routed through OpenCV’s VideoWriter.
- Non-OpenCV-safe codecs are now encoded via FFmpeg piping, preventing OpenH264 DLL errors and codec initialization failures.
- OpenCV VideoWriter is now limited to compatible FourCC codecs (mp4v, XVID, DIVX) with automatic fallback handling.

### Depth Inference Performance & Pipeline Optimizations

- Reduced redundant image resizing during video depth inference to avoid double-scaling overhead.
- Consolidated resize to a single pass per frame, reducing CPU overhead.
- Enabled CUDA-optimized memory layout (`channels_last`) for Hugging Face depth models when running on GPU.
- Improved FP16 inference handling for supported Hugging Face models to increase throughput on CUDA devices.
- Optimized ONNX Runtime session configuration using safe graph optimizations and memory arena usage.
- Improved batch handling logic to reduce per-frame overhead during video processing.
- FFmpeg piping is now preferred by default for video output, significantly reducing encoding bottlenecks.

### Letterbox & Black Bar Handling (Video)

- Fixed letterbox (black bar) regions incorrectly contributing to depth inference.
- Depth estimation now consistently ignores top and bottom letterbox bars instead of assigning artificial depth.
- Improved letterbox detection with multi-frame fallback probing and stabilization to prevent flicker.
- Letterbox regions are now filled with a neutral depth value, preventing pop-out artifacts and white banding in 3D renders.

---

## 2) 3D Video Generator Tab

### 3D Rendering Pipeline Performance & Stability

- Implemented full render-state reset at the start of each video and image render to prevent temporal drift and accumulated smoothing artifacts between sessions.
- Reset internal pixel shift EMA buffers per render, ensuring clean disparity initialization and improved real-time stability.
- Reset floating window convergence trackers and easing states to eliminate carry-over offsets and unintended masking behavior across renders.
- Reinitialized depth percentile normalization per render, allowing depth range calibration to adapt cleanly to each clip for more consistent parallax response.
- Improved convergence and floating window behavior during the first frames of each render, eliminating “settling” artifacts and jitter.
- Resulted in significantly smoother live 3D playback and notable FPS improvements during real-time rendering.

### Output Geometry & Eye Mode Fixes

- Fixed output sizing logic for VR, Passive Interlaced, and single-eye export modes.
- Ensured per-eye resolution handling remains consistent across all 3D formats.
- Corrected floating window width calculations to always operate on per-eye dimensions instead of SBS frame width.
- Added safety resizing to guarantee encoded frames always match target output resolution.

### Preview GUI

- Preview GUI now supports an optional Convergence Crosshairs overlay for faster convergence tuning.

### UI Label Consistency

- Fixed mismatched labels for Foreground Shift and Background Shift.
- Sliders now correctly match their tooltips.

### Encoding Settings Layout

- Reworked the Encoding Settings dialog layout for improved spacing and readability.
- Grouped checkboxes, dropdowns, and quality controls into clearer rows.

### Processing Options

- Moved Clip Range (start/end time) controls into the Processing Options dialog.
- Clip range settings respect the selected UI language and include translated labels and tooltips.

### Menu Fixes, Presets, and Updater Integration

- Help → Check Updates now launches the bundled **VisionDepth3D Updater** window (`VisionDepth3D_Updater.exe`) to download and install the latest official Windows release.
- Added a confirmation prompt before launching the updater, since VisionDepth3D closes itself to allow safe updating.
- Fixed **File → Load Preset** failing from the dropdown due to the preset apply function not being available in scope.
- Fixed **File → Output Path** dropdown not opening the save dialog while the hotkey worked, by routing the menu action through the same handler used by `Ctrl+O`.
- Removed **Save Settings** and **Load Settings** from the File menu since preset save/load already covers the same workflow and simplifies the UI.

## 3) VD3D Live 3D (Real-Time Depth + SBS Pipeline)

### Live Depth Inference Performance Overhaul

- Implemented persistent GPU tensor staging for live frame uploads, eliminating per-frame CUDA allocations and significantly reducing memory transfer overhead.
- Optimized live depth input preprocessing to reuse GPU buffers instead of recreating tensors each inference cycle.
- Reduced redundant CPU to GPU conversions during live depth updates.
- Improved FP16 autocast handling for Depth Anything V2 live inference to ensure stable mixed-precision execution on CUDA.

### Real-Time Pixel Shift Pipeline Optimization

- Added persistent CUDA frame buffers for the live pixel-shift SBS renderer to avoid per-frame GPU reallocations.
- Reduced per-frame normalization overhead by using in-place GPU operations.
- Improved handling of mixed return types from `pixel_shift_cuda` (CUDA tensors or NumPy fallback), ensuring stable live output without crashes.
- Prevented pipeline stalls caused by repeated tensor construction and shape revalidation.

### Live Depth Update Scheduling & Stability

- Implemented controlled depth refresh rate (Depth FPS) to decouple depth inference from preview frame rate for smoother live playback.
- Improved EMA depth smoothing behavior for live mode to reduce temporal jitter while preserving responsiveness.
- Reduced live preview hitching caused by first-frame warm-up and inference spikes.

### Live Capture & Preview Improvements

- Reduced capture overhead by allowing lower capture FPS without affecting SBS rendering smoothness.
- Improved screen capture pacing using high-precision timers to prevent uneven frame delivery.
- Improved live preview stability when mixing screen capture and GPU depth inference.

### Overall Live Mode Gains

- Live 3D preview performance increased by approximately 40 to 70 percent depending on GPU and inference resolution.
- Significantly reduced stutter caused by GPU memory churn.
- More consistent frame pacing for real-time SBS output.

---

> **Upgrade Note**  
> Back up your `weights/` and `presets/` folders before uninstalling v3.8.1  
> Then run **VisionDepth3D_Setup_Downloader** to download the official  
> VisionDepth3D v3.8.2 Windows installer and required `.bin` files.
