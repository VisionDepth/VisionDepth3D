## VisionDepth3D v4.2.1 Changelog

### Main Focus

This update focuses on improving depth backend routing, cleaning up CUDA/DirectML handling in depth adapters, adding initial ViGeo depth-model integration, reducing avoidable preprocessing overhead, improving model-load reliability, and fixing several video-depth processing edge cases.

---

### Depth Anything V2

- Added automatic redirect from Hugging Face DA-V2 models to the optimized safetensors DA-V2 adapter.
- Added fast DA-V2 safetensors menu entries for Small/Base/Large FP16 and FP32 variants.
- Added DA-V2 model caching to avoid accidental repeated reloads.
- Added CUDA FP16/TF32 optimization support for DA-V2.
- Added DirectML-aware DA-V2 loading with FP32 fallback for compatibility.
- Moved more DA-V2 preprocessing to GPU where possible.
- Added NumPy RGB frame input support to reduce unnecessary `NumPy to PIL to NumPy` conversions.
- Fixed non-writable NumPy warning during DA-V2 preprocessing.
- Added clearer DA-V2 logging for selected weight, backend, dtype, and FP16 state.
- Added optional environment toggle to disable DA-V2 fast redirect:

```bash
VD3D_DISABLE_DAV2_FAST_REDIRECT=1
```

---

### DirectML / Backend Handling

- Added broader DirectML routing support across the main depth pipeline.
- Added optional DirectML force mode:

```bash
VD3D_FORCE_DIRECTML=1
```

- Added shared backend helpers for CUDA, DirectML, MPS, and CPU detection.
- Kept FP16 CUDA-only for stability.
- Added DirectML-safe cleanup and conservative DirectML batch sizing.
- Updated adapter loading to pass `device` and `use_directml` when supported.
- Kept CUDA-only optimizations isolated from DirectML paths.
- Added safer backend-specific Torch cleanup for CUDA and DirectML.

---

### DA3 and Video Depth Anything

- Added DirectML-aware device handling to DA3 and VDA adapters.
- Added backend/device metadata reporting for DA3 and VDA.
- Kept DA3/VDA FP16 CUDA-only.
- Forced DirectML VDA path to FP32 for compatibility.
- Improved adapter error messages when DirectML unsupported ops are encountered.
- Improved VDA streaming behavior so long videos only keep the active sliding window and pending overlap predictions in memory.
- Added VDA overlap blending to reduce visible snapping between temporal windows.
- Added VDA ONNX fixed temporal-length handling, including automatic batch-size correction for fixed-`T` ONNX exports.

---


### ViGeo Integration

- Added initial experimental ViGeo depth model integration.
- Added a new ViGeo adapter:

```text
core/adapters/vigeo_adapter.py
```

- Added `vigeo:` checkpoint routing in `ensure_model_downloaded()`.
- Added ViGeo to the depth model list.
- Added ViGeo adapter loading through the same VD3D adapter system used by DA3 and Video Depth Anything.
- Added support for ViGeo sequence inference through VD3D's existing video-depth render pipeline.
- ViGeo now returns `predicted_depth` frames back into the main VD3D pipeline instead of writing output directly inside the adapter.
- Kept video writing, normalization, progress, batching, cancel handling, and output encoding inside the main VD3D depth renderer.
- Added ViGeo as a non-tiled video model path so it is not processed through the single-image tile system.
- Added ViGeo support to the depth pipe routing logic so `inference_size` and adapter kwargs are passed correctly.
- Added CUDA device handling for ViGeo adapter loading.
- Added FP16 autocast support for ViGeo on CUDA.
- Kept FP16 disabled for non-CUDA backends for stability.
- Added backend metadata reporting for ViGeo, including device, backend type, and DirectML status.
- Skipped ViGeo warm-up by default to avoid unnecessary startup VRAM spikes.
- Added fallback import handling for local ViGeo development installs.
- Verified ViGeo can be loaded inside the VD3D build environment after installing the ViGeo package into the active VD3D environment.

---

### ViGeo Depth Processing Behavior

- Added aspect-preserving inference sizing for ViGeo.
- Square presets such as `518x518` are treated as a max-side target for ViGeo unless exact sizing is forced.
- This prevents widescreen videos from being squeezed into a square before inference.
- ViGeo depth outputs are resized back to the original source frame resolution before VD3D writes the final depth video.
- Added raw ViGeo depth direction correction in the adapter.
- ViGeo raw output tested as black-near / white-far, so the adapter flips the raw depth ordering before VD3D normalization.
- VD3D's main depth normalization now stabilizes ViGeo output, reducing background depth flicker compared with raw standalone ViGeo testing.
- Confirmed ViGeo output looks more stable inside VD3D than the standalone repo test because it benefits from VD3D's normalization and video-depth pipeline.

---

### Hugging Face / ONNX

- Improved generic Hugging Face depth path logging.
- Reduced unnecessary duplicate resizing in the HF depth wrapper.
- Added DirectML-aware ONNX provider selection using `DmlExecutionProvider` when available.
- ONNX preprocessing now accepts RGB NumPy frames as well as PIL images.
- Added model-selection validation to prevent placeholder or invalid checkpoints from reaching the loader.
- Added safer checkpoint validation in `ensure_model_downloaded()`.
- Added a guard for missing Hugging Face processors so models are not incorrectly reported as loaded when their processor failed to initialize.
- Improved ONNX warm-up handling for fixed input sizes, fixed temporal dimensions, and VDA-style `/32` snapped dimensions.

---

### Video Depth Rendering Fixes

- Fixed a crash during depth rendering caused by `current_pipe` being referenced before it was initialized.
- Added a thread-safe active-pipeline snapshot at the start of video processing.
- Updated Marigold video detection to use the snapped active pipeline instead of the mutable global pipeline.
- Fixed non-VDA video processing so the final partial batch is flushed at end-of-file even when OpenCV reports an incorrect or unknown frame count.
- Improved video batch flushing reliability for variable-frame-count or problematic video files.
- Reduced risk of inconsistent output if the active model changes while a video job is running.
- Added safer VDA/ONNX fixed-batch handling using the active pipeline snapshot.
- Improved FFmpeg writer stderr handling by bounding stored diagnostic output to avoid unbounded memory growth on repeated encoder errors.
- Fixed Marigold frame re-encoding by explicitly setting FFmpeg `-start_number 1` for extracted frame sequences.
- Improved AV1 input warning text to clarify that the limitation is due to OpenCV decoding support.

---

### Image / Folder Processing Fixes

- Fixed image-folder processing after pipeline-lock changes by replacing accidental video-only UI helper calls with the correct image-folder UI helpers.
- Added natural sorting for image-folder processing so numbered image sequences process in expected order.
- Added safer active-pipeline checks for single-image and folder-image processing.
- Updated Marigold image handling to use the captured active pipeline instead of the mutable global pipeline.

---

### Thread Safety / UI Safety

- Added a shared `pipe_lock` and helper functions for active pipeline access:
  - `get_active_pipe_snapshot()`
  - `set_active_pipe()`
  - `clear_active_pipe()`
- Updated depth execution paths to use a thread-safe snapshot of the active pipeline and pipeline type.
- Made `PIPE_EXTRA_ARGS` updates lock-protected.
- Reduced direct global `pipe` / `pipe_type` usage in rendering paths.
- Stopped reading several Tk variables/widgets from background worker threads by snapshotting UI values before launching workers.
- Updated warm-up code to call local model callables directly instead of relying on the mutable global `pipe`.

---

### Performance / Cleanup

- Reduced CPU cost of `normalize_depth()` on very large depth maps by estimating percentiles from a downsampled sample.
- Removed several unused or heavyweight imports from the depth module.
- Continued reducing avoidable `PIL to NumPy` conversions in depth preprocessing paths.

---

### 3D Generator Controls and Preview Tuning

- Added new 3D Assistant control groundwork for subject screen-plane handling.
- Reworked the experimental subject placement control into **Subject Zero Lock** after testing showed that direct subject plane shifting could move too much of the scene and reduce background depth.
- Added **Subject Zero Lock** as a local subject disparity cancel control.
- Subject Zero Lock now measures the tracked subject's current shift and pulls that subject area closer to screen plane without flattening the full background.
- Improved the workflow for deeper scenes where the background needs to stay pushed back while the main subject remains comfortable.
- Added tuning support for close-up and dialogue-heavy shots where faces or central subjects need less stereo separation.
- Updated simple 3D style behavior so users can choose a style first, then adjust sliders without the style collapsing back into a weaker generic formula.
- Simple 3D controls now adjust from the selected style as a base instead of replacing the advanced preset values immediately.
- Added safer handling for Subject Zero Lock in saved state, keyframe settings, preview settings, and render settings.
- Updated preview debug output to report Subject Zero Lock clearly during live preview testing.

---

### Subject Lock and Screen Plane Behavior

- Improved subject anchoring so screen-depth adjustments can push the world deeper while keeping the tracked subject closer to screen plane.
- Added local subject disparity correction inside the CUDA pixel-shift path.
- Changed subject locking logic from a target-shift blend to a measured subject-shift cancel method.
- This helps preserve background depth while reducing uncomfortable subject separation.
- Added subject-mask expansion and soft silhouette filling so Subject Zero Lock affects more of the visible subject instead of only thin depth contours.
- Improved correction coverage for faces and inner subject details when the depth map separates the face, clothing, and outline into slightly different depth bands.
- Added softened correction masks to avoid hard stereo transitions around the subject.

---

### Encoding Preset UI Fixes

- Fixed an encoding preset dropdown bug where selecting a preset could briefly apply partial settings, trigger sync logic, and force the dropdown back to **Custom**.
- Added an internal encoding-preset application guard so child widget updates do not mark the preset as custom while the preset is still being applied.
- Blocked related checkbox, codec, CRF, and CQ signal updates during encoding preset application.
- Forced the encoding preset dropdown to remain on the selected preset after the preset finishes applying.
- Improved reliability when switching between multiple encoding presets in the Output and Encoding dialog.

---

### Stability and State Handling

- Added additional guard logic around multi-control preset application to avoid accidental UI state changes while presets are loading.
- Fixed a crash caused by leftover Subject Screen Plane variable references after converting the control to Subject Zero Lock.
- Fixed a startup syntax issue caused by a duplicated `subject_lock_strength` argument in the render function signature.
- Improved consistency between 3D Assistant sliders, advanced controls, preview generation, and final render settings.

