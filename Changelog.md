## VisionDepth3D v3.1.9 Changelog

### Fixes

- Fixed all **3D preview modes not rendering correctly** due to bypassed routing for `generate_preview_image()` in `HSBS`, `Red-Blue Anaglyph`, and others.
- Corrected **"Zero Parallax Strength" slider** which was stuck due to identical `from_`/`to` range.
- Fixed preview image disappearing issue in Tkinter by preserving `ImageTk.PhotoImage` reference with `preview_canvas.image = img_tk`.

---

### GUI Enhancements

- Replaced **Convergence Strength input** with a proper **slider (range -0.05 to 0.05)** for precision control.
- Added missing preview dropdown options including **Shift Map Grayscale**, **Clipped ±5px**, and **Left-Right Diff**.
- Improved slider layout and preview resolution handling for **Preview GUI**.
- Ensured **preview type memory persistence** via `save_settings()` / `load_settings()` for `"preview_type"`.

---

### 3D Code & Pipeline Improvements

- Unified routing logic for preview image generation, ensuring consistent handling across all modes (including tensors vs. frames).
- Refactored **`estimate_subject_depth()`**:
  - Added robust histogram + median blending.
  - Improved off-center handling using wider center crop.
  - More stable fallback behavior for empty/flat depth fields.
- Refactored **`compute_dynamic_parallax_scale()`**:
  - Now uses normalized depth variance for consistent scaling.
  - Smoother adaptive interpolation across varied scene depths.
- Added clamping and safety guards to prevent crashes from degenerate depth values.

---

### Pipeline / Model Loading Overhaul

- Refactored **ONNX loader** to detect:
  - Static input shapes (`(N, 3, H, W)` or `(1, 32, 3, H, W)`)
  - Required inference resolution (auto-patched at runtime)
- ONNX models now safely fall back to enforced shape and resolution during warm-up and runtime.
- Integrated **ONNX `inference_size` override** logic to prevent model crash on mismatched input dims.
- Added automatic **dummy warm-up batches**:
  - (1, 3, H, W) for spatial models
  - (1, 32, 3, H, W) for temporal models
- Refactored Hugging Face model loader to support:
  - All `AutoModelForDepthEstimation` variants
  - Batch-safe wrapper logic for inference compatibility
- Added **Diffusers model support**, including `Marigold v1.1`:
  - Supports 16-bit PNG export via `.export_depth_to_16bit_png()`
  - Smart inversion for high bit-depth paths

### Depth Processing Improvements

- Depth normalization and resizing is now dynamically adjusted to original image size.
- Added **colormap previews** using `matplotlib.cm` for fast switching between raw + enhanced depth views.
- Depth maps from ONNX/Diffusers/HF are converted to:
  - 8-bit grayscale for preview
  - 16-bit PNGs (Marigold only) for export
- Improved subject smoothing logic via dynamic histogram equalization (early DOF prep).
- Tuned parallax scaling to follow **depth variance curves** — avoids harsh pop-out artifacts.
- Temporarily removed **temporal smoothing** for clarity in high-motion scenes (will return as toggle later).

### Model Support Highlights

- ✅ **ONNX Models**:
  - Distill Any Depth Base / Large
  - Video Depth Anything (static size 518x518 enforced)
  - Fully accelerated with CUDA + TensorRT (fallback to CPU gracefully)
- ✅ **Hugging Face Transformers**:
  - Depth Anything V1/V2 (Large, Base, Small)
  - Intel MiDaS / ZoeDepth / DPT BEiT variants
- ✅ **Diffusion Models**:
  - Marigold v1.0 / v1.1 via `diffusers`
  - Supports native 16-bit export and runs on GPU-accelerated pipeline

### Backend Utilities

- Added `get_dynamic_batch_size()` helper for VRAM-aware inference scaling.
- Preview image size capped to `(480x270)` for smoother UI loading.
- AV1 codec check added using FFprobe — warns users to re-encode to H.264 if unsupported.
- Added support for **per-model enforced resolutions**, with auto-scaling and override warning.

# 🔄 Update Now!
**Link:** [https://github.com/VisionDepth/VisionDepth3D](https://github.com/VisionDepth/VisionDepth3D)

- Download new version or clone repo.
- Backup your `weights/` folder just in case.
- Overwrite the old VisionDepth3D folder.
- Enjoy a cleaner, more polished 3D rendering experience!

---
