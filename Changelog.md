# VisionDepth3D v3.3 – Changelog

*This update is a major overhaul to both the **Depth Estimation Pipeline** and the **3D Rendering Pipeline**, focusing on stability, accuracy, and artifact reduction.*

---

## **Depth Pipeline Updates**

### **1. Black Bar Cropping for Depth Estimation**
- New `ignore_letterbox_bars` option detects letterbox bars in the first non-empty frame.
- Crops top/bottom bars before sending frames to the depth model.
- Re-applies bars after processing with **neutral depth values**, preventing black regions from appearing closer or farther than the main scene.

### **2. Output Resolution Preservation**
- Depth maps are resized back to the original cropped resolution before re-adding bars.
- Ensures final depth video matches **exact original width and height**.

### **3. Safety Checks**
- If bars exceed frame height or the frame is empty (e.g., fade-in/black intro), bars reset to zero to prevent OpenCV assertion errors.

### **4. Unified Depth-to-Grayscale Conversion**
- `convert_depth_to_grayscale()` now handles:
  - `PIL.Image`, `torch.Tensor`, and `numpy.ndarray`.
  - NaN values and shape inconsistencies.
- Centralized function ensures consistent grayscale conversion across all output modes.

### **5. Sidecar Metadata for Bars**
- Saves `.letterbox.json` with `top`, `bottom`, and `original_resolution` next to the depth video for future reference.

---

## **3D Pipeline Updates**

### **Stability & Robustness**
- Entire render loop wrapped in `try/except/finally` for guaranteed cleanup.
- Defensive initialization for `ffmpeg_proc` and `out`.
- Early exit if OpenCV `VideoWriter` fails to open.
- Pause handling correctly maintains frame index and updates ETA/FPS while paused.
- Cancel paths checked both during active processing and while paused.
- Automatic codec fallback if FFmpeg encoder is invalid.

### **Depth Map Processing (Smoother, More Stable Depth)**
- **TemporalDepthFilter** (EMA smoothing) reduces depth flicker frame-to-frame.
- Percentile-based depth normalization (EMA of low/high quantiles) for consistent depth range across shots.
- **Midtone shaping** (gamma curve) improves depth layering.
- Optional **curvature enhancement** to add roundness to objects.

### **Stereo / Parallax Control**
- **ShiftSmoother** damps rapid disparity changes for foreground, midground, and background.
- **Edge-aware masking + feathering** reduces tearing and ghosting at depth edges.
- **Dynamic IPD scaling** adapts stereo strength based on scene depth variance.
- **Subject-tracked zero parallax** with **floating window easing** prevents abrupt window size changes.
- Optional **dynamic convergence bias** tied to subject depth.
- **IPD factor knob** for global stereo strength adjustment.

### **Image Quality Enhancements**
- GPU-based **depth-of-field blur** with multi-level Gaussian blending.
- **Brightness-preserving sharpening** with highlight protection.

### **Framing, Aspect Ratio & Output Formats**
- Auto letterbox detection + cropping before depth processing (optional per frame).
- Aspect-ratio safe resizing with `pad_to_aspect_ratio` for perfect per-eye alignment.
- Two processing modes:
  - **Preserve Original Aspect Ratio** exactly.
  - **Target Output Aspect** for cinema/VR formats.
- Multiple stereo formats supported:
  - **Full-SBS**, **Half-SBS**, **VR 1440×1600**, **Dubois anaglyph**, **passive interlaced**.

### **Encoding & I/O**
- FFmpeg over stdin with:
  - CRF for `libx*` codecs.
  - CQ for NVENC with `-b:v 0` for constant quality.
- Codec mapping for CPU/GPU encoders, OpenCV fallback if FFmpeg is unavailable.

### **UX / Telemetry**
- Smooth, real-time **progress/FPS/ETA updates** — also works while paused.
- More descriptive logging (skipped blank frames, crop decisions, encoder issues).

---

This version is a **big step toward rock-solid 3D output** — cleaner depth maps, more comfortable parallax, and fewer artifacts.

