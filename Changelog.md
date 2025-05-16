## VisionDepth3D v3.1.9 Changelog (Expanded)

### Fixes

- Fixed all **3D preview modes not rendering correctly** due to bypassed routing for `generate_preview_image()` in `HSBS`, `Red-Blue Anaglyph`, and others.
- Corrected **"Zero Parallax Strength" slider** which was stuck due to identical `from_`/`to` range.
- Fixed preview image disappearing issue in Tkinter by preserving `ImageTk.PhotoImage` reference with `preview_canvas.image = img_tk`.
- Fixed **`invert_var` not being passed** to folder-based depth inference.
- Fixed preview GUI not calling the correct **interlaced/3D preview** render function.
- Fixed **preview sliders overflowing layout** or being out of view on smaller screens.
- Fixed **language loading bug** in `en.json`.

---

### GUI Enhancements

- Added full **Preview GUI** for still images — now shows stereo results with adjustable preview types.
- Added **Preview dropdown options**: Shift Map Grayscale, Clipped ±5px, Left-Right Difference.
- Improved slider layout and resolution detection in Preview tab.
- GUI now **remembers selected preview type** with `save_settings()` / `load_settings()` support.
- Reworked layout to improve visibility and make room for new preview tools.
- Added **tooltips** for most new controls.
- Began integrating **language toggle support** into Preview tab.

---

### Code & Pipeline Improvements

- Unified routing logic for preview image generation — applies consistently across image, folder, and video inputs.
- Revised internal architecture to allow **depth map reuse** across tabs (future DOF & 3D tabs can reference results).
- Refactored `estimate_subject_depth()`:
  - Better subject targeting using median and histogram logic.
  - Robust fallback if depth region is too flat or empty.
- Rewrote `compute_dynamic_parallax_scale()` for smoother adaptive parallax based on scene depth variance.
- Safe guards added to prevent preview crashes from malformed or flat depth maps.
- Improved pixel shift + convergence offset logic to support both **static** and **subject-tracking** modes.
- Depth preview now handles **16-bit Marigold exports** by downscaling them safely to 8-bit thumbnails.

---

### New Model Support

- ✅ Added support for **Marigold Diffusion depth model** from Hugging Face.
- ✅ Integrated **Stable Diffusion-based depth pipelines** via `diffusers`.
- ✅ Added ONNX model fallback path for efficient processing (CUDA + TensorRT support).
- All models now **auto-warm** on dummy inputs for faster first inference.
- ONNX and Diffusion models now labeled internally to allow conditional 16-bit path routing.

---

### Depth Perception & 3D Quality

- Foreground subject detection improved using wider crops and histogram smoothing.
- DOF effect tuned to favor background blur while keeping subjects crisp.
- Pixel-shift + convergence logic now adapts to subject depth in real-time.
- Adaptive parallax scale uses normalized depth variance to minimize distortion in wide/flat scenes.

---

### Coming Soon in v3.2.x

- Language toggle support for Preview tab controls.
- Smart DOF presets: *Cinematic*, *Macro*, *Portrait*, *VR Lens*.
- Heatmap overlays for **depth confidence and mask debugging**.
- Auto-tuning of parallax scaling based on **scene complexity**.
- Full integration of **Saliency + Face-Aware subject targeting**.
- Streamlined input/output linking between tabs (generate once, reuse everywhere).
