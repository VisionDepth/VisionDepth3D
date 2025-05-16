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

### Code & Pipeline Improvements

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

### Depth Perception & 3D Quality

- Improved **foreground subject stability** with adaptive DoF centered on dominant subject region.
- Parallax dynamically adapts based on depth complexity, reducing distortion in flat/wide shots.
- Minor fixes to sharpen/feather application order for better edge fidelity.

---

### Coming Soon in v3.2.x

- Language toggle support for Preview tab controls.
- Saliency/face-aware subject depth targeting.
- Smart DoF presets (e.g., "Cinematic", "Macro", "VR Lens").
- Heatmap overlays for depth variance & mask debugging.
- Auto-tuning of parallax ranges based on scene classification.

---

## Community Contributions

Have an idea, fix, or translation? [Open an Issue or PR](https://github.com/VisionDepth/VisionDepth3D/issues)
