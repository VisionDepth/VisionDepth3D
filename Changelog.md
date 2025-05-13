## VisionDepth3D v3.1.8  Changelog

### Major Fixes

* Fixed bugged depth progress label that didn’t reset after 3D rendering.
* Implemented model warmup logic for ONNX-based depth estimators (prevents first-frame lag).

### New Features

* **Depth of Field** support added to full render pipeline.
* DoF blur controlled via GUI slider with tooltip and multi-language support.
* Save Preset dialog now allows custom filename entry.
* Graceful CLI notifications for all major operations.
* DOF controls added to Preview GUI for consistency.

### Refactors & Pipeline Logic

* **Convergence Offset** now fully integrates with subject-aware zero-parallax system.
* Removed legacy convergence shifting logic.
* Refactored `pixel_shift_cuda` – removed clamp from subject shift.
* Removed background shift multiplier logic — BG shift now reflects **true raw value**.

---

## Community Contributions

Have an idea, fix, or translation? [Open an Issue or PR](https://github.com/VisionDepth/VisionDepth3D/issues)
