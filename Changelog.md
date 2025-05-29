# VisionDepth3D v3.2.4 – Changelog

*Note: Although the last official version was listed as 3.1.9, several intermediate patches were applied via GitHub and consolidated under version 3.2.4.*

---

## GUI Enhancements

### Inference Steps Control
- Introduced `inference_steps_entry` field to support user-defined inference steps for diffusion models.
- Includes input validation and fallback handling.
- Dynamically updates on `<Return>` and `<FocusOut>` events.

### Resolution Dropdown Improvements
- Expanded resolution options to include model-native sizes:
  - `512x256`, `704x384`, `960x540`, `1024x576`, and others for improved performance and visual quality.
- Automatically strips display hints like `" (DC-Fastest)"` for cleaner parsing of dimensions.

### CPU Offload Mode Selection
- Added support for multiple modes:
  - `"model"`, `"vae"`, `"unet"`, `"sequential"`, `"none"`
- The selected value is passed directly to the pipeline logic via `offload_mode_dropdown.get()`.

### Sidebar Layout
- Sidebar width increased from `22` to `30` for improved component spacing and usability.

---

## DepthCrafter Integration (Work-in-Progress)

### Pipeline Enhancements
- `load_depthcrafter_pipeline()` now supports the following arguments:
  - `inference_steps`
  - `offload_mode`
- Additional parameters are currently hardcoded and will be configurable in future updates.
- Device mapping is handled dynamically based on `offload_mode`:
  - `"sequential"` runs all operations on the GPU.
  - Other modes selectively offload components to CPU to manage VRAM usage.

---

## Stability Fixes and Improvements

- Warm-up logic now includes spinner feedback to prevent GUI freeze during model loading.
- All models, including local ones, now run reliably. However, local models still require manual configuration of inference size and batch size due to unresolved dynamic resolution handling.
- `invert_var` toggle is now functioning correctly.
- Subject depth smoothing introduced in the 3D pipeline to reduce temporal jitter in estimated depth maps.
- Focal depth consistency added for stereo rendering: subject depth is now shared across both eye views.
