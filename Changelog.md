# VisionDepth3D v3.4 – Changelog (Session Updates)

## 1) 3D Rendering Pipeline – New Depth & Subject Controls
**These parameters are now fully integrated into the main 3D render pipeline and the live preview.**  
They affect the actual disparity math during final rendering, not just the preview.

### New Parameters:
- `depth_pop_gamma` *(0.70–1.20, default 0.85)* – Gamma on normalized depth to shape mid-depth contrast.
- `depth_pop_mid` *(0.00–1.00, default 0.50)* – Midpoint around which the gamma curve pivots (controls “where the pop happens”).
- `depth_stretch_lo` *(0.00–1.00, default 0.05)* – Lower percentile clamp for depth normalization (expands shallow depth range).
- `depth_stretch_hi` *(0.00–1.00, default 0.95)* – Upper percentile clamp for depth normalization (compresses deep range).
- `fg_pop_multiplier` *(1.00–1.60, default 1.20)* – Extra disparity boost for **foreground** layers.
- `bg_push_multiplier` *(1.00–1.40, default 1.10)* – Extra disparity boost for **background** layers.
- `subject_lock_strength` *(0.00–2.00, default 1.00)* – Bias that keeps the tracked subject closer to zero-parallax plane.

### Benefits:
- Greater control over **where and how depth pops** in the scene.
- Ability to fine-tune **foreground pop** vs **background push** for comfort and drama.
- Subject lock makes **character-focused shots** more comfortable and cinematic.

---

## 2) Live 3D Preview Window
- Same new controls as the render pipeline, updated **in real time** when sliders are moved.
- Debounced updates (150 ms) to keep preview smooth while dragging.
- Tooltips for all controls, including the new parameters.

---

## 3) Main 3D Generator UI (VD3D tab)
- Added **“Pop & Subject Controls”** panel for the new parameters.
- Dark-themed sliders and numeric entry boxes.
- Tooltips added for all new and existing controls.
- Settings persistence in `settings.json`.
- Presets system updated to save/load these new values.
- `reset_settings()` updated with defaults for the new parameters.

---

## 4) Dark Theme & Layout Overhaul
- Full ttk dark theme (`clam` base + custom styles) applied to:
  - Notebook tabs, frames, scrollbars, and progressbars.
- New **ScrollableFrame** with dark canvas/background and hover-activated wheel scrolling.
- Replaced default menu bar with **custom dark header bar** (Language, File, Help menus).
- Dark-themed `OptionMenu` replacements for consistent styling.

---

## 5) Frametool – Upscaling Engine Update
**Frame-by-frame upscaling is now faster, more stable, and better suited for large videos.**

### Core Improvements:
- **Tiled Upscaling Support** – Added `tile` and `tile_pad` options to prevent VRAM overflows when processing high-resolution frames.
- **Target Size Output** – Can now upscale directly to the desired resolution without an extra resize step.
- **Adaptive Interpolation** –  
  - Uses `cv2.INTER_AREA` when downscaling (sharper details).  
  - Uses `cv2.INTER_CUBIC` when upscaling (smoother transitions).
- **ONNX Runtime Optimizations** – Reduced CPU overhead with tuned threading (`intra_op_num_threads`, `inter_op_num_threads`) and full graph optimization (`ORT_ENABLE_ALL`).
- **Faster Disk-to-GPU Pipeline** – Prefetches frames in a background loader thread with an 8-frame queue to overlap I/O and processing.
- **Real-Time Progress Updates** – Progress bar and status updates are now thread-safe and smooth during processing.
- **Improved Encoder Presets** –  
  - NVENC: `h264_nvenc -preset p4 -tune hq -cq 19` with lookahead & B-frames.  
  - libx264: `-preset medium -crf 18`.
- **Safe FFmpeg Shutdown** – Prevents hanging processes when cancelling a job mid-run.


---

### Quick Reference: New Controls
| Parameter | Purpose |
|-----------|---------|
| **Depth Pop Gamma** | Shapes depth midtones; lower = more mid-depth pop, higher = smoother compression. |
| **Pop Mid** | Adjusts where the gamma curve pivots, shifting emphasis along the depth range. |
| **Stretch Lo / Hi** | Clamps and rescales the depth range to increase usable depth contrast. |
| **FG Pop ×** | Multiplies disparity for foreground layers for more pop-out. |
| **BG Push ×** | Multiplies disparity for background layers for more push-back. |
| **Subject Lock** | Keeps the tracked subject near screen depth for comfort and reduced window violations. |
