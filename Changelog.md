# VisionDepth3D v3.6 – Changelog

---

## 1) Upscaling & Interpolation – Massive Speed Boost
* Rewrote frames tab pipeline with threaded workers + queues.
* RIFE interpolation, ESRGAN upscaling, and FFmpeg writing now run concurrently instead of sequentially.
* Intelligent frame indexing + buffering preserves exact order while maximizing throughput.
* Render time reduced from 10 hours → ~1 hour on long clips.
* Result: creators can now upscale & interpolate full-length videos in a fraction of the time without crashes or dropped frames.

---

## 2) Depth Pipeline – Refinements & Blending
* Integrated **Depth Blender prototype** as a tab in main UI.  
* Added early **sliders/knobs for tuning blend weight** between models.  
* Improved handling of **16-bit depth outputs** for richer disparity.  
* Experimented with **percentile clipping** to suppress outliers without flattening depth.  
* **Result:** cleaner separation between foreground and background, less “fuzz” in blended depth maps, and more consistent 3D parallax.
* Added **Depth Anything V2 Giant** model support 
* Integrated **FP16 half-precision toggle** in main UI for faster inference and reduced VRAM usage (on supported GPUs).

---

## 3) HDR10 Support – Preservation & Metadata
* Fixed washed-out output issue when re-encoding HDR10 videos.  
* Added flags to preserve:  
  * **10-bit pixel format** (`yuv420p10le`)  
  * **BT.2020 colorspace**  
  * **PQ transfer curve** (`smpte2084`)  
  * **Master-display / MaxCLL metadata**  
* Toggle option in UI: **Preserve HDR10 Metadata**.  
* **Result:** HDR sources retain their original punch and dynamic range instead of flattening to SDR.

---

## 4) Experimental Live 3D (WIP)

* Added **real-time 3D pipeline** for external inputs (console, capture cards, webcams).
* Integrated **Depth Anything v2 Small** for on-the-fly depth generation (larger models can be swapped if GPU allows).
* Implemented **VisionDepth3D method** for stereoscopic conversion of live depth maps.
* Early **end-to-end testing completed** with capture → depth → stereo output loop.
* **Performance tuning ongoing**: fps optimization, GPU acceleration, and reduced latency paths.
* Expanded support for **multiple input sources** (video devices, capture cards, webcams).
* **Result:** playable console/video feeds in live 3D, showing strong potential for real-time applications.

---

## 5) General Fixes & Stability

### Rendering
* Fixed: duplicate clip-window guard removed to prevent redundant early exits.  
* **New:** Clip range UI — users can now set precise Start/End times for partial renders.  
* **New:** Left-only and Right-only rendering modes now work directly (no post-split required).  
* Optimized: hoisted per-eye resize + aspect ratio calculations out of the frame loop.  
* Fixed: floating-window bar now scales correctly for single-eye renders.  
* Optimized: DOF/Color grading checks moved outside the render loop to cut overhead.  
* single-eye mode writes only the active eye, reducing memory work.  
* Additional **padding + edge reflection** during stereo warp to reduce bleed-through.  

### UI & Error Handling
* Patched white-edge artifact issue caused by 16-bit depth normalization.  
* Improved error handling in Preview GUI when models fail to load.  
* **Language files**: synced new controls (HDR toggle, depth blender knobs, etc.) across all packs.  
* All buttons and inputs use **dark theme styling** for a cleaner look.  

### Codec & Output
* Fixed: FFmpeg no longer forces `-preset slow` when a GPU codec is selected.  
  * NVENC now uses encoder-appropriate flags (`preset p5`, `rc vbr`, `cq`).  
  * CPU codecs continue to use CRF + preset for quality consistency.  

---

## 6) UI & Workflow Enhancements
* **Full** 3D Generation Tab UI overhaul for a cleaner look
* Save/Load settings and presets with hotkeys  
* Hotkey Import video & depth maps directly into the 3D workflow  
* Reset in one click  
* Jump straight to docs, bug reports, GitHub, or community links from the Help menu  
 
* **Result:** smoother day-to-day workflow and better visual testing inside VD3D.

---

**Summary:**  
v3.4 gave creators fine-grained depth & subject control.
v3.5 brings cinematic polish with stabilized DOF, a true audio tool with sync + codec options, a GPU color grading suite,
a stereo separation (IPD) adjustment for display comfort, and now ONNX pipeline + clip rendering support for faster experimentation.
v3.6 focuses on **depth blending refinements** and **proper HDR10 preservation**, 
while also introducing **precise clip-range rendering** and **direct Left/Right eye output** for greater workflow flexibility.  
This release tightens up edge handling, boosts stability, and continues to evolve the UI toward an all-in-one workflow.
