# VisionDepth3D v3.7 – Changelog

---

## 1) Live 3D Capture – Real-Time Overhaul

### Audio Support Added

* Added optional live audio passthrough monitor for external capture devices  
* New flags:  
  * `--audio-device "<device name>"`  
  * `--audio-delay-ms <value>`  
* Supports DirectShow and fallback to WASAPI  
* FFplay-based audio pipeline integrated for low latency monitoring  
* Audio sync delay control for VR headset streaming  

### Color Channel Controls

* Added `--pixelshift-rgb` to properly route RGB output from CUDA warp kernel  
* Added `--force-bgr-swap` fallback for cameras that provide BGR swapped frames  
* Fixed purple and red tint issues on some capture devices  

### Performance and FPS Optimization

* Tuned live depth inference resolution defaults for smoother real-time processing  
* Recommended real-time profile: `infer-w 448`, `infer-h 256`, depth FPS ~ 8 to 12  
* Better GPU scheduling for live stereo warp when audio is active  
* More stable stream with reduced stutter on high resolution capture input (1080p HDMI sources)  

### Network Streaming Prep (Experimental)

* Added headless running mode (`--no-preview`) to remove local display overhead  
* Prepared architecture for browser-based SBS VR streaming  
* In development: WebRTC path for synchronized audio + video to Quest browser  

### General Stability

* Improved diagnostic output and runtime logs  
* Verified working flags for external capture cards (Elgato HD60 S+)  
* Ensured fallback modes for low resource conditions  

---

## 2) Initial Bugfixes & Issues Resolved (Live 3D Cleanup)

### UI Settings Not Applying

* GUI sliders and dropdowns weren’t being respected when starting live capture  
* Fixed runtime sync so all selected GUI settings apply properly  

### Manual CLI vs GUI Discrepancy

* Runtime ignored resolution, backend, FPS, and other settings when launched via GUI  
* Now fixed, all selected parameters propagate as expected  

### Video Tint Correction

* Purple/red tint caused by incorrect channel routing  
* Fixed with `--pixelshift-rgb` and `--force-bgr-swap` fallbacks  

### Capture Failures (Webcam/HDMI)

* Fixed `no frames arriving` error by enforcing `--fourcc MJPG` and proper MSMF backend handling  

### No Audio Output

* Live mode initially had no sound output  
* Added full audio routing support for monitoring devices (DirectShow/WASAPI)  

### FPS Performance Bottlenecks

* Live stereo + depth ran around 6.5–7 FPS at first  
* Now tuned with smaller inference size, better scheduling, and CUDA-side warp boost  

---

## 3) Floating-Window and Stability Fixes

### Dynamic Floating Window (DFW) Logic

* Fixed undefined `zero_parallax_offset` references and long term stabilization drift.
* Rebuilt the DFW into an asymmetric, side only window that masks a single edge (left or right) based on the dominant parallax direction.
* Added a minimum parallax threshold so the window stays completely off when depth is very close to the screen plane.
* Computes bar width as a blend of parallax magnitude and how far the tracked subject is from mid depth, then clamps it to a small fraction of the per eye width for a subtle mask.
* Uses `FloatingWindowTracker` to smooth the horizontal offset over time and reduce small frame to frame jitters.
* Uses `FloatingBarEaser` to ease the bar width in and out so it expands and collapses gradually instead of popping on and off.
* Supports both soft faded masks and solid black cinema bars through a single toggle, with faded mode as the default for VR and monitor playback.
* Result: a more invisible and cinema friendly floating window that reduces window violations, keeps edges clean, and stays out of the viewer’s way.

### Frame Jitter and Temporal Stability

* Fixed flickering, depth “breathing,” and jitter caused by unsmoothed convergence and subject tracking.
* Applied `SubjectDepthEMA`, `DepthPercentileEMA`, and `ConvergenceEMA` for improved temporal stability.
* Added a global `ShiftSmoother` to stabilize foreground, midground, and background parallax.
* Result: consistent parallax motion across frames, no more in-and-out depth pulsing, and a much cleaner, more comfortable stereo flow.

### Auto-Crop Black Bar Logic

* Fixed crop mis detections during fade ins and fade outs.
* Added a mean brightness guard so black bar detection does not update on very dark transition frames.
* Added per frame re evaluation when black bar height changes significantly.
* Result: letterboxed content (2.35:1 and similar) now auto crops reliably without vertical drift.

---
## 4) Unified Depth Pipeline Upgrade & Platform Stability Improvements

v3.7 introduces the largest Depth tab upgrade yet, combining:

- **GPU backend support across NVIDIA, AMD, and Apple Silicon**
- **FFmpeg codec selection for hardware-accelerated depth exports**
- **Pause / Resume / Cancel controls** for long video renders

---

### GPU Backend & Platform Support (CUDA / ROCm / MPS / CPU)

* Full rewrite of device detection — CUDA is no longer assumed  
* Automatic selection of best available compute backend  

#### Supported Depth Inference Backends
- **CUDA** — NVIDIA GPUs  
- **ROCm** — AMD GPUs  
- **MPS** — Apple Silicon GPUs  
- **CPU fallback** when no GPU is detected  

#### Benefits
* Prevents CPU-only fallback on capable GPUs  
* Removes CUDA-only links that caused AMD/macOS crashes  
* Core foundation for **Linux** deployment moving forward

> Result: VD3D depth inference now runs properly on **NVIDIA, AMD, Apple, and CPU-only** systems.

---

### FFmpeg Codec Selection for Depth Video Rendering

* New Video Codec dropdown added to the Depth Tab  
* Hardware and software encoding now user-selectable  

#### Supported Encoders

**NVIDIA NVENC**
- `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`

**AMD AMF**
- `h264_amf`, `hevc_amf`, `av1_amf`

**Intel QSV**
- `h264_qsv`, `hevc_qsv`, `vp9_qsv`, `av1_qsv`

**CPU Fallback**
- `libx264`, `libx265`, `libaom-av1`, `libsvtav1`, `mp4v`, `XVID`, `DIVX`

---

#### Safety & Compatibility Enhancements
* Fixes XVID encoding failures on AMD/Intel systems  
* Unifies codec support with 3D Converter & FrameTools  
* AV1 warning system alerts users when OpenCV cannot decode  
* Built to allow full FFmpeg pipeline in next release  

---

### Depth Pipeline Control System

* **Pause**, **Resume**, and **Cancel** depth rendering  
* Real-time resource management during pauses  
* Safe termination prevents file corruption  
* Clear progress status states:
  - Running  
  - Paused  
  - Canceling  
  - Completed  

> Enables fast workflow adjustments and prevents wasted GPU/CPU time.

---

## 5) 3D Pipeline & UX Polish

- Added a **Keep Original Audio** checkbox to optionally copy the source video’s audio into the final 3D render (no re-encoding).
- Hooked a new **image-based 3D pipeline** directly into the main 3D renderer for single-frame conversions.
- Wired the **Mode** selector so it cleanly switches between **Single**, **Batch**, and **Image** workflows.
- Implemented an automatic **3D filename suffix system** so exports are labeled by format and eye mode  
  (for example: `_LRF_Full_SBS`, `_LRF_Half_SBS`, `_VR`, `_Anaglyph`, `_Interlaced`, `_LRF_Left`, `_LRF_Right`).
- Reviewed and cleaned up **multi-language labels and tooltips** across all supported locales.

## 6) Depth Blender – Live Preview & Scrubber

- The **Depth Blender** tab now has a fully working live preview, showing the V2 base map and the blended result side by side.
- All blend parameters (white strength, feather blur, CLAHE, bilateral filters) now update the preview in real time, making it easier to dial in a clean, stable depth mix before running a full batch.


## Summary

v3.7 focuses on stabilizing **Live 3D Capture**, fixing sync and GUI issues, and refining floating window behavior.  
Combined with unified **GPU backend support** and **hardware-accelerated codecs** for depth rendering, VD3D is now significantly more stable across:

- NVIDIA systems  
- AMD ROCm systems  
- Apple Silicon (M1/M2/M3)  
- CPU-only environments

These improvements build the foundation for **Linux** and broader cross-platform support.  
