# VisionDepth3D v4.0

---

## New PySide6 Interface — A Complete Visual Overhaul

VisionDepth3D v4.0 leaves the old Tkinter UI behind and moves to a modern PySide6 (Qt) interface. Every panel, control, and preview surface has been rebuilt for a cleaner, more responsive desktop experience.

---

## What's New in VisionDepth3D v4.0

**Complete PySide6 Interface Rewrite**  
VisionDepth3D has been rebuilt with a modern PySide6 interface, replacing the older Tkinter-style layout with a cleaner desktop application design. Tabs, dialogs, controls, preview panels, and workflow sections now feel unified across the entire app.

**Modern Dark Theme**  
The full application now uses a cohesive dark theme with consistent styling across tabs, dialogs, controls, cards, buttons, sliders, and status panels. The interface is cleaner, more readable, and better suited for long editing sessions.

**New VisionDepth3D Method**  
The 3D Generator now uses the updated VisionDepth3D Method, featuring subject-aware depth normalization, pop-control depth shaping, structured near / mid / far disparity weighting, GPU stereo warping, dynamic convergence, edge-aware repair, and floating-window protection.

**Updated Shift Convention**  
The new stereo pipeline uses a revised shift direction model. Foreground shift is now typically negative, midground is slightly negative or near zero, and background is positive. This gives the new renderer a more structured near-to-far stereo field, but older presets may need to be rebuilt for the new method.

**Live 3D Preview Tab**  
A new Live 3D tab allows realtime 2D-to-3D testing from camera, capture card, or screen capture sources. Users can select depth models, tune stereo settings, preview SBS output, inspect depth behavior, and test pop/convergence before committing to a full render.

**Depth Engine Model Integration**  
Depth model loading is now tied more closely into the app workflow, with model selection, status feedback, and support for modern depth pipelines. The Depth Engine can provide depth maps for full renders, previews, blending, and Live 3D testing.

**Depth Blender Workflow**  
Depth Blender is integrated into the tabbed workflow for combining and refining depth sources. Users can blend different depth maps, adjust smoothing and contrast controls, preview results, and prepare cleaner depth videos for the 3D Generator.

**FPS / Upscale Enhancer**  
The FPS/Upscale tab adds RIFE interpolation and Real-ESRGAN upscaling workflows for preparing smoother, higher-resolution video sources. It supports merged and threaded pipelines, scene detection, codec options, and progress reporting.

**Responsive Layout Improvements**  
Panels, preview areas, file pickers, parameter cards, and scroll sections now behave better when the window is resized or maximized. Wide workflow pages use scroll-safe layouts so controls stay readable instead of crushing together.

**Debounced Preview Controls**  
Stereo shift, parallax, depth shaping, IPD, and color controls now use debounced preview updates. This prevents heavy preview renders from triggering on every slider tick, making parameter tuning smoother and less frustrating.

**Unified Queue Dock and Progress System**  
Long-running workflows now report through a shared queue/status dock at the bottom of the app. 3D rendering, depth estimation, depth blending, FPS upscaling, and Live 3D status updates are easier to track from one place.

**Language System Integration**  
VisionDepth3D now includes multi-language UI support across major tabs and shell controls. Language files are loaded from the resources folder, with support for English, French, Spanish, German, Japanese, Simplified Chinese, and Traditional Chinese.

**GPU Detection and Acceleration Paths**  
The app detects available acceleration backends and displays GPU/device information during startup and processing. CUDA remains the recommended path for NVIDIA users, with DirectML support available for AMD and Intel GPU users on Windows where supported.

**Cleaner Preset Handling**  
3D presets are now treated as reusable render profiles instead of project files. Presets save stereo, depth, processing, encoding, and color settings without overwriting the user’s current input video, depth map, or output path.

### Result

VisionDepth3D v4.0 is more than a visual refresh. It is a full workflow upgrade built around a modern desktop UI, a new stereo rendering method, realtime Live 3D testing, cleaner depth tools, better progress feedback, and a more professional end-to-end 2D-to-3D conversion pipeline.

---

## 3D Video Generator — New Stereo Pipeline

The stereo rendering pipeline has been rebuilt around the current VisionDepth3D Method.

Depth interpretation, subject placement, convergence, edge handling, and stereo repair now share a more unified mathematical model. The result is cleaner stereo structure, more predictable tuning, and a stronger connection between the depth map, shift field, preview output, and final render.

### Key Improvements

- Reworked depth interpretation and subject-lock flow for more consistent near/far structure
- Depth layers now separate more predictably, with clearer foreground, midground, and background control compared to the older shift model
- Updated zero-parallax subject anchoring so it follows the same depth-weighting logic as the main stereo shift field
- Unified dynamic convergence around the tracked subject depth path
- Reduced reliance on global disparity boosting in favor of cleaner structural depth separation
- Edge tearing, split contours, and silhouette artifacts are reduced, especially on wide shots and fast motion
- Upgraded floating-window behavior to respond to measured edge-risk instead of inferred offset alone
- Added stereo debug telemetry and shift-EMA testing controls for direct pipeline validation
- Added original source resolution and aspect ratio display in the preview metadata bar
- Updated preset behavior so render presets no longer overwrite active video/depth/output paths

### New Shift Direction

The new renderer uses a revised shift convention:

```text
Foreground Shift: usually negative
Midground Shift: usually slightly negative or near zero
Background Shift: usually positive
```

Older presets that used positive foreground values may not transfer directly. Users should start with the new defaults and rebuild older presets using the current convention.

Example natural starting point:

```text
Foreground Shift: -6.0
Midground Shift:  -0.8
Background Shift: +2.2
```

Example stronger showcase point:

```text
Foreground Shift: -8.5
Midground Shift:  -1.2
Background Shift: +3.5
```

### Original Resolution Display

The 3D Generator preview metadata now displays the original source video resolution and aspect ratio when available.

Example:

```text
Original: 1920×1080 (1.78:1)
```

This helps users choose the correct output aspect ratio and avoid accidental stretching or cropping.

### Preset Behavior Change

3D presets now save render settings only.

Presets no longer save:

- input video path
- depth map path
- output path

This means users can switch between stereo presets without losing their currently loaded movie, depth map, or output location.

### Result

The updated pipeline produces cleaner subject anchoring, better separation across depth layers, improved edge handling, and a more mathematically coherent stereo render overall.

---

## Live 3D Preview

VisionDepth3D v4.0 introduces a new **Live 3D** tab for realtime stereo testing.

Live 3D is designed as a realtime sandbox for testing depth models, stereo direction, pop-out behavior, screen capture, camera input, and convergence settings before committing to full offline renders.

### Key Features

- Realtime capture from camera, capture card, or screen source
- Screen 1 / Screen 2 capture support for desktop testing
- Depth model selection using the same supported model list as the Depth Engine
- Lightweight defaults for realtime use
- SBS preview mode for headset and 3D display testing
- Passthrough and depth preview modes for debugging
- Foreground, midground, background, parallax, max shift, and depth pop controls
- Subject tracking, dynamic convergence, edge masking, feathering, and floating-window toggles
- Optional preview masking to prevent screen-capture feedback loops
- Live status reporting through the application status system

### Capture Sources

Live 3D can be used with:

- camera input
- capture cards
- desktop/screen capture
- secondary monitor capture

This makes it useful for testing source footage, games, desktop playback, capture devices, and live preview workflows.

### Depth Model Integration

Live 3D uses the Depth Engine model list instead of requiring users to manually type model IDs. This allows users to quickly switch between supported depth models while keeping Live 3D connected to the same model ecosystem as the main Depth Engine.

### Result

Live 3D acts as a realtime VisionDepth3D testing environment where users can evaluate depth models, stereo controls, capture behavior, and comfort settings before exporting a full video.

---

## Depth Engine Updates

VisionDepth3D v4.0 includes updates to the depth engine, model handling, inference resolution presets, and video-depth workflows.

### Video Depth Anything Improvements

- Improved Video Depth Anything handling for both native PyTorch and ONNX model paths
- Added better runtime detection for Video Depth Anything ONNX models
- Video Depth Anything ONNX is now treated as a sequence-based video model instead of a generic ONNX depth model
- Added fixed temporal-size handling for exported VDA ONNX models
- VDA ONNX now forces the batch size to match the model’s fixed exported frame count, usually `T=8`
- Added safeguards to prevent fixed-T ONNX models from silently truncating larger frame batches
- Added trimming for padded final batches so duplicated padding frames are not written as real output frames
- Native Video Depth Anything keeps its own runtime options such as `target_fps` and `input_size`
- VDA ONNX does not receive unnecessary native VDA extras because its frame count and resolution are already fixed in the exported model
- Updated smoothness testing so `target_fps=-1` can be used to preserve the source video timing instead of forcing low-FPS depth sampling

### Depth Model Resolution Presets

The inference resolution list has been updated with clearer model-specific labels so users can better understand which presets are model-native, repo-default, or video-friendly.

### Depth Flicker and Normalization

Video depth workflows benefit from more stable normalization behavior. This helps reduce depth breathing and flicker in models that produce less stable frame-to-frame depth ranges.

This is especially useful for models that create strong per-frame depth detail but may need temporal or percentile-based stabilization for smoother video output.

### Result

The depth engine is easier to understand, more accurate about model-specific defaults, and better prepared for both image-based and video-based depth models.

---

## Depth Blender — GPU Optimization & Single Image Mode

The Depth Blender has been fully migrated to PySide6 and received significant performance and feature updates.

### GPU Path Rewrite

- The GPU blending path now keeps operations on the GPU where possible
- Reduced redundant CPU round-trips for CLAHE, bilateral filtering, and normalization
- Added `_median_blur_torch` and `_normalize_to_v2_torch_gpu` for GPU-resident processing
- Cached white threshold detection to avoid duplicate percentile calculations
- Improved per-frame blending performance on GPU

### New Single Image Mode

- Added **Image** mode alongside existing Frames and Videos modes
- Blend two depth map images directly without extracting frame sequences
- Live preview with scrubber for frame/video modes
- Instant preview behavior for single images

### Preset System

Added built-in blend presets:

- Default (Balanced)
- Sharp Edges
- Smooth Blend
- Metric + Mono
- High Contrast

These presets allow quick application of common depth blending styles with debounced preview updates.

### Layout and Language Updates

- Depth Blender now supports the new PySide6 page structure
- Labels, buttons, mode controls, presets, and actions are integrated into the language system
- Input/output panels were adjusted for translated text and wider labels
- The page uses the shared queue/progress system instead of redundant local status boxes

### Result

Depth Blender is faster, cleaner, easier to preview, and better integrated into the full VisionDepth3D v4.0 workflow.

---

## FPS / Upscale Enhancement

The FPS/Upscale tab has been fully migrated to PySide6 with a modern card-based layout, live render plan summary, and shared progress integration.

RIFE frame interpolation and Real-ESRGAN upscaling run through the same unified job queue as the other pipelines.

### Key Features

- RIFE interpolation through ONNX workflow
- Real-ESRGAN / RealESR upscaling options
- Merged and threaded processing paths
- Scene detection and extraction workflow
- Codec and output format settings
- Render plan summary
- Session info panel
- Shared queue/progress integration
- Scroll-safe responsive layout for smaller window sizes

### Result

The FPS/Upscale Enhancer is now a first-class workflow tab for preparing smoother and higher-resolution video sources before 3D conversion or VR playback.

---

## Multi-GPU, AMD, Intel, and CPU Support

VisionDepth3D v4.0 improves hardware detection and fallback behavior across different GPU vendors.

### Improvements

- Added DirectML device detection for AMD and Intel GPUs on Windows
- Added ROCm detection for AMD GPUs on Linux
- ONNX providers now auto-detect available execution providers where supported
- Added support paths for `DmlExecutionProvider` and `ROCMExecutionProvider`
- FFmpeg encoder auto-fallback to AMF for AMD and QSV for Intel when NVENC is unavailable
- PyTorch device guards were added throughout the app to prevent crashes on non-NVIDIA systems
- CPU fallback remains available when GPU acceleration is not available

### Install Documentation Update

The install guide now separates PyTorch setup by backend:

- NVIDIA users install CUDA PyTorch
- AMD / Intel Windows users can use `torch-directml`
- CPU-only users can install CPU PyTorch as a fallback

### Result

VisionDepth3D is still best on NVIDIA CUDA, but v4.0 is more flexible for AMD, Intel, DirectML, ROCm, and CPU users.

---

## Language System & Localization

VisionDepth3D v4.0 adds multi-language UI support across the main application shell and major workflow tabs.

### Included Language Files

- English
- French
- Spanish
- German
- Japanese
- Simplified Chinese
- Traditional Chinese

### Improvements

- Main tab buttons now update when switching languages
- File and Help menu entries now translate
- Page labels, buttons, group titles, placeholders, and status labels refresh dynamically
- Language files are loaded from `resources/languages`
- Queue dock status text now refreshes correctly when switching languages
- Added missing translation coverage for Depth Engine, Depth Blender, 3D Generator, FPS/Upscale, and Live 3D
- Language refresh behavior now works more consistently across page widgets, shell controls, and dialogs

### Result

The UI is easier to localize and maintain as VisionDepth3D grows.

---

## Application Shell

### Splash Screen

- Added branded splash screen with loading progress during initialization
- Supports PNG/JPG splash images in `resources/icons/`
- Improves startup presentation and gives users feedback while the app initializes

### Native Menu Bar

- File menu with preset save/load and input shortcuts
- Help menu with links to website, GitHub, documentation, and bug tracker
- GPU diagnostics action in Help menu
- Menu labels are integrated into the translation system

### Debug Console

- Toggle-able debug button in the top bar
- When enabled, console output is captured and displayed in the shared queue dock
- Useful for 3DDBG telemetry, model loading, FFmpeg output, and pipeline diagnostics
- Powered by a stream emitter that redirects stdout/stderr into Qt signals

### Queue Dock

- Shared status area at the bottom of the app
- Tracks workflow status across render, depth, blend, FPS/upscale, and Live 3D tasks
- Status labels update correctly after language changes
- Reduces redundant per-page progress/status boxes

### VRAM Management

- Automatic GPU cache clearing when switching between tabs
- Helps prevent memory accumulation from preview textures and model usage across sessions
- Useful for heavier depth models, preview testing, and long editing sessions

### Result

The application shell now feels closer to a complete professional desktop tool instead of a collection of separate scripts.

---

## Responsive Layout & Scroll-Safe Workflow Pages

Several wide workflow tabs were updated so controls no longer crush together when the window is resized.

### Updated Areas

- FPS / Upscale Enhancer
- Live 3D
- 3D Generator preview and dialog sections
- Depth Blender input panels
- Wide card-based workflow pages

### Improvements

- Scroll-safe body layouts
- Protected minimum panel widths
- Better preview/card scaling
- Cleaner behavior on smaller screens and resized windows
- Less clipping for translated labels
- More stable layout behavior when maximized, resized, or restored

### Result

Workflow pages stay usable and readable even when the application window is not maximized.

---

## Preset System Updates

3D presets now behave as reusable render profiles.

### Changed Behavior

Presets no longer save:

- input video path
- depth map path
- output path

This means users can:

- load a movie once
- load a depth map once
- switch between multiple stereo presets
- keep their current source and output fields intact

### Result

Presets are now better suited for reusable looks, render styles, comfort profiles, and scene-specific stereo tuning.

---

## Requirements and Repository Setup

The project requirements and install documentation have been updated for the v4.0 workflow.

### Requirements Notes

- `PySide6` is required for the new interface
- PyTorch should be installed separately based on the user’s GPU backend
- CUDA users should install CUDA PyTorch from the official PyTorch selector
- AMD / Intel Windows users can install `torch-directml`
- CPU users can install the CPU PyTorch build
- Cache folders such as `__pycache__` should not be uploaded to GitHub
- Core adapters should live inside the `core/adapters/` package structure

### Result

Users cloning the repository get clearer setup instructions, and the source tree stays cleaner.

---

## Upgrade Note

Back up your important folders before replacing an older install:

```text
weights/
presets/
outputs/
custom models or downloaded assets
```

Older presets may still load, but because the v4.0 stereo method uses a new shift convention, users should rebuild or retune older presets using the new negative-foreground / positive-background model.

Then run **VisionDepth3D_Setup_Downloader** to download the official VisionDepth3D v4.0 Windows installer and required `.bin` files.

---

## Final Result

VisionDepth3D v4.0 is a major workflow update.

It combines:

- a full PySide6 interface rewrite
- the new VisionDepth3D stereo method
- Live 3D realtime testing
- Depth Engine model integration
- Depth Blender improvements
- FPS/Upscale workflow support
- multi-language UI support
- better GPU/backend detection
- cleaner preset behavior
- a shared queue/status system
- improved documentation and install guidance

VisionDepth3D v4.0 is built to feel like a unified desktop application for depth generation, depth blending, realtime testing, stereo rendering, and VR-ready video preparation.

---
