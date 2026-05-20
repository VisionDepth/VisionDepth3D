# VisionDepth3D v4.1.1 Hotfix Changelog

VisionDepth3D v4.1.1 is a focused hotfix update for v4.1. This patch improves clean-install reliability, fixes missing dependency issues, improves installer behavior, and adds better diagnostics so users can more easily report what the app can see on their system.

## Hotfix Highlights

- Fixed missing FFmpeg and FFprobe issues on clean Windows installs
- Added clearer missing-file detection for FFmpeg tools
- Fixed 3D Generator `[WinError 2]` errors caused by missing external tools
- Improved GPU Diagnostics to check FFmpeg, FFprobe, and NVENC visibility
- Fixed FPS/Upscale frame extraction hanging during video info probing
- Improved frame extraction progress behavior
- Fixed installer/downloader workflow issues
- Added cleaner setup file cleanup and uninstall behavior
- Prevented old loose source folders from interfering with the bundled app

---

## FFmpeg and FFprobe Fixes

- Bundled both `ffmpeg.exe` and `ffprobe.exe` with the application package
- Fixed cases where VisionDepth3D worked on developer systems but failed on clean user installs
- Added a shared FFmpeg tool resolver so the app can check:
  - Bundled app folder
  - `_internal`
  - `ffmpeg` folder
  - System PATH
- Improved missing-file errors so users can see whether `ffmpeg.exe` or `ffprobe.exe` is missing instead of only seeing `[WinError 2]`
- Improved FFmpeg error reporting so render/export failures show more useful backend details

---

## 3D Generator Fixes

- Fixed 3D Generator startup/render failures on systems without FFmpeg installed globally
- Improved 3D render audio merging by using the shared FFmpeg/FFprobe resolver, detecting source audio more reliably, cleaning up failed/empty audio merge files, and using AAC audio for MP4 outputs to avoid codec compatibility issues.
- Improved FFmpeg path handling for:
  - Video probing
  - HDR/raw frame reading
  - Final video writing
  - Audio muxing
  - Split-eye export
- Reduced confusion between missing dependency errors and actual render pipeline errors
- Gated backend 3D render debug logging behind the Debug button
- Reduced normal render-loop console spam during 3D generation
- Improved packaged `.exe` render performance by avoiding unnecessary debug string/output work

---

## Depth Engine Pipeline Updates

- Added a new **Disable Depth Normalizer** option in the Depth Engine
- The Depth Normalizer helps create smoother and more stable depth output across video frames
- When enabled, the Depth Normalizer can reduce depth flicker and help keep depth range more consistent from scene to scene
- Added the option to disable it for users who want faster depth rendering
- Disabling the Depth Normalizer can improve FPS during depth generation, but may result in more visible depth breathing or frame-to-frame depth changes
- Reduced the Depth Normalizer bootstrap behavior to make it lighter and less expensive than the previous scene-level sampling method
- Improved depth pipeline testing tools with internal profiling for decode, preprocessing, inference, postprocessing, and writing stages
- Gated Depth Engine backend telemetry behind the Debug button
- Reduced normal depth-render console output during model inference, depth normalization, and VDA sliding-window processing
- Kept warnings and errors visible while hiding routine debug telemetry unless Debug mode is enabled

---

## FPS/Upscale Fixes

- Fixed frame extraction appearing stuck on `Preparing frame extraction`
- Removed slow exact frame-count probing that could cause long delays before extraction started
- Replaced slow probing with a faster frame-count estimate using metadata
- Added safer extraction progress behavior so progress does not jump to 100% too early when frame count is estimated
- Improved FFmpeg/FFprobe path handling in the FPS/Upscale pipeline
- Fixed mismatched or stale `merged_pipeline.py` import issues caused by old loose install folders

---

## GPU Diagnostics Updates

- GPU Diagnostics now checks whether VisionDepth3D can see:
  - PyTorch
  - CUDA
  - GPU device
  - cuDNN
  - FFmpeg
  - FFprobe
  - NVENC encoders
  - NVIDIA driver info where available
- Diagnostics now reports whether FFmpeg/FFprobe are coming from:
  - Bundled app files
  - `_internal`
  - System PATH
  - Missing
- This should make support much easier when users report render or install issues

---

## Installer and Setup Hub Fixes

- Renamed the downloader/updater workflow to better fit its purpose as a setup hub
- Fixed installer UI crash caused by a missing `open_install_folder` setting
- Added an option to open the actual installed VisionDepth3D folder after installation
- Added a separate cleanup button for downloaded installer and `.bin` setup files
- Added a separate uninstall workflow that runs the Inno Setup uninstaller
- Fixed uninstall button behavior so it runs `unins000.exe` from the real installed VisionDepth3D directory
- Improved separation between:
  - Downloaded setup cache
  - Actual installed app folder
  - Inno uninstaller
  - Setup file cleanup

---

## Packaging Fixes

- Updated packaging flow to include both FFmpeg and FFprobe
- Added installer cleanup for old loose source folders that could shadow the bundled `_internal` app files
- Prevented old folders such as `core`, `ui`, `services`, and `models` from causing mismatched imports after updating
- Kept user/runtime folders safe, including:
  - `weights`
  - `presets`
  - `themes`
  - `settings.json`
  - `ffmpeg`

---

## Notes

This hotfix is recommended for all users who installed v4.1, especially anyone who experienced:

- `[WinError 2] The system cannot find the file specified`
- 3D Generator failing before render
- FPS/Upscale frame extraction not starting
- Missing FFmpeg or FFprobe issues
- Installer/setup cleanup issues
- Stale install files causing import errors
- The Depth Normalizer improves depth consistency, but it can reduce render FPS depending on the model, resolution, and hardware. Users who prefer faster depth generation can disable it, while users who want smoother depth stability can leave it enabled.

v4.1.1 does not replace the larger v4.1 polish update. It is a stability and packaging hotfix to make sure users can install, launch, diagnose, and render more reliably.
