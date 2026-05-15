# VisionDepth3D v4.1 Changelog

VisionDepth3D v4.1 is a major polish, workflow, and usability update built on top of the v4.0 PySide6 rewrite.

This update focuses on making the app feel smoother, more consistent, easier to monitor during long renders, and more customizable for users.

---

## Major Highlights

### Core UI and Workflow Polish

- Added a unified Job Queue progress system across more of the app
- Improved progress display with percentage, FPS, elapsed time, ETA, CPU, RAM, GPU, and VRAM
- Added user-selectable themes
- Added Theme Studio for creating custom themes directly inside VisionDepth3D
- Added support for built-in themes and user themes
- Added theme reloading without restarting the app
- Added adjustable page panels and columns across major workflow tabs
- Improved visual consistency across 3D Generator, Depth Engine, FPS/Upscale, Depth Blender, and Live 3D
- Added a User Guide link under the Help menu
- Cleaned up noisy theme loading console output

---

## 3D Generator

- Restored and improved render mode support
- Added clearer mode behavior for:
  - Single Video Render
  - 3D Image Render
  - Batch Video Folder Render
  - Image Folder Render
- Improved image render support
- Fixed still-image aspect handling for SBS and anaglyph output
- Improved image folder render progress reporting
- Improved left/right eye output direction by moving toward a safer SBS split workflow
- Added better debug output for FFmpeg and frame-size mismatch issues
- Improved stereo tuning around foreground curvature, zero parallax, convergence, and pop-out control

---

## Depth Engine

- Added video preview samples before full depth render
- Preview samples now show original frame and generated depth map side by side
- Added Generate Preview, Previous, Next, and preview counter controls
- Improved Depth Engine processing mode labels
- Fixed Browse behavior for video, video folder, image, and image folder modes
- Improved Depth Engine queue progress reporting
- Normalized legacy depth progress into the shared Job Queue format
- Improved pause, resume, cancel, failed, and done state handling

---

## FPS / Upscale

- Added preview generation before full processing
- Preview samples compare Original vs Preview output
- Added preview navigation with Previous, Next, and preview counter
- Added mouse wheel zoom and click-drag panning for preview inspection
- Improved frame extraction progress and completion feedback
- Integrated PySceneDetect progress into the shared Job Queue
- Reorganized Source Tools for frame extraction and scene detection
- Made the Render Plan panel more compact
- Increased preview area usability
- Moved FPS/Upscale processing feedback into the shared Job Queue

---

## Depth Blender and Live 3D

- Updated Depth Blender to follow the shared theme system
- Updated Live 3D to follow the shared theme system
- Removed hardcoded styling from these pages
- Improved consistency with the rest of VisionDepth3D
- Added adjustable layouts where needed for a more flexible workspace

---

## Theme System

- Added support for official themes in:

```text
resources/themes/
```

- Added support for user-created themes in:

```text
themes/
```

- Added JSON theme support
- Added optional QSS theme support
- Added Theme Studio for creating themes visually
- Added color swatches, live preview, and automatic save/apply workflow
- Added custom theme examples such as Matrix Green, Neon Blue, Cyber Purple, Eagle Gold, Crimson Depth, and Arctic Light

---

## Fixes and Stability

- Fixed stylesheet crashes caused by unsafe CSS inside Python f-strings
- Fixed startup crashes from misplaced or missing callback methods
- Fixed Theme Creator method indentation issues during development
- Fixed Qt ampersand display issues in labels such as Detect Scenes & Extract
- Improved Windows light-mode compatibility so the dark UI does not inherit broken white panels
- Improved packaging support for new v4.1 folders such as themes, dialogs, and styles

---

## Upgrade Note

Users updating from v4.0 should back up:

```text
presets/
weights/
themes/
```

---

## Final Notes

v4.0 was the major PySide6 rewrite.

v4.1 is the first major polish and usability pass on top of that foundation.

This update makes VisionDepth3D feel more consistent, more customizable, and easier to use across long 3D, depth, and FPS/Upscale workflows.
