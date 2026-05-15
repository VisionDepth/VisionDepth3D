# VisionDepth3D v4.1

---

## VisionDepth3D v4.1 Changelog

VisionDepth3D v4.1 is a focused polish update built on top of the v4.0 PySide6 release.

This update focuses on the changes added after v4.0, including improved stereo tuning, foreground subject curvature, preset support, convergence behavior, Windows light-mode compatibility, and missing dependency cleanup.

---

## What's New in VisionDepth3D v4.1

### Foreground Curvature Control

Added a new **Foreground Curvature** control to the 3D Generator.

This adds rounded depth only to near/foreground regions using a soft foreground mask and subject-centered curvature shaping. It helps reduce the flat/cardboard look on people, faces, bodies, and hero subjects without globally warping the entire depth map.

Recommended values:

```text
0.00 = disabled
0.04 = subtle curvature
0.06 = recommended default
0.08 = stronger subject volume
0.10+ = can look inflated on some shots
```

---

### Foreground Curvature Preset Support

Presets now support:

```text
foreground_curvature_strength
```

This allows the new Foreground Curvature value to be saved and restored with render presets.

Older presets that do not include this value will fall back to a safe default.

---

### Improved Subject Pop-Out Tuning

Updated the v4.0 stereo tuning around the current shift convention.

In the current v4.0/v4.1 pipeline:

```text
Negative Zero Parallax Strength = Pull subject forward / more pop-out
Positive Zero Parallax Strength = Push subject back / sink into screen
```

This is important for stronger 3D results because negative zero-parallax values allow foreground subjects to sit farther forward instead of being pushed back into the scene.

---

### Dynamic Convergence Backend Adjustment

Adjusted the backend behavior for Dynamic Convergence so the control has a more noticeable effect during final renders.

Dynamic Convergence is treated as a render-time stereo placement trim rather than the main pop-out control.

Recommended control relationship:

```text
FG/MG/BG Shifts        = actual stereo depth separation
Zero Parallax Strength = subject forward/back placement
Foreground Curvature   = subject/body/face roundness
Convergence Strength   = final render-time convergence trim
```

Dynamic Convergence still affects the final render path and may not visibly update in the preview panel the same way Zero Parallax Strength does.

---

### Dynamic Floating Window Awareness

Updated tuning behavior and guidance around Dynamic Convergence and Dynamic Floating Window interaction.

When Dynamic Floating Window is enabled, convergence should generally stay lower because it interacts with the render-time comfort/windowing behavior.

Suggested usage:

```text
Floating Window OFF:
Convergence can be stronger for showcase-style stereo staging.

Floating Window ON:
Convergence should stay lower to avoid overcorrecting the window.
```

---

### Cleaner Strong 3D Tuning

Updated recommended v4.1 tuning around cleaner pop-out and edge-safe rendering.

The strongest results so far come from balancing:

```text
negative zero parallax
foreground curvature
moderate subject lock
strong FG/MG/BG separation
controlled max pixel shift
edge masking enabled
feathering enabled
floating window optional
```

This helps improve:

- foreground pop-out
- face/body volume
- scene stability
- edge safety
- VR comfort
- wide-shot structure
- reduced warping
- reduced edge tearing

---

## Updated Preset Direction

v4.1 presets are being tuned around:

- negative foreground shift
- positive background shift
- negative zero parallax for subject pop-out
- foreground curvature
- adjusted subject lock values
- safer convergence values
- controlled max pixel shift values

Example strong cinema tuning direction:

```json
{
  "fg_shift": -9.9,
  "mg_shift": -3.0,
  "bg_shift": 3.3,
  "zero_parallax_strength": -0.012,
  "max_pixel_shift": 0.071,
  "parallax_balance": 1.0,
  "depth_pop_gamma": 1.0,
  "fg_pop_multiplier": 1.11,
  "bg_push_multiplier": 1.05,
  "subject_lock_strength": 0.85,
  "foreground_curvature_strength": 0.06,
  "convergence_strength": 0.006,
  "enable_dynamic_convergence": true,
  "use_floating_window": false
}
```

---

## Windows Light Mode Compatibility

Fixed an issue where VisionDepth3D could inherit parts of the Windows light app theme when Windows personalization was set to:

```text
Windows Mode: Dark
App Mode: Light
```

This could cause white panels, white scroll areas, or broken-looking UI sections inside the dark VisionDepth3D interface.

v4.1 now applies a more consistent dark base across the main window, panels, scroll areas, group boxes, queue dock, and nested Qt containers so the interface stays visually consistent regardless of Windows app theme settings.

The app still keeps some native/system accent behavior for controls such as sliders and progress indicators where possible, so users may see their Windows accent color reflected in parts of the UI.

User-selectable themes have now been added through the new theme system and Theme Studio workflow.

---

## UI Styling Cleanup

Adjusted the PySide6 stylesheet so the app keeps a consistent dark base while avoiding unnecessary over-styling of native controls.

Changes include:

- darker base styling for panels and nested Qt containers
- improved scroll area dark-mode consistency
- cleaner queue/debug area visibility
- retained native/system accent behavior where possible
- removed unnecessary separate menu-bar stylesheet duplication

---

## Depth Engine Queue Progress Fixes

Improved Depth Engine progress reporting in the Job Queue dock.

Depth processing now reports progress more consistently to the main queue progress bar, including better handling for legacy frame-based progress values from the depth pipeline.

This fixes cases where depth generation was running correctly, but the bottom queue progress bar did not visually fill during processing.

The Job Queue now better supports Depth Engine progress updates such as:

```text
progress percentage
status text
elapsed time
estimated time remaining
system usage telemetry
```

This helps make long depth map renders easier to monitor, especially for full movie depth generation.

---

## Depth Processing Status Labels

Improved Depth Engine status labels during processing.

The Depth Engine now updates the interface more clearly when processing starts, pauses, resumes, finishes, fails, or is cancelled.

Updated status behavior includes:

```text
Processing...
Paused.
Resuming...
Done
Failed
Cancelled
```

This makes the Depth Engine feel more responsive and helps users understand what state the current depth job is in.

---

## Depth Pause and Resume UI Fix

Fixed an issue where the Depth Engine could successfully pause a depth process, but the Resume button did not become clickable afterward.

The Depth Engine page now properly listens for pause and resume state changes from the controller and updates the action buttons correctly.

Expected button behavior is now:

```text
Idle:
Start enabled
Suspend disabled
Resume disabled
Cancel disabled

Processing:
Start disabled
Suspend enabled
Resume disabled
Cancel enabled

Paused:
Start disabled
Suspend disabled
Resume enabled
Cancel enabled
```

This improves reliability when pausing and resuming long depth map generation jobs.

---

## Processing State Handling Improvements

Improved internal processing state tracking for render and depth jobs.

The controller now better tracks when a render or depth process is running or suspended, helping prevent incorrect button states and improving cancel behavior while paused.

Cancel handling was also improved so a paused job can be resumed internally before cancellation, preventing the process from getting stuck in a paused state.

---

## 3D Debug Telemetry

Expanded debug output for stereo validation.

Debug logs can include:

```text
subject depth
zero parallax offset
edge violation left/right
repair mask amount
protect mask amount
warp validity left/right
convergence strength
convergence bias
convergence smoothing
convergence offset
```

Example:

```text
[3DDBG] f=2520 subj=0.436 zpo=0.00897 evL=0.00000 evR=0.00809 rL=0.0008 rR=0.0005 pL=0.0009 pR=0.0017 vL01=1.0000 vR01=1.0000
[CONVDBG] strength=0.03000 bias=0.013689 smooth=0.013968 gain=4.00 offset=0.00005820
```

This helps validate that the stereo pipeline is producing stable eye warps with low repair/protection pressure and clean warp validity.

---


## Latest v4.1 Development Updates

### 3D Generator Render Mode Restoration

Restored and expanded render mode support in the 3D Generator tab.

The 3D Generator now supports clearer render modes:

```text
Single Video Render
3D Image Render
Batch Video Folder Render
Image Folder Render
```

The Sources panel now updates its labels and expected input/output paths depending on the selected render mode.

Examples:

```text
Single Video Render:
Input Video
Depth Map
Output

3D Image Render:
Input Image
Depth Map Image
Output Image

Batch Video Folder Render:
Input Video Folder
Depth Video Folder
Output Folder

Image Folder Render:
Input Image Folder
Depth Image Folder
Output Folder
```

The selected render mode is pushed into the app state before rendering starts, preventing cases where the UI shows one mode but the backend still renders using a previous mode.

---

### 3D Image Render Fixes

Fixed the 3D image render path so still-image conversion works correctly.

This fixes an issue where image rendering could fail with an OpenCV error similar to:

```text
OpenCV error in cvtColor
src is not a numpy array
```

The issue was caused by CUDA tensor output being passed into OpenCV image functions without being converted back into a proper NumPy/OpenCV image first.

The image render path now handles tensor and NumPy frame formats more safely.

---

### Correct Image Aspect and SBS Behavior

Improved still-image 3D output sizing so images are no longer forced into video-style aspect rules by default.

For image rendering, VisionDepth3D now preserves the source image shape unless the user explicitly chooses otherwise.

Correct behavior example:

```text
Square source 627x627:

Full-SBS:
1254x627

Half-SBS:
627x627

Red-Cyan Anaglyph:
627x627
```

This prevents square, portrait, poster, microscope, AI-generated, or unusual-shaped images from being squeezed or cropped incorrectly.

The image render path now treats aspect ratio as an image/eye canvas concern rather than forcing the final packed SBS output into the selected video aspect ratio.

---

### Image Folder Render Progress

Improved Image Folder Render progress reporting.

Image folder rendering now reports batch progress through the shared Job Queue format instead of only showing per-image status text.

The queue now shows consistent batch progress such as:

```text
41.78% | FPS: 3.12 | Elapsed: 00:02:04 | ETA: 00:02:53
CPU: 0% | RAM: 54% | GPU: 13% | VRAM: 50%
```

This makes image sequence rendering easier to monitor and keeps progress behavior consistent with full video rendering and depth generation.

---

### Left / Right Eye Export Debugging Improvements

Improved diagnostics for left-eye, right-eye, and split-eye render issues.

FFmpeg command output is now easier to inspect when frame writing fails, especially for broken pipe errors or output-size mismatches.

Additional safeguards were added around frame dimensions before FFmpeg writes, helping reveal when the generated frame size does not match the expected encoder size.

This helped identify cases where single-eye output was being treated differently from the internal Full-SBS render frame size.

---

### Safer Split-Eye Render Direction

Updated the left/right eye output direction so split-eye rendering can be handled more safely.

Instead of rendering the left eye and right eye as two completely separate 3D conversion passes, the improved direction is:

```text
Render one SBS result
Split the finished SBS into left and right eye outputs
Delete the temporary SBS file
Return the requested left/right outputs
```

This avoids duplicate GPU work and helps keep left and right eye outputs more consistent with each other.

---

### Unified Job Queue Progress Format

Standardized progress reporting across more VisionDepth3D processes.

The Job Queue now uses a cleaner shared format:

```text
74.66% | FPS: 3.85 | Elapsed: 09:08:19 | ETA: 02:43:06
CPU: 89% | RAM: 50% | GPU: 58% | VRAM: 97%
```

This format is now used more consistently across:

```text
3D rendering
Depth rendering
Image folder rendering
FPS/Upscale processing
Frame extraction
Scene detection and scene export
Preview generation
```

This makes long-running operations easier to compare and monitor.

---

### Queue Dock Cleanup

Cleaned up the Job Queue dock behavior.

The queue now separates:

```text
progress/status
system telemetry
debug log output
```

The debug log area can be hidden during normal use and shown only when Debug mode is enabled.

This reduces duplicate-looking bottom panels and keeps the UI cleaner for regular users.

---

### Render Callback and Startup Crash Fixes

Fixed startup issues caused by missing or duplicated render callback methods.

Restored proper render callbacks for:

```text
render started
render finished
render failed
render suspended
render resumed
render cancelled
```

Also fixed an issue where queue log visibility code was placed inside the wrong class, causing startup crashes related to:

```text
AttributeError: 'JobQueueDock' object has no attribute 'queue'
```

---

### Depth Engine Processing Mode UI

Improved Depth Engine source labeling based on processing mode.

The Depth Engine now updates the input section depending on the selected mode:

```text
Process Video:
Input Video

Process Video Folder:
Input Video Folder

Process Image:
Input Image

Process Image Folder:
Input Image Folder
```

The Browse button now opens the correct file or folder picker depending on the selected processing mode.

This avoids confusion where Process Video mode still showed Input Image in the Sources panel.

---

### Depth Engine Browse Behavior

Updated Depth Engine browsing so each processing mode opens the correct selector.

Expected behavior:

```text
Process Video:
video file picker

Process Video Folder:
folder picker

Process Image:
image file picker

Process Image Folder:
folder picker
```

This makes the Depth Engine workflow clearer for both video and image depth generation.

---

### FPS / Upscale Shared Queue Integration

Updated the FPS/Upscale tab to use the shared Job Queue dock instead of relying on a separate local progress bar inside the Render Plan panel.

Frame extraction, scene detection, scene export, RIFE/ESRGAN processing, and preview generation now report through the same queue system used by the rest of the app.

This keeps progress behavior consistent across the application.

---

### Frame Extraction Progress and Completion Feedback

Improved frame extraction from video.

Frame extraction now reports progress to the Job Queue, including:

```text
percentage
FPS
elapsed time
estimated time remaining
system usage
```

A completion message now appears when extraction finishes, so users know where the extracted frames were saved.

This fixes cases where frame extraction completed successfully but the UI did not clearly show anything had happened.

---

### PySceneDetect Progress Integration

Improved PySceneDetect scene detection and scene splitting feedback.

Scene detection now reports progress through the Job Queue.

The process is split into clearer phases:

```text
Preparing scene detection
Scanning video for scene changes
Detected scenes
Exporting scene clips
Done
```

Scene export progress now shows the current scene number and total scene count.

Example:

```text
Exporting scene 4/18
43.00% | Scenes/s: 0.34 | Elapsed: 00:01:12 | ETA: 00:01:36
```

Users now receive a completion or failure message when scene detection/export finishes.

---

### FPS / Upscale Source Tools Layout

Reorganized the FPS/Upscale tab so source preparation tools are easier to find.

The top-left area now groups source tools together:

```text
Source Tools
Extract Frames from Video
Detect Scenes & Extract
Scene Settings
```

Scene detection settings are placed under the Detect Scenes & Extract button so they feel connected to that workflow.

This makes the tab flow more naturally:

```text
Source Tools
Paths
Processing Options
Output Settings
Models
Render Plan
Preview / Job Output
```

---

### FPS / Upscale Render Plan Layout

Updated the Render Plan panel to use a compact grid layout instead of a long vertical list.

The Render Plan now summarizes:

```text
Resolution
Frame Rate
Codec
Pipeline
Models
```

in a wider grid so it uses less vertical space and leaves more room for the preview panel.

---

### FPS / Upscale Preview System

Added a new preview generation system to the FPS/Upscale tab.

Users can now generate sample previews before committing to a full render.

The preview system samples frames from across the selected frame folder:

```text
beginning
25%
50%
75%
near the end
```

Each preview creates an Original vs Preview comparison image.

This allows users to test output settings, upscale quality, blend amount, resolution, and model choice before running a full video process.

---

### Preview Navigation

Added preview navigation controls.

Users can move through generated preview samples using:

```text
Previous
Next
Preview counter
```

This makes it easy to compare different parts of the video without rendering the entire project.

---

### Preview Zoom and Inspection Tools

Added interactive preview inspection tools for the FPS/Upscale preview panel.

Users can now:

```text
hover over the Original or Preview side
use the mouse wheel to zoom into that side
click and drag to pan around the zoomed preview
reset back to the normal view
```

This makes it easier to inspect sharpness, texture detail, pixel quality, and upscale artifacts up close.

The preview inspection workflow is designed for comparing fine details between the original frame and processed preview output.

---

### Preview Panel Size Improvements

Increased the preview panel size so generated previews are easier to inspect.

The Render Plan panel was made more compact so the Preview / Job Output panel has more usable vertical space.

---

### System Accent Color Cleanup

Updated slider and progress styling so controls use the system/application accent color instead of a hardcoded green value.

This keeps the interface more consistent with the user’s Windows accent color and avoids faking a specific color theme.

Also fixed stylesheet formatting issues caused by Python f-string handling of CSS braces.

---

### Qt Text Display Cleanup

Fixed button text display issues caused by Qt treating ampersands as shortcut markers.

This prevents labels like:

```text
Detect Scenes & Extract
```

from displaying incorrectly as:

```text
Detect Scenes_Extract
```

---

### Output and Encoding Label Cleanup

Improved the 3D Generator action labels so the Output & Encoding settings button displays correctly without Qt shortcut marker issues.

This keeps settings labels more readable and avoids confusing button text in the Actions panel.

---

### FPS / Upscale Translation Additions

Added new translation keys for the FPS/Upscale preview workflow, including:

```text
Generate Preview
Previous
Next
Reset View
Mouse wheel over Original or Preview to zoom.
Could not load preview image.
Generate sample previews from the beginning, middle, and end of the frame folder.
```

This keeps the new preview and inspection workflow ready for the existing multilingual UI system.

---


## Latest v4.1 Theme, Layout, and Usability Polish Updates

### User-Selectable Theme System

Added a user-selectable theme system to VisionDepth3D.

Themes can now be loaded from two locations:

```text
resources/themes/
```

Used for official built-in themes shipped with VisionDepth3D.

```text
themes/
```

Used for user-created and user-installed custom themes beside the app.

This keeps the app safe with bundled fallback themes while also allowing users to customize the interface without editing internal resources.

The theme loader now supports:

```text
built-in fallback themes
resource theme files
user theme files
theme reloading from the File menu
user themes overriding bundled themes when IDs match
```

---

### JSON and QSS Theme Support

Themes now support both color palettes and optional stylesheet files.

Supported theme layouts include:

```text
theme_name.json
```

Color-only theme using the default VisionDepth3D page styling.

```text
theme_name.json
theme_name.qss
```

Full theme with custom colors and custom Qt stylesheet control.

```text
theme_name.qss
```

QSS-only theme using fallback default colors.

This gives theme creators a simple path for color themes and a deeper path for advanced themes that customize boxes, sliders, buttons, checkboxes, panels, borders, and other widget styling.

---

### Theme Reload Menu

Added a **Reload Themes** option under the File > Themes menu.

Users can now add or edit files in the themes folder, then reload themes from inside the app without restarting VisionDepth3D.

This makes testing and sharing custom theme packs much easier.

---

### Theme Studio / Create Theme Tool

Added a new **Create Theme...** option under the File > Themes menu.

This opens a built-in Theme Studio window where users can create custom themes visually.

The Theme Studio includes:

```text
theme name entry
live preview panel
color swatch grid
clickable color blocks
native color picker support
Save Theme action
automatic save to the user themes folder
automatic reload and apply after saving
```

The color grid is organized into sections:

```text
Core
Borders
Text
Accents
```

This lets users customize important theme roles such as:

```text
Background
Top Bar
Panel
Panel Dark
Panel Raised
Preview Area
Border
Soft Border
Text
Bright Text
Muted Text
Accent
Accent Text
Danger
Warning
Success
```

This makes theme creation approachable for users who do not want to manually edit JSON files.

---

### Theme Studio Current Theme Styling

Updated Theme Studio so the dialog itself follows the currently selected theme.

Previously, the Theme Studio window used its own hardcoded purple styling, which could look disconnected from the rest of the app.

The Theme Studio now uses the active theme for:

```text
dialog background
panel background
labels
input fields
buttons
section labels
hover accents
```

The live preview still shows the theme being created, while the outer dialog follows the currently active app theme.

This keeps the theme creator visually consistent with VisionDepth3D.

---

### Custom Theme Creation Workflow

Users can now create custom themes entirely inside the app.

Example workflow:

```text
File > Themes > Create Theme...
enter a theme name
click color swatches
pick colors
preview the theme live
save the theme
VisionDepth3D reloads and applies it
```

Themes created this way are saved as JSON files in the user themes folder.

Example custom themes can include styles such as:

```text
Matrix Green
Pinetree Green
Neon Blue
Cyber Purple
Eagle Gold
Crimson Depth
Arctic Light
```

---

### Official and User Theme Separation

Improved the theme folder behavior so official themes and user themes have clear roles.

Official bundled themes belong in:

```text
resources/themes/
```

User-created or downloaded themes belong in:

```text
themes/
```

This helps avoid confusion between shipped themes and user modifications.

---

### Theme Service Log Cleanup

Removed noisy theme loading debug output from normal startup.

The app no longer prints every loaded theme on launch or reload.

Theme loading errors and broken theme warnings can still be kept visible when needed, but normal successful theme loading is no longer spammed into the console.

---

### Unified Page Styling Across Pipeline Tabs

Updated the major pipeline tabs to use a shared page styling system.

The goal was to make every tab feel like part of the same VisionDepth3D suite instead of separate tools with different visual styles.

Pages now follow the same theme-aware styling approach across:

```text
3D Generator
Depth Engine
FPS / Upscale
Depth Blender
Live 3D
```

This improves visual consistency for:

```text
panels
cards
group boxes
buttons
inputs
combo boxes
spin boxes
sliders
checkboxes
preview panels
status labels
scroll areas
```

---

### FPS / Upscale Page Theme Cleanup

Updated the FPS/Upscale page so it no longer uses an overly bright independent stylesheet.

The page now follows the same unified theme system as the rest of the app while keeping its preview, render plan, source tools, and action areas functional.

This makes the FPS/Upscale tab match the cleaner Depth Engine style more closely.

---

### 3D Generator Page Theme Cleanup

Updated the 3D Generator page styling so it better matches the Depth Engine page.

The 3D Generator now uses the shared theme-aware page styling instead of feeling visually separate from the rest of the application.

This improves consistency across controls such as:

```text
Sources
Presets
Actions
Output & Encoding
Stereo Shift
Depth & Parallax
Pop & Subject Controls
Color Grading
Preview panel
```

---

### Depth Blender Theme Support

Updated the Depth Blender page to support the unified theme system.

The hardcoded preview panel styling was removed and replaced with theme-aware preview panel styling.

Depth Blender now follows the active theme for:

```text
mode controls
preset controls
input/output paths
blend parameters
preview frame controls
action buttons
preview panel
```

---

### Live 3D Theme Support

Updated the Live 3D page to support the unified theme system.

The previous lightweight local stylesheet was replaced with the shared theme-aware page styling.

Live 3D now follows the active theme for:

```text
capture source controls
depth model controls
live stereo controls
preview/output settings
status/actions panel
```

---

### Adjustable Page Panels

Added adjustable splitter layouts to more pipeline pages.

Users can now resize page columns by dragging panel dividers.

This improves flexibility for different monitors, resolutions, and workflows.

Resizable areas include combinations such as:

```text
left settings panel
center preview panel
right actions/status panel
```

This was applied across more major pages, including:

```text
3D Generator
Depth Engine
FPS / Upscale
Depth Blender
Live 3D
```

This helps users give more room to previews, controls, or status panels depending on what they are working on.

---

### Adjustable Queue Dock Layout

Improved the layout direction around the bottom Job Queue dock.

The queue area can be resized vertically so users can make the queue/debug area taller when monitoring logs, or smaller when they want more preview space.

This pairs with the new adjustable page columns to make the application layout feel more flexible and professional.

---

### Depth Engine Video Preview Samples

Added a Depth Engine preview workflow for video mode.

Users can now generate sample depth previews before committing to a full video depth render.

The preview system samples frames from the selected video at positions such as:

```text
beginning
25%
50%
75%
near the end
```

The preview area shows:

```text
source frame
generated depth map
```

Users can move through the generated preview samples with Previous and Next controls.

This helps users test model choice, inversion, colormap, inference resolution, and depth settings before running a full video.

---

### Depth Engine Preview Controls

Added preview controls to the Depth Engine preview area.

The Depth Engine now includes:

```text
Generate Preview
Previous
Next
Preview counter
```

This matches the direction of the FPS/Upscale preview workflow but keeps the Depth Engine preview simpler, without zoom and pan tools.

The focus is quick validation of depth output quality.

---

### Depth Engine Queue Format Finalization

Improved the Depth Engine queue integration so it now more closely matches the 3D Generator and FPS/Upscale queue format.

Depth progress now reports in the shared format:

```text
82.87% | FPS: 1.13 | Elapsed: 00:11:00 | ETA: 00:02:16
CPU: 21% | RAM: 57% | GPU: 41% | VRAM: 96%
```

Legacy depth status text such as:

```text
56/7188 | FPS: 1.2 | ETA: 01:43:18
```

is now normalized into the shared queue payload.

This keeps Depth Engine progress consistent with the rest of the app.

---

### Help Menu User Guide Link

Added a **User Guide** entry to the Help menu.

The Help menu now provides a direct path to the online Markdown user guide on GitHub.

This gives users a clearer way to find usage instructions without needing to search through GitHub manually.

The Help menu now includes items such as:

```text
About VisionDepth3D
User Guide
Official Website
GitHub Repository
Documentation / Method
Report a Bug
GPU Diagnostics
```

A future update may add an in-app tutorial or guide panel, but for now the menu item provides a simple and reliable external documentation link.

---

### Theme Creator Translation Additions

Added new translation keys for the Theme Studio and Help menu additions.

New translatable labels include:

```text
Create Theme...
Theme Created
Theme created and applied successfully.
Could not locate the user themes folder.
User Guide
```

These additions keep the new theme creation workflow ready for multilingual UI files.

---

### Theme Creator Stability Fixes

Fixed Theme Creator startup issues caused by method indentation and missing dialog methods during development.

Corrected dialog methods include:

```text
_apply_dialog_style
_slugify
_pick_color
_update_preview
_save_theme
```

This ensures the Create Theme dialog opens, previews, saves, and applies themes reliably.

---

### Theme and Stylesheet Crash Fixes

Fixed additional stylesheet startup crashes caused by Qt CSS being interpreted as Python f-string expressions.

This affected areas where stylesheet blocks contained CSS braces inside f-strings.

The updated styling direction avoids unsafe f-string stylesheet usage and uses safer string replacement or shared theme helpers instead.

This prevents crashes such as:

```text
NameError: name 'background' is not defined
IndentationError: unexpected indent
```

---

### Custom Theme Packs

Added and tested multiple custom theme directions for VisionDepth3D.

Theme examples include:

```text
Neon Blue
Cyber Purple
Eagle Gold
Crimson Depth
Arctic Light
Matrix Green
Pinetree Green
```

These themes help demonstrate the new theme system and give users a starting point for building their own look.

---

## Known Notes

### Wide Shot Edge Tearing

Some wide shots may still show edge tearing if the depth map contains errors around silhouettes, hard object boundaries, thin structures, ships, or complex scene geometry.

In these cases, the issue may come from the depth source rather than the stereo renderer.

Recommended fixes:

```text
lower max_pixel_shift slightly
lower zero_parallax_strength slightly
enable floating window
blend or repair the depth map
use a different depth model
lower foreground curvature if the subject looks inflated
```

### Convergence Preview Behavior

Dynamic Convergence affects the final render path and may not visibly update the preview panel in the same way as Zero Parallax Strength.

Use:

```text
Zero Parallax Strength
```

for preview-visible subject placement.

Use:

```text
Dynamic Convergence
```

for final render-time convergence behavior and floating-window interaction.

---

## Upgrade Note

Users updating from v4.0 should back up:

```text
presets/
weights/
```

---

## Final Result

VisionDepth3D v4.1 is a focused polish and workflow expansion update for the PySide6 rewrite.

It improves:

- foreground subject volume
- strong 3D pop-out tuning
- edge-safe render behavior
- VR comfort tuning
- Dynamic Convergence behavior
- Dynamic Floating Window awareness
- preset support for foreground curvature
- Windows light-mode compatibility
- depth adapter dependency coverage
- debug telemetry
- UI dark-base consistency
- system accent color support for sliders and progress indicators
- Depth Engine queue progress bar reporting
- Depth processing status label updates
- Depth pause/resume button state fixes
- Depth Engine processing mode labels
- restored 3D Generator render modes
- 3D image rendering
- image folder rendering
- image aspect/SBS output correctness
- unified Job Queue progress formatting
- cleaner queue/debug behavior
- FPS/Upscale frame extraction progress
- PySceneDetect queue progress and completion feedback
- FPS/Upscale source tools layout
- compact Render Plan layout
- larger preview panel
- sample preview generation
- preview navigation
- mouse wheel preview zoom
- click-and-drag preview panning
- improved processing state handling for long renders
- user-selectable themes
- JSON and QSS theme loading
- official and custom user theme folders
- Theme Studio / Create Theme workflow
- theme reloading without restarting
- unified theme styling across major pipeline tabs
- Depth Blender theme support
- Live 3D theme support
- adjustable page columns and panels
- adjustable queue dock layout
- Depth Engine video preview samples
- Depth Engine preview navigation
- finalized Depth Engine queue progress formatting
- Help menu User Guide link
- Theme Creator translation additions
- custom theme pack support
- cleaner ThemeService startup logging
- improved packaging and runtime folder handling

v4.0 was the major PySide6 rewrite.

v4.1 is the first major polish, workflow, and usability pass for the new pipeline.

---
