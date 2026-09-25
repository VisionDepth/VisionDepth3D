# VisionDepth3D v5.2 Changelog

## Live 3D Overhaul, Dockable Workspace, Cleaner Z-Splat Rendering, HDR10, and Frame Extraction Fixes

VisionDepth3D v5.2 focuses on making Live 3D more responsive, easier to control, and more stable during playback. It brings the familiar 3D Assistant workflow from the main 3D tab into Live 3D, alongside a customizable dockable workspace, further Z-Splat artifact fixes, HDR10 pipeline repairs, frame-extraction compatibility and progress fixes, and refreshed application visuals.

These notes cover changes since v5.1. The release remains in preparation; no final release date is assigned here.

---

## Highlights

- Reworked Live 3D capture, depth inference, and rendering around the latest available frames.
- Added the main 3D tab's six-control **3D Assistant** and named styles to Live 3D.
- Added live style switching and slider adjustments without restarting playback.
- Added **Use current 3D tab settings** to transfer an existing setup into Live 3D.
- Replaced manual stereo number entry with sliders and visible value readouts.
- Corrected depth-layer mapping and the reversed Live SBS eye order.
- Reduced wavy depth distortion and cumulative motion-warp artifacts.
- Fixed the Live renderer return-value crash.
- Improved scene-cut handling, streaming, shutdown, and runtime diagnostics.
- Further reduced Z-Splat ghost halos and flashing around moving edges.
- Fixed HDR10 input conversion failures in the FFmpeg RGB48 reader.
- Replaced the removed FFmpeg `-vsync` option with `-fps_mode passthrough` in the HDR reader and single-video/folder frame extraction.
- Improved HDR encoder validation, NVENC option handling, and error diagnostics.
- Confirmed a successful HDR10 render using NVENC on the tested Windows setup.
- Added movable, collapsible, floating, and tabbed panels to the 3D Generator workspace.
- Added a dockable shared Job Queue, saved workspace layouts, panel visibility menus, and layout reset.
- Fixed the blank 3D workspace caused by an embedded window being treated as a separate window.
- Fixed extraction progress jumping to 100% when frame-count metadata is missing.
- Refreshed the startup banner and corrected its on-screen sizing.
- Created additional themes and refreshed the companion website.

---

## Live 3D Performance and Responsiveness

### Asynchronous Capture and Depth Processing

- Separated depth inference from the live render loop so rendering can continue while a new depth result is being calculated.
- Added latest-frame capture and depth handoff to avoid building up a queue of old frames.
- Added separate CUDA-stream handling for asynchronous depth work.
- Added scene-cut epochs so results belonging to an earlier scene cannot overwrite the current scene's depth.
- Improved cleanup of capture, inference, and output resources when stopping a session.

### Processing Controls and Diagnostics

- Added **Low Latency**, **Balanced**, and **Quality** presets to the standalone Live launcher.
- Added clearer processing and output limits for balancing responsiveness and image quality.
- Added runtime visibility into frame rate, inference timing, and depth age.
- Corrected generic Hugging Face preprocessing to use the model processor's normalization values and appropriate inference dimensions.

Performance depends on the selected model, resolution, capture source, and GPU. These changes reduce blocking and stale-frame buildup; they do not represent a fixed FPS or percentage speedup across hardware.

---

## 3D Assistant for Live 3D

Live 3D now uses the familiar assistant workflow from the main 3D tab. Each assistant control adjusts the related stereo settings together, making it easier to shape the effect without manually balancing every advanced value.

### Six Familiar Controls

- **3D Strength**
- **Pop-Out**
- **Depth Comfort**
- **Subject Stability**
- **Screen Depth**
- **Subject Zero Lock**

### Live Styles

- Comfortable Cinema
- Strong Pop-Out
- Deep Background
- Close-Up Safe
- VR Comfortable
- Wide / Deep Scene
- Clean Edge
- Showcase Mode
- Custom

**Comfortable Cinema** is the default assistant style.

### Adjust While Watching

- Changing a style applies its stereo settings during the running session.
- Assistant sliders and advanced stereo adjustments update on the next render iteration.
- Related settings are passed together so a frame does not receive a partially updated preset.
- Manual advanced edits switch the style to **Custom**.
- Advanced controls start collapsed and can be expanded for detailed adjustment.

Capture, model, and inference configuration changes still require restarting the session.

### Transfer Settings from the Main 3D Tab

- Added **Use current 3D tab settings**.
- Copies the current assistant and advanced stereo values into Live 3D.
- Preserves manually tuned values by treating the transferred setup as **Custom**.
- Leaves the main 3D tab's settings intact.
- Expanded slider ranges where needed so assistant presets are not clipped by narrower UI limits.

The integrated Live Assistant uses the CUDA Core renderer to support its full set of controls. Live 3D remains a Pro feature.

---

## Live Stereo Controls and Depth Direction

- Replaced manual numeric stereo entry with sliders and visible value readouts.
- Corrected foreground, midground, and background mapping between the UI and renderer.
- Aligned the Live Assistant with the main 3D tab's native preset values.
- Corrected the default SBS eye order after assistant integration, fixing the reversed depth effect.
- Retained the preview's **X** hotkey for swapping eyes.
- Forwarded subject-plane lock, lock width, screen-plane placement, and foreground curvature through the live settings path.
- Connected assistant sharpness to live source processing.

---

## Reduced Waviness and Better Temporal Stability

- Removed cumulative depth reprojection that could repeatedly warp an already-warped depth map.
- Changed optional motion alignment to work from an anchor depth result.
- Disabled motion alignment by default following the wavy-output investigation.
- Removed repeated confidence blending that could progressively flatten depth toward the midpoint.
- Restricted temporal smoothing to stable pixels to reduce trailing during movement.
- Improved scene-cut resets and rejection of outdated depth results.
- Adjusted automatic convergence and window behavior to avoid competing with the chosen stereo settings.
- Enabled subject tracking for the integrated assistant so its stability and screen-depth controls can take effect.

---

## Live Reliability and Streaming Fixes

### Renderer Return-Value Crash

Fixed the crash reported shortly after starting Live 3D:

```text
ValueError: not enough values to unpack (expected 3, got 2)
```

The live path now accepts both two-value and three-value stereo renderer results. Disabling unused metadata no longer breaks output unpacking.

### Streaming and Session Lifecycle

- Moved MJPEG encoding into a background worker.
- Used bounded latest-frame handoff to prevent streaming from accumulating old output frames.
- Improved client notification and stream shutdown/restart behavior.
- Strengthened cleanup when a session exits or encounters an error.
- Routed standalone UI updates through a message queue instead of updating Tk widgets directly from worker threads.

### Standalone Launcher

- Reorganized controls into tabs.
- Added a portable fast-warp fallback for supported standalone use.
- Retained Full/Half SBS and eye-order controls alongside the new processing options.

The portable fallback does not replace the CUDA Core renderer required by the integrated Live Assistant.

---

## Z-Splat Halo and Motion-Edge Fixes

This release further refines the Subpixel Z-Splat renderer introduced in v5.1.

- Removed an unshifted-image overlay that could leave a ghost contour around shifted subjects.
- Corrected handling of stale motion depth to reduce lingering edge artifacts.
- Preserved fractional splat coverage instead of discarding useful partial-pixel contributions.
- Corrected covered-pixel composition so valid projected pixels use the splat result.
- Reduced visible halos and intermittent flashing around moving silhouettes.

Follow-up visual testing reported much less noticeable edge artifacts and no flashing in the tested motion sequences.

---

## HDR10 Rendering and FFmpeg Compatibility

### HDR Input and Output Conversion

- Fixed pixel-format negotiation in the HDR reader that could cause FFmpeg to fail before returning a frame.
- Added an explicit planar 16-bit RGB step (`gbrp16le`) before packing frames as `rgb48le` for the tensor pipeline.
- Corrected the conversion path responsible for the reproduced `YUV color family cannot have RGB matrix coefficients` error.
- Made the output conversion explicitly pass through planar RGB and 10-bit YUV before producing P010 for encoding.
- Kept the HDR conversion in BT.2020/PQ without introducing SDR tone mapping.

### Current FFmpeg Compatibility

- Replaced `-vsync 0` in the HDR reader with `-fps_mode passthrough`.
- Fixed startup failure on FFmpeg builds that no longer recognize `-vsync`.

### Better Decode Diagnostics and Cleanup

- Replaced discarded decoder error output with captured FFmpeg diagnostics.
- Decode errors now include the selected FFmpeg executable, exit code, and received versus expected frame bytes.
- Added complete-frame read handling and explicit reporting of incomplete frames.
- Added validation for invalid frame dimensions and explicit little-endian RGB48 interpretation.
- Improved decoder shutdown during cancellation or generator cleanup, including a terminate/kill timeout.
- Prevented intentional decoder shutdown from being reported as a decode failure.

### HDR Encoder Validation and NVENC Options

- Added early validation for HDR output encoder selection, with guidance to choose a supported 10-bit HEVC or AV1 path.
- Corrected AV1 NVENC to use the `main` profile while retaining `main10` for HEVC NVENC.
- Added checks for encoder-advertised HDR metadata options before passing `-master_display` or `-max_cll` to NVENC.
- Avoided sending those options when the selected FFmpeg encoder does not advertise them.
- Added diagnostic guidance when static metadata options are omitted. The existing libx265 path supplies mastering-display and content-light metadata through x265 parameters.

### HDR Test Results and Setup Notes

- Created a six-second, 1080p, 24 FPS HEVC Main 10 test clip with BT.2020/PQ signaling and static HDR metadata.
- Verified synthetic HDR decoding and software HEVC 10-bit output with BT.2020/PQ tags.
- Confirmed successful HDR10 rendering with NVENC through user testing after the renderer fixes and FFmpeg/driver updates.
- Identified an NVENC API mismatch during testing: the selected FFmpeg build required API 13.1 and reported a minimum NVIDIA driver version of 610.00, while the installed driver exposed API 13.0.

The driver requirement depends on the FFmpeg build. **610.00 is the requirement reported by the tested build, not a universal minimum for every VD3D installation.** Updating the CUDA Toolkit was not required to resolve that encoder mismatch.

For HDR output, enable **Use FFmpeg Renderer** and **Preserve HDR10**, then select a compatible encoder such as **HEVC NVENC** or **libx265**. The Basic Codec selection, including `mp4v`, is ignored when the FFmpeg renderer is active.

Render completion has been confirmed on the tested Windows setup. Independent inspection of that NVENC output's bit depth, color tags, and static HDR metadata remains pending; the successful render alone does not establish full metadata preservation or Dolby Vision support.

---

## Dockable Workspace and Panel Controls

### Arrange the 3D Generator Workspace

- Added movable **Inputs & Render**, **3D Controls**, and **Timeline & Preview** panels around the central preview.
- Added docking at different edges, floating panels, and tabbed panel groups within the workspace.
- Added title-bar controls to collapse/expand, float/dock, and hide panels.
- Kept the 3D Assistant and its advanced tuning tabs together in **3D Controls**.
- Preserved existing processing controls and signal connections while changing their layout containers.

### Shared Job Queue and Workspace Menus

- Made the shared **Job Queue** movable, collapsible, and floatable across application pages.
- Removed fixed maximum heights that restricted queue and log resizing.
- Added **View > Workspace** and **View > 3D Panels** menus for reopening hidden panels.
- Added **Expand All Panels** and **Reset Layout** for each workspace.
- Hiding or collapsing a panel does not stop its running job.

### Layout Persistence and Startup Fix

- Save panel layout, visibility, and collapsed states on normal application exit.
- Store workspace preferences separately from rendering settings.
- Hide floating 3D panels when leaving the 3D Generator tab and restore their intended visibility when returning.
- Reposition restored floating panels onto an available screen.
- Fixed a blank-workspace startup regression by clearing the separate-window flag on embedded dock hosts after construction.
- Added a nested shell/page startup check to cover the arrangement missed by the initial standalone dock tests.

This update covers the 3D Generator's internal panels and the shared Job Queue. Other pages retain their existing internal layouts. Panels dock within their respective workspace hosts; the central 3D preview remains fixed.

---

## Frame Extraction Compatibility and Progress Fixes

### Single-Video and Folder Extraction

- Replaced `-vsync 0` with `-fps_mode passthrough` in both extraction commands on the FPS/Upscale page.
- Fixed the **Frame Extraction Failed** error on FFmpeg builds that report `Unrecognized option 'vsync'`.
- Retained passthrough frame timing to avoid synchronization-based duplication or dropping during extraction.

### More Accurate Progress Reporting

- Fixed the fallback that treated a video with missing frame-count metadata as a one-frame video, causing immediate 100% progress and a zero ETA.
- Added container-duration fallback when stream duration is unavailable, using duration and frame rate to estimate the frame total.
- Handle missing or `N/A` duration metadata without inventing a frame count.
- Display `?` for an unknown frame total and leave ETA unavailable when a reliable estimate cannot be calculated.
- When a single video's total is unknown, retain frame-count and extraction-rate updates without inventing a percentage. Folder extraction falls back to progress by videos processed when any total is unknown.
- Cap running extraction progress at 99%; report 100% after the extraction process finishes. Errors remain explicitly reported as errors.
- Include BMP and WebP alongside PNG/JPEG in the single-video completion file count.

Frame totals derived from duration and frame rate are estimates. These changes improve reporting; they do not claim an extraction-speed increase.

---

## Startup and Visual Refresh

- Added a new cyberpunk-inspired VisionDepth3D startup banner.
- Reworked splash scaling to fit the wide banner's proportions.
- Replaced the old fixed-size presentation with screen-aware sizing.
- Fixed the refreshed artwork appearing too small during startup.

---

## Companion Themes and Website Updates

These items accompany the desktop work and are listed separately from runtime changes.

### Additional Themes

- Created a theme inspired by the refreshed VisionDepth3D website.
- Created a collection of additional themes using the existing JSON theme format.

### Website Refresh

- Updated the website's visual presentation to match the refreshed branding.
- Added separate Windows CUDA and DirectML download choices.
- Clarified the workflow: enhancement/upscaling, depth generation, optional depth blending, and final 3D output.
- Updated the VisionVault section to feature the application window and a download button linking to its release page.

---

## Compatibility and Release Scope

- The new Live Assistant follows the existing main-tab workflow; the main 3D Assistant itself is not new in v5.2.
- Live 3D's Pro requirement is unchanged.
- The integrated Live Assistant requires the CUDA Core path.
- HDR10 output requires FFmpeg and a compatible 10-bit encoder. The basic OpenCV writer remains SDR-only.
- NVENC requires an NVIDIA driver compatible with the API used by the selected FFmpeg build; libx265 provides a CPU encoding path.
- Z-Splat improvements apply to the CUDA renderer. These notes do not announce new DirectML Z-Splat support.
- RIFE 4.25, custom SR model support, batch processing, VR Optimized Merge, and Auto Crop improvements were already documented in v5.1.
- The proposed combined Depth/3D pipeline is not included as a completed v5.2 feature.
- The reported brightness difference between the Depth Engine preview and the Blender's loaded V2 map remains under investigation. A difference between per-image and fixed video normalization is suspected; no corrective change is included as a completed fix.

## Validation Notes

CPU regression checks, offscreen Qt checks, assistant preset/formula checks, renderer-call checks, and eye-order checks were completed during development. Follow-up user testing confirmed improved Live 3D output and the splash sizing fix. HDR validation also covered synthetic decoding, cancellation, error reporting, software HEVC output tags, and NVENC argument construction. User testing confirmed HDR10 + NVENC render completion after the compatibility fixes and environment updates. The resulting NVENC file has not yet been independently inspected for HDR metadata preservation. GPU performance has not been benchmarked across a hardware matrix.


Additional workspace checks passed for collapse/expand, hide/reopen, floating panels, tabbed docks, layout persistence, reset, off-screen recovery, and nested startup visibility. Full application testing of native Windows docking remains pending.

Both updated frame-extraction commands extracted all five frames from a synthetic test video. Progress checks covered missing and `N/A` metadata, container-duration fallback on a test MKV, unknown ETA, the running-progress cap, and completion reporting. User confirmation of the latest extraction progress fix is pending.
