# VisionDepth3D v5.1 Changelog

## Hybrid 3D Rendering, Subpixel Z-Splat, RIFE 4.25, FPS/Upscale, Batch Workflow, Live 3D, Scene Detection, and Auto-Framing Update

VisionDepth3D v5.1 is the first major post-v5.0 update and focuses on
improving the core rendering engine, processing performance, VR
workflow, and day-to-day usability.

Following the Free/Pro foundation introduced in v5.0, this release adds
the new **Subpixel Z-Splat stereo renderer**, major Hybrid 3D quality
and temporal-stability improvements, a modernized **RIFE 4.25
interpolation engine**, faster and more flexible super-resolution
processing, expanded batch tools, VR-oriented video reassembly, improved
Live 3D, smarter Scene Detection, and automatic cinematic framing.

------------------------------------------------------------------------

# Highlights

-   Added the new **Subpixel Z-Splat stereo rendering method** with
    fractional-pixel projection and depth-aware surface ownership.
-   Added visible **Stereo Renderer selection** for switching between
    Subpixel Z-Splat and the Classic Stereo Warp.
-   Improved close-subject popout and foreground shaping.
-   Improved foreground edge protection, disocclusion repair, and
    right-eye halo control.
-   Reduced breathing, shimmer, and unstable frame-to-frame parallax.
-   Improved Subject Plane Lock, Subject Zero Lock, and screen-plane
    behavior.
-   Added more adaptive Depth-Order Warp processing.
-   Modernized frame interpolation with **RIFE 4.25** and **RIFE 4.25
    Lite**.
-   Added CUDA/FP16 optimized RIFE processing with direct timestep
    interpolation.
-   Improved high-resolution RIFE processing, frame reuse, batching, and
    fallback behavior.
-   Reworked the FPS/Upscale pipeline for better performance and model
    flexibility.
-   Added support for **custom Super Resolution ONNX models**, including
    lightweight modern SR architectures.
-   Added **Single / Batch Folders** processing workflows.
-   Added **Extract Frames from Folder of Videos**.
-   Added automatic scene-to-frame-folder matching for batch processing.
-   Added automatic processed-video naming and audio restoration.
-   Added **Merge Completed Videos** for batch reassembly.
-   Added **VR Optimized Merge** with configurable headset-oriented
    output frame rates.
-   Improved Live 3D model routing, temporal-state handling, and
    aspect-ratio preservation.
-   Improved Scene Detection with adaptive cut detection and
    minimum-scene handling.
-   Improved FFmpeg/FFprobe resolution with system installation support
    and bundled fallback.
-   Reworked **Auto Crop Black Bars** with multi-frame detection and
    automatic active-picture aspect ratio.
-   Fixed Auto Crop zooming and stretched Full-SBS output.
-   Expanded render, RIFE, SR, and workflow diagnostics.
-   Fixed DirectML anaglyph preview corruption by routing unsupported
    Subpixel Z-Splat operations through the Classic Stereo Warp
    compatibility path.
-   Improved Video Depth Anything compatibility on DirectML by avoiding
    unsupported inference-mode tensor behavior.
-   Updated the startup News dialog for v5.1 with reliable version/date
    tracking, localized edition messaging, and a direct link to the full
    GitHub changelog.
-   Expanded localization coverage for FPS/Upscale, Stereo Generator,
    Depth Generation, News, Browse controls, preview guidance, and
    tooltips across the CUDA and DirectML interfaces.

------------------------------------------------------------------------

# Hybrid 3D Rendering Updates

## Close-Subject Popout Improvements

The Hybrid 3D renderer now provides stronger and safer close-subject
popout.

### Improvements

-   Added subject-relative close-popout handling so near parts of a
    subject can move farther toward the viewer without shifting the
    entire subject equally.
-   Improved foreground depth shaping for more natural close-up 3D.
-   Added foreground curvature behavior to give close subjects a
    less-flat stereo appearance.
-   Improved subject-relative depth thresholds so popout is concentrated
    on genuinely closer geometry.
-   Added local popout caps to reduce excessive disparity.
-   Improved frame-edge safety so strong popout is reduced near screen
    boundaries.
-   Added stronger depth-edge protection around faces, heads, shoulders,
    hands, tools, and other hard silhouettes.
-   Improved feathering of the close-popout influence so transitions are
    less abrupt.
-   Improved balance between close foreground detail, the tracked
    subject plane, mid-ground, and background depth.

The goal is stronger native-style 3D closeups without creating the harsh
tearing and outline artifacts that can appear when foreground disparity
is pushed too aggressively.

------------------------------------------------------------------------

## Cinematic Window Depth and Foreground Shaping

-   Improved cinematic window-depth sculpting.
-   Improved foreground, mid-ground, and background disparity weighting.
-   Added stronger subject-aware depth recentering.
-   Improved depth percentile stretching used by the 3D pop pipeline.
-   Improved depth gamma shaping for foreground emphasis.
-   Improved the balance between foreground pop and background pushback.
-   Preserved strong depth separation without requiring Depth of Field
    blur.

------------------------------------------------------------------------

# Subpixel Z-Splat Stereo Renderer

v5.1 introduces a new stereo projection method designed to improve how
displaced pixels are positioned and depth-ordered during eye generation.

The previous Depth-Order Forward Warp used integer destination pixels,
meaning projected coordinates were rounded before being written.
Subpixel Z-Splat preserves fractional destination positions and
distributes projected samples across neighboring pixels.

## Subpixel Projection

-   Replaced single-pixel integer forward writes with fractional
    subpixel splatting.
-   Horizontal stereo projection now preserves the fractional component
    of the destination coordinate instead of discarding it with
    `round()`.
-   Adjacent destination pixels receive weighted contributions based on
    the projected subpixel position.
-   Out-of-view projected samples are discarded instead of being clamped
    onto the image border.
-   Improved projected-edge smoothness before later repair stages are
    applied.

## Depth-Aware Surface Ownership

-   Added a nearest-depth Z-buffer for forward-projected stereo samples.
-   Foreground surfaces receive ownership over farther background
    surfaces when projected samples compete for the same output region.
-   Added a small depth tolerance so samples belonging to the same
    visible surface can contribute without allowing farther geometry to
    bleed through.
-   Weighted color accumulation is normalized after visibility testing
    for smoother projected edges.

## Stereo Renderer Selection

A new **Stereo Renderer** selector is available in the main Stereo
Generator workflow.

Available methods:

-   **Subpixel Z-Splat (v5.1)** - the new default depth-aware subpixel
    renderer.
-   **Classic Stereo Warp** - the previous integer forward-warp method
    for compatibility and A/B testing.

Renderer selection is handled at runtime rather than requiring backend
code changes.

## Compatibility and Fallback

-   The existing continuous `grid_sample` stereo warp remains available
    as the primary safety/compatibility path.
-   The legacy integer Depth-Order Forward Warp remains available when
    Classic Stereo Warp is selected.
-   If the Z-Splat path encounters an unsupported runtime condition,
    VD3D can fall back to the legacy forward-warp path.
-   Existing hole filling and edge-repair stages remain active.
-   Added Z-Splat diagnostic output for validity, Z-buffer ownership,
    depth tolerance, and fill behavior.

The goal is to create cleaner stereo geometry earlier in the pipeline so
later repair stages have less broken projection data to correct.

------------------------------------------------------------------------

# Edge Repair and Stereo Artifact Control

## Cleaner Subject Edges

Major work was completed around stereo edge stability, especially around
close subjects and stronger pixel shifts.

### Improvements

-   Improved contour-safe edge protection around strong depth
    transitions.
-   Added depth-edge stress analysis to identify areas most likely to
    tear or halo.
-   Improved intended-shift-based repair and protection masks.
-   Added stronger contour barriers so repair passes are less likely to
    smear foreground detail into the background.
-   Improved directional background filling for exposed pixels created
    by stereo displacement.
-   Improved one-sided disocclusion repair.
-   Added post-warp visual-stress detection for unstable edges.
-   Added a second post-eye cleanup stage for close-subject halo strips
    and strong local popout artifacts.
-   Improved cleanup around faces, heads, shoulders, hands, and other
    high-contrast silhouettes.

## Right-Eye Halo and Tearing Reduction

A major post-v5.0 stability target was right-eye/displaced-edge tearing
during stronger close-subject popout.

### Changes

-   Improved Depth-Order Forward Warp blending so forward-warp results
    are used only where valid warped pixels exist.
-   Prevented fallback pixels from being blended into invalid regions
    where they could create visible halos.
-   Improved validity-aware forward-warp blending.
-   Added stronger post-eye halo cleanup around displaced silhouettes.
-   Improved background repair behind shifted foreground objects.
-   Reduced outline artifacts created by large foreground disparity
    changes.

These changes allow stronger 3D settings to remain usable without
forcing users to reduce popout simply to avoid edge tearing.

------------------------------------------------------------------------

# Reduced Breathing, Shimmer, and Temporal Instability

Several stability systems were added or refined to reduce frame-to-frame
3D movement that can make depth appear to breathe or shimmer.

## Temporal Stability Improvements

-   Added smoother subject-depth tracking.
-   Improved convergence smoothing.
-   Improved floating-window easing.
-   Added shift-velocity limiting so disparity cannot change too
    aggressively between frames.
-   Improved stabilized occlusion-mask behavior.
-   Added flat-region shift smoothing to reduce noisy disparity movement
    inside otherwise stable surfaces.
-   Added depth-edge stress masking so smoothing is reduced around hard
    silhouettes.
-   Improved temporal handling of close-popout masks.
-   Improved depth-normalization EMA behavior for more consistent
    frame-to-frame depth-range handling.
-   Reduced unnecessary smoothing across important depth boundaries.
-   Added safer temporal-state reset behavior when render dimensions
    change.

The renderer now applies more stabilization in flat and safe regions
while protecting hard subject edges from being blurred or dragged.

------------------------------------------------------------------------

# Subject Plane, Zero-Parallax, and Screen Depth Improvements

## Subject Plane Lock

-   Improved Subject Plane Lock behavior around strong close-subject
    scenes.
-   Added safer silhouette filling around locally locked subject
    regions.
-   Improved interaction between Subject Plane Lock and close-subject
    popout.
-   Improved consistency between tracked subject depth, convergence, and
    screen-plane placement.

## Subject Zero Lock / Screen Depth

-   Fixed behavior where Subject Zero Lock could continue accumulating
    its influence instead of returning cleanly toward neutral.
-   Improved slider behavior so reducing the control toward 0 correctly
    releases the previous lock influence.
-   Reduced stacking between subject-lock and screen-depth adjustments.
-   Improved zero-parallax anchoring around the tracked subject plane.
-   Improved consistency when moving between stronger screen-depth
    settings and neutral values.

------------------------------------------------------------------------

# Depth-Order Warp Improvements

Depth-Order Warp continues to act as an additional stereo stability and
hole-repair path alongside the continuous GPU eye warp.

## Adaptive Runtime Behavior

-   Improved validity-aware blending between the continuous stereo warp
    and Depth-Order Forward Warp.
-   Added adaptive Depth-Order Warp strength based on Edge Repair
    Quality.
-   Added adaptive hole-fill limits based on render resolution.
-   Reduced unnecessarily expensive fill attempts on high-resolution
    renders.
-   Added safer maximum fill limits.
-   Improved performance scaling for Full-SBS and high-resolution
    output.
-   Kept Depth-Order Warp available to the core rendering path while
    making its runtime cost more predictable.

## Edge Repair Quality Integration

Depth-Order Warp now scales more intelligently with:

-   Off
-   Fast
-   Balanced
-   High
-   Showcase

Higher quality settings can use stronger repair behavior while lower
settings prioritize render speed.

------------------------------------------------------------------------

# FPS / Upscale Pipeline Overhaul

The FPS/Upscale workflow received a major post-v5.0 overhaul focused on
processing speed, GPU utilization, model flexibility, batch workflow,
and clearer feedback during long-running jobs.

## Processing and Performance Improvements

-   Reworked major portions of the threaded FPS/Upscale processing path.
-   Improved resolution-aware Super Resolution processing to avoid
    unnecessary work at oversized intermediate resolutions.
-   Improved CUDA execution paths and provider handling.
-   Reduced avoidable CPU/GPU transfer overhead.
-   Improved reuse of frames and processing resources between stages.
-   Improved progress reporting during initialization and first-frame
    processing so long startup work is less likely to appear frozen.
-   Added clearer stage/status output during FPS and upscale jobs.
-   Improved recovery and diagnostics when a model/provider
    configuration cannot be used as requested.
-   Added profiling support to help identify whether interpolation, SR,
    encoding, or I/O is the active bottleneck.

Actual throughput remains dependent on GPU, source resolution, selected
model, scale, interpolation multiplier, and output settings.

------------------------------------------------------------------------

# Modern RIFE Interpolation Engine

v5.1 modernizes the interpolation side of the FPS/Upscale workflow with
newer RIFE processing paths designed for current CUDA hardware.

## RIFE 4.25

-   Added support for **RIFE 4.25**.
-   Added **RIFE 4.25 Lite** as a faster interpolation option suited to
    high-throughput and VR-oriented workflows.
-   Added direct PyTorch CUDA inference for the modern RIFE path.
-   Added FP16 processing where supported.
-   Removed the requirement to use TensorRT for the modern RIFE path.
-   Retained the legacy RIFE ONNX workflow for compatibility.

## Direct Timestep Interpolation

-   Added direct arbitrary-timestep interpolation for modern RIFE
    models.
-   Reduced reliance on recursive midpoint generation.
-   Improved handling of multi-frame interpolation multipliers.
-   Improved temporal consistency by generating requested intermediate
    positions directly from the source frame pair where supported.

## High-Resolution and GPU Optimizations

-   Added high-resolution inference scaling controls.
-   Added timestep batching where appropriate.
-   Improved source-frame reuse across multiple interpolation timesteps.
-   Reduced unnecessary device transfers and repeated preprocessing.
-   Added safer memory-aware behavior for high-resolution interpolation.
-   Added fallback/reduction behavior for GPU out-of-memory conditions.
-   Improved profiling and diagnostic output for modern RIFE processing.

These changes significantly reduce the interpolation bottleneck in
supported workflows compared with the older processing path, while
keeping quality-oriented and compatibility options available.

------------------------------------------------------------------------

# Super Resolution and Custom Upscale Models

The Super Resolution side of the FPS/Upscale workflow is now more
flexible and better suited to both quality-focused and high-throughput
processing.

## Custom SR ONNX Support

-   Added support for loading **custom Super Resolution ONNX models**.
-   Allows users to test and use compatible lightweight or
    quality-oriented SR architectures without waiting for them to be
    hard-coded into VD3D.
-   Improved model/provider handling for custom ONNX SR inference.
-   Added better diagnostic output around model initialization and
    processing.

This makes it possible to use newer lightweight SR approaches, such as
compatible SPAN-based ONNX models, alongside existing VD3D upscale
options.

## Scale-Aware Processing

-   Improved handling of native model scale versus requested output
    scale.
-   Reduced unnecessary processing when the desired output resolution
    can be achieved through a smaller model-input stage.
-   Improved resize planning around high-resolution Full-SBS sources.
-   Reduced excessive intermediate-frame sizes that could previously
    create major performance penalties.

------------------------------------------------------------------------

# Batch FPS / Upscale Workflow

v5.1 expands FPS/Upscale from a primarily single-job workflow into a
more practical scene-based batch pipeline.

## Single and Batch Folder Modes

Added dedicated workflow selection for:

-   **Single**
-   **Batch Folders**

Batch Folders mode allows an entire scene-processing job to be
configured at once.

## Automatic Scene and Frame-Folder Matching

-   Select a folder containing source scene videos.
-   Select a root folder containing extracted-frame folders.
-   VD3D automatically matches scene videos with their corresponding
    frame folders.
-   Matching uses scene/file naming so users do not need to manually
    configure every scene one at a time.
-   Each matched scene can be processed through the selected RIFE and/or
    upscale workflow.

Example:

``` text
scene_005.mkv
scene_005_jpeg_frames/
```

## Batch Audio Restoration

-   Original scene audio can be automatically restored to processed
    outputs.
-   Reduces the need to manually reattach audio after
    interpolation/upscaling.
-   Improved batch handling for scenes where audio is present or absent.

## Editable Paths and Output Selection

-   Improved FPS/Upscale path fields so paths can be edited directly
    when needed.
-   Added clearer folder-selection workflow for batch sources, extracted
    frames, and output.
-   Reduced repetitive browse-and-select operations.

------------------------------------------------------------------------

# Automatic Processed-Video Naming

Batch and processed outputs can now be named automatically according to
the work performed.

Examples:

``` text
scene_005_RIFE2x.mkv
scene_005_RIFE4x.mkv
scene_005_UPS4x.mkv
scene_005_RIFE4x_UPS4x.mkv
```

### Improvements

-   Added automatic RIFE multiplier suffixes.
-   Added automatic upscale multiplier suffixes.
-   Added combined RIFE/Upscale naming.
-   Improved handling so VD3D processing suffixes are not repeatedly
    duplicated.
-   Retained user control over base output naming where required.

------------------------------------------------------------------------

# Batch Frame Extraction

A new **Extract Frames from Folder of Videos** workflow was added.

Instead of extracting frames from every scene manually:

-   Select a folder containing source videos.
-   Select the desired output/root folder.
-   VD3D processes each video in the folder.
-   A separate frame folder is created for each source video.
-   Folder naming is designed to work with the Batch FPS/Upscale
    scene-matching workflow.

This substantially reduces setup time for scene-based movie processing.

------------------------------------------------------------------------

# Batch Video Reassembly

## Merge Completed Videos

Batch processing can now automatically reassemble completed scene
outputs.

### Features

-   Added **Merge Completed Videos** option.
-   Completed scene outputs are collected in scene order.
-   Users can provide a custom name for the final merged video.
-   Individual processed scene files are retained.
-   Merge status is exposed through the batch progress workflow.

## Fast Stitch

A fast stream-copy mode remains available when users simply want
compatible scene files joined without another video encode.

This mode prioritizes speed and preserves the independently encoded
scene streams.

------------------------------------------------------------------------

# VR Optimized Merge

v5.1 adds a dedicated final-merge mode intended for smoother playback of
scene-processed video in VR/headset workflows.

Separately encoded scene files can contain independent timestamps, GOP
structures, and frame-timing characteristics. While stream-copy joining
is extremely fast, those independent characteristics are not always
ideal for a final VR playback file.

**VR Optimized Merge** rebuilds the completed scenes into one continuous
delivery stream.

## VR Merge Features

-   Re-encodes the concatenated scene timeline as one continuous video
    stream.
-   Regenerates timestamps for the final merged output.
-   Uses constant-frame-rate output when a target VR frame rate is
    selected.
-   Uses HEVC NVENC for efficient GPU-accelerated final encoding.
-   Uses broadly compatible `yuv420p` video output.
-   Uses AAC 48 kHz audio for the final delivery file.
-   Includes timestamp/audio resampling safeguards for continuous
    playback.
-   Supports `faststart` for compatible MP4-style containers.

## Configurable VR Output Frame Rate

Available final merge targets include:

-   60 FPS
-   72 FPS
-   80 FPS
-   90 FPS
-   120 FPS
-   Source / Processed FPS

This allows a high-interpolation processing workflow to be delivered at
a practical final frame rate suited to the target headset or playback
configuration.

------------------------------------------------------------------------

# Auto Crop Black Bars Improvements

Auto Crop Black Bars has been substantially reworked.

The previous implementation could fail to detect bars if the first
sampled frame contained a fade, title card, subtitle, logo, compression
noise, or slightly raised black levels. It could also remove the bars
correctly and then force the remaining cinematic frame back into the
manually selected aspect ratio, causing zoomed or stretched output.

## Improved Detection

-   Auto Crop no longer relies on only the first render frame.
-   Added multi-frame black-bar sampling across the selected render
    range.
-   Added stability checks so multiple frames must agree on the detected
    framing.
-   Improved tolerance for compressed blacks and slightly raised black
    levels.
-   Improved detection around title cards, fades, subtitles, and logos.
-   Improved bar-edge handling and overscan behavior.
-   Added safer rejection of unstable or changing bar measurements.

## Automatic Active-Picture Aspect Ratio

When Auto Crop Black Bars is enabled:

-   The detected active picture becomes the authoritative render
    framing.
-   VD3D automatically calculates the active-picture aspect ratio after
    the bars are removed.
-   The manually selected aspect ratio no longer overrides the detected
    movie framing.
-   The already-cropped image is no longer cropped a second time to
    match the selected aspect ratio.
-   Cinematic framing is preserved automatically.

When Auto Crop is disabled, the manually selected aspect-ratio workflow
continues to behave normally.

## Fixed Auto Crop Zooming

Fixed an issue where VD3D could:

1.  correctly remove black bars,
2.  detect a wide cinematic frame,
3.  then crop the sides again because 16:9 or another manual aspect
    ratio was selected.

This caused the rendered image to appear zoomed in.

Auto Crop now keeps the full detected active picture instead.

## Fixed Auto Crop Stretching

Fixed an additional output-geometry issue where the cropped cinematic
image could still be resized into the old uncropped Full-SBS dimensions.

For example, a cinematic source stored inside a 1920x1080 frame could be
cropped to approximately 1920x804, but the renderer could still force
each eye back to 1920x1080, stretching the image vertically.

The output geometry now follows the cropped active-picture dimensions.

Example:

``` text
Encoded source:
1920x1080

Detected active movie frame:
~1920x804

Full-SBS output:
~3840x804
```

The exact cropped height depends on the detected bar size.

------------------------------------------------------------------------

# Live 3D Stability and Compatibility

Live 3D received several fixes to bring it back in line with the current
VD3D depth and stereo architecture.

## Depth Engine Integration

-   Updated Live 3D to recognize callable depth adapters instead of
    assuming every loaded depth model is a raw Hugging Face
    `torch.nn.Module`.
-   Improved routing for optimized Depth Anything v2, Video Depth
    Anything, Depth Anything v3, ZoeDepth, ViGeo, ONNX, and other
    adapter-style backends.
-   Prevented callable adapters from incorrectly entering the generic
    `.to(device).eval()` Hugging Face path.
-   Reduced unnecessary Live 3D GPU synchronization by avoiding unused
    shift-map readback.

## Temporal-State Safety

-   Added safer reset behavior for stereo temporal state when Live 3D
    starts.
-   Fixed shift-EMA shape mismatches when switching between different
    resolutions.
-   Added automatic stereo-state reset when the live capture resolution
    changes.

## Aspect-Ratio Preservation

-   Fixed Full-SBS preview squeezing that could make people and objects
    appear unnaturally thin.
-   Live preview now preserves the Full-SBS aspect ratio instead of
    forcing the stereo frame into a 16:9 preview box.
-   Screen-capture resize behavior now fits within the requested
    processing dimensions while preserving the captured source aspect
    ratio.

## Live Output Reliability

-   Fixed HTTP/MJPEG streaming so frames are pushed while Live 3D is
    running instead of only after the live loop exits.
-   Made optional HTTP-stream dependencies less likely to prevent the
    rest of Live 3D from loading when streaming is not being used.

------------------------------------------------------------------------

# Scene Detection and FFmpeg Workflow Improvements

Scene Detection received additional work to make it more useful for
splitting movie footage into practical processing chunks.

## Smart Scene Detection

-   Added adaptive scene-change detection to reduce over-triggering from
    flashes, camera movement, and strong lighting changes.
-   Kept the user-facing threshold control, with lower values detecting
    more cuts and higher values detecting fewer cuts.
-   Changed the default detection threshold to a more conservative value
    for movie footage.
-   Added minimum-scene handling so very short detections can be merged
    instead of automatically creating large numbers of tiny clips.
-   Added a configurable minimum scene duration for better control over
    exported scene lengths.
-   Kept ContentDetector available as a compatibility fallback when
    adaptive detection is unavailable.

## FFmpeg Resolution

-   Fixed Scene Detection export code so FFmpeg is resolved before scene
    clips are written.
-   Updated FFmpeg/FFprobe lookup behavior to prefer a valid system
    installation when available.
-   Added fallback to VD3D's bundled FFmpeg/FFprobe when a system
    installation is unavailable.
-   Improved diagnostic output so the resolved FFmpeg executable can be
    identified during troubleshooting.

------------------------------------------------------------------------

# Render Performance and Diagnostics

## Adaptive Rendering Improvements

-   Added more resolution-aware Depth-Order Warp limits.
-   Reduced excessive forward-warp fill work on high-resolution output.
-   Improved GPU-side processing around stereo warp and mask generation.
-   Added cached base sampling grids to avoid rebuilding the same render
    grid every frame.
-   Reduced unnecessary GPU-to-CPU transfers in optional matte/roto
    paths when no matte is present.
-   Improved internal reuse of fixed render geometry.

## Render Stage Profiling

Added internal render-stage profiling support for diagnosing slow
sections of the 3D pipeline.

Profiling can measure areas such as:

-   preprocessing and resize
-   temporal depth normalization
-   stereo warp stages
-   Z-Splat / Depth-Order Warp processing
-   repair processing
-   other per-frame render operations

## FPS / RIFE / SR Diagnostics

Additional diagnostic output was added around the media-processing
workflow to make performance bottlenecks and model initialization easier
to identify.

Diagnostics can expose areas such as:

-   RIFE model/backend initialization
-   RIFE inference timing
-   SR model/provider initialization
-   SR processing timing
-   source and processing resolution
-   interpolation multiplier
-   upscale scale
-   batch scene progress
-   final merge stage
-   FFmpeg invocation/failure information

------------------------------------------------------------------------

# Hybrid 3D Diagnostics and Telemetry

Additional internal telemetry was added to make advanced stereo behavior
easier to inspect during testing.

Diagnostic information can now better expose:

-   tracked subject depth
-   zero-parallax offset
-   convergence behavior
-   close-popout behavior
-   edge-window pressure
-   repair-mask strength
-   contour protection
-   forward-warp validity
-   Z-Splat surface ownership
-   Depth-Order Warp behavior
-   render-stage timing

These diagnostics are primarily intended for development and
troubleshooting rather than normal user workflow.

------------------------------------------------------------------------

# Current Auto Crop Behavior

### Auto Crop Black Bars ON

VD3D now follows this framing priority:

``` text
Detect stable black bars
        ↓
Remove black bars
        ↓
Measure active picture
        ↓
Use detected content aspect ratio
        ↓
Ignore manual aspect ratio for framing
        ↓
Render using cropped active-picture geometry
```

### Auto Crop Black Bars OFF

VD3D continues to use the manually selected aspect ratio and normal
framing controls.

------------------------------------------------------------------------

# Final v5.1 Compatibility, DirectML, and Localization Hot Fixes

Final release validation included an additional parity and usability
pass across the CUDA and DirectML packages. These changes focus on
backend-safe behavior, clearer release communication, and complete
runtime translation of recently added controls.

## CUDA and DirectML Release Parity

-   Synchronized the current shared v5.1 changes across Live 3D,
    FFmpeg utilities, the merged FPS/Upscale pipeline, stereo rendering,
    depth rendering, render services, and their associated pages.
-   Preserved backend-specific device, provider, precision, and fallback
    behavior while aligning shared workflow and interface changes.
-   Synchronized current Free and Pro feature definitions across both
    packages, including Job Queue and Projector/MVC feature flags.
-   Completed separate CUDA and DirectML validation passes before final
    packaging.

## DirectML Stereo Renderer Compatibility

Subpixel Z-Splat relies on duplicate-index scatter reductions that are
not currently reliable through the DirectML backend. Although these
operations may appear to be available, they can produce invalid surface
ownership and sparse red/cyan contour fractures in anaglyph previews.

### Fixes

-   Added an explicit DirectML compatibility check before Subpixel
    Z-Splat execution.
-   DirectML now routes depth-order projection through the proven
    Classic Stereo Warp path instead of attempting unreliable Z-Splat
    reductions.
-   Fixed the broken red/blue anaglyph preview artifacts caused by the
    unsupported DirectML reduction behavior.
-   Updated the DirectML Stereo Renderer interface so only the supported
    Classic Stereo Warp option can be selected.
-   Added translated DirectML compatibility guidance to the renderer
    selector and description.
-   CUDA continues to use **Subpixel Z-Splat (v5.1)** as the new default
    renderer, with Classic Stereo Warp remaining available for
    compatibility and A/B testing.

## Video Depth Anything DirectML Compatibility

-   Fixed Video Depth Anything failures caused by DirectML operations
    attempting to use PyTorch inference tensors without a compatible
    version counter.
-   DirectML VDA inference now uses `torch.no_grad()` instead of
    `torch.inference_mode()`, keeping autograd disabled without creating
    incompatible inference tensors.
-   Preserved the optimized inference-mode path for CUDA and supported
    CPU execution.
-   Kept DirectML VDA on the FP32 compatibility path because CUDA-style
    half-precision behavior is not consistently supported by
    `torch-directml`.
-   Improved VDA sequence diagnostics and short-window guidance.

Video Depth Anything remains significantly more demanding on DirectML
than on supported CUDA hardware. Available GPU memory can limit the
usable sequence/window size, and DirectML processing is expected to be
slower because it uses the FP32 compatibility path.

## News and Update Presentation

-   Updated bundled News metadata to **v5.1**, dated **2026-08-18**.
-   Startup News tracking now uses a stable `version|date` identifier so
    viewing an older v5.0 announcement does not suppress the v5.1
    update.
-   The startup announcement is marked as viewed only for its current
    version/date identity.
-   Unified the startup and Help-menu News openers and improved behavior
    when the dialog is already open.
-   Added a direct **View Full Changelog on GitHub** action pointing to
    the Main-Stable technical changelog.
-   Added localized News data support with language-code and English
    fallback behavior.
-   Fixed the Free/Pro edition banner so its status message, action,
    tooltip, and license-management text use the active language.
-   The edition banner now refreshes its license state whenever the News
    dialog is shown.

## Localization and Interface Polish

-   Expanded runtime translation registration across the FPS/Upscale,
    Stereo Generator, and Depth Generation pages in both builds.
-   Fixed controls and descriptions that could remain in the previous
    language after switching back to English.
-   Fixed untranslated **Browse** buttons in Stereo Generator file rows.
-   Fixed the FPS/Upscale preview zoom hint so it updates with the active
    language.
-   Fixed Stereo Renderer descriptions and tooltips so they refresh when
    the language changes.
-   Converted additional static tooltips to registered translation keys
    so they update without restarting VD3D.
-   Added clearer Depth Generation guidance for depth inversion, saved
    frame sequences, depth normalization, VDA depth-range stabilization,
    motion-aware depth smoothing, precision, batch size, overlap, and
    profiling.
-   Improved tooltip refresh behavior across both CUDA and DirectML
    pages.

## Localization Maintenance Tooling

-   Improved the VisionDepth3D Language Tool so a plain entry is saved
    as both its English key and English value.
-   Preserved trailing colons in labels such as `RIFE Backend:` and
    `Minimum Scene:`.
-   Stopped splitting tooltip text at `:` or `=`, allowing complete
    sentences and labels to remain valid translation keys.
-   Added repair handling for malformed entries produced by the older
    separator-based parser.
-   Fixed placeholder validation so ordinary percentage phrases such as
    `Input % settings` and `100% input` are not mistaken for printf
    placeholders.
-   Kept failed translations retryable instead of saving English text as
    a falsely completed translation.

------------------------------------------------------------------------

# Summary

VisionDepth3D v5.1 has grown into a major rendering, performance, VR,
and workflow update following the v5.0 Free/Pro release.

The largest improvements include:

-   new **Subpixel Z-Splat stereo renderer**
-   visible **Stereo Renderer selection**
-   stronger close-subject popout
-   cleaner foreground silhouettes
-   reduced right-eye halo and tearing
-   improved Depth-Order Warp behavior
-   reduced breathing and shimmer
-   improved subject-plane and zero-parallax stability
-   more adaptive edge repair
-   modern **RIFE 4.25 and RIFE 4.25 Lite** interpolation
-   CUDA/FP16 RIFE acceleration and direct timestep interpolation
-   improved high-resolution interpolation processing
-   reworked FPS/Upscale performance
-   custom Super Resolution ONNX model support
-   scale-aware upscale processing
-   new Batch Folders workflow
-   automatic scene/frame-folder matching
-   batch frame extraction
-   automatic audio restoration
-   automatic RIFE/Upscale output naming
-   automatic batch video reassembly
-   new **VR Optimized Merge**
-   configurable 60/72/80/90/120 FPS VR delivery
-   improved Live 3D model routing, temporal-state safety, and
    aspect-ratio handling
-   smarter Scene Detection with adaptive cut detection and
    minimum-scene control
-   system FFmpeg detection with bundled fallback
-   significantly improved Auto Crop Black Bars detection
-   automatic cinematic aspect-ratio handling
-   fixed Auto Crop zooming
-   fixed stretched Full-SBS output after bar removal
-   expanded render, RIFE, SR, and batch diagnostics
-   fixed DirectML anaglyph preview corruption with an explicit Classic
    Stereo Warp compatibility path
-   improved Video Depth Anything compatibility on DirectML
-   reliable v5.1 News version tracking and direct GitHub changelog access
-   expanded live localization and translated tooltip coverage across
    CUDA and DirectML interfaces

The result is a more complete end-to-end 2D-to-3D production workflow:
stronger stereo generation, cleaner projection, faster interpolation and
upscale processing, easier scene-based batch work, and a more practical
final-delivery path for high-frame-rate VR playback.
