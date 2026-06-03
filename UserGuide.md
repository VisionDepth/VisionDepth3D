# VisionDepth3D User Guide 

## Overview  

VisionDepth3D (VD3D) is a high-performance 2D-to-3D conversion suite built for real-time previewing, cinematic stereo rendering, and advanced depth-based video processing.  

It integrates AI depth estimation, pixel-accurate stereo warping, live 3D visualization, FPS interpolation, and AI upscaling into a unified, GPU-accelerated workflow.  

VD3D is designed to scale from fast scene testing to full-length feature conversions, giving creators precise control over depth, comfort, and visual quality.

This user guide walks you through the complete VisionDepth3D workflow, including:

- Generating high-quality depth maps from images and video  
- Blending multiple depth sources for cleaner results  
- Converting 2D footage into cinematic stereoscopic 3D  
- Enhancing FPS and resolution using AI tools  
- Restoring and syncing audio after processing  
- Using the real-time VD3D Live system for live 3D preview and external output  

By the end of this guide, you’ll be able to confidently create smooth, comfortable, and high-quality 3D content using VD3D from start to finish.


---

## Where to Start: Recommended VD3D Workflow

If you are new to VisionDepth3D, start here before jumping into every feature.

The cleanest workflow is:

1. **Prepare your source video**  
   Check the source resolution, frame rate, aspect ratio, and whether it has black bars. If your video changes aspect ratio during the movie, split it into separate sections first.

2. **Generate a depth map**  
   Go to the [Depth Estimation Tab](#depth-estimation-tab) and render a matching depth video from your source. For most users, a Depth Anything V2 model is a good starting point.

3. **Use Depth Normalization when stability matters**  
   In the [Depth Estimation Tab](#depth-normalization), keep Depth Normalization enabled when you want smoother frame-to-frame depth stability. Disable it only when you need faster rendering and can accept more depth breathing.

4. **Optionally blend depth maps**  
   If one depth model gives better subjects and another gives better backgrounds, use the [Depth Blender Tab](#depth-blender-tab) to combine them.

5. **Load the source and depth video into the 3D Generator**  
   Go to the [3D Generator Tab](#3d-generator-tab), load the original video, load the matching depth video, and choose your output format.

6. **Tune the stereo effect in preview first**  
   Use [Preview Modes](#preview-modes), especially **Anaglyph**, **Shift Heatmap**, and **Overlay Arrows**, before running a full render.

7. **Set the screen plane and subject behavior**  
   Use [Screen Plane Offset](#screen-plane-offset-formerly-zero-parallax), [Subject Lock](#subject-lock), and [Dynamic Convergence](#dynamic-convergence) to keep the scene comfortable and stable.

8. **Choose edge repair quality**  
   Use [Edge Repair Quality](#edge-repair-quality) to balance render speed against cleaner edges and disocclusion repair.

9. **Use VR180 settings only when making VR180 output**  
   If you are creating VR180 content, configure [VR180 Output Settings](#vr180-output-settings) after choosing the VR180 output format.

10. **Render a short test clip first**  
    Use [Clip Range Rendering](#clip-range-rendering) to test 10 to 20 seconds before committing to a full movie.

11. **Render the final video**  
    Once the preview and test clip look good, render the full video using your chosen codec and quality settings.

---

## Quick Feature Map

| Goal | Start Here |
|---|---|
| Make a depth map | [Depth Estimation Tab](#depth-estimation-tab) |
| Stabilize depth over time | [Depth Normalization](#depth-normalization) |
| Combine two depth sources | [Depth Blender Tab](#depth-blender-tab) |
| Convert 2D video to 3D | [3D Generator Tab](#3d-generator-tab) |
| Tune stereo depth safely | [Preview Modes](#preview-modes) and [Depth and Parallax Controls](#depth-and-parallax-controls) |
| Control where the screen plane sits | [Screen Plane Offset](#screen-plane-offset-formerly-zero-parallax) |
| Keep subjects stable | [Subject Lock](#subject-lock) |
| Add rounded foreground shape | [Foreground Curvature](#foreground-curvature) |
| Reduce edge artifacts | [Edge Repair Quality](#edge-repair-quality) |
| Create VR180 output | [VR180 Output Settings](#vr180-output-settings) |
| Improve FPS or upscale | [FPS / Upscale Enhancer](#fps--upscale-enhancer) |
| Use real-time 3D | [VD3D Live](#vd3d-live-real-time-2d-to-3d) |


---

## Quick Links: What Are You Working On?

| If you want to... | Go here |
|---|---|
| Learn the full beginner workflow | [Where to Start: Recommended VD3D Workflow](#where-to-start-recommended-vd3d-workflow) |
| Make your first depth map | [Quick Start: Render Your First Depth Map](#quick-start-render-your-first-depth-map) |
| Improve depth stability | [Depth Normalization](#depth-normalization) |
| Blend two depth maps together | [Depth Blender Tab](#depth-blender-tab) |
| Convert a 2D video into 3D | [3D Generator Tab](#3d-generator-tab) |
| Use the new simple 3D controls | [3D Assistant Beginner Controls](#3d-assistant-beginner-controls) |
| Use the full manual 3D controls | [Advanced 3D Controls](#advanced-3d-controls) |
| Fix a flat background | [Layered Depth and Background Depth](#layered-depth-and-background-depth) |
| Keep the subject from flattening with the background | [Subject Stability and Subject Plane Controls](#subject-stability-and-subject-plane-controls) |
| Understand the new screen-depth control | [Screen Depth and Screen Plane Offset](#screen-depth-and-screen-plane-offset) |
| Check depth before rendering | [Preview Modes](#preview-modes) |
| Pick the right codec preset | [Codec Presets](#codec-presets) |
| Make a quick test render | [Clip Range Rendering](#clip-range-rendering) |
| Render for VR180 | [VR180 Output Settings](#vr180-output-settings) |
| Increase FPS or upscale a video | [FPS / Upscale Enhancer](#fps--upscale-enhancer) |
| Use live real-time 3D | [VD3D Live (Real-Time 2D-to-3D)](#vd3d-live-real-time-2d-to-3d) |
| Troubleshoot common problems | [Common Issues & Fixes](#common-issues--fixes) |

## Table of Contents

1. [Overview](#overview)
2. [Where to Start: Recommended VD3D Workflow](#where-to-start-recommended-vd3d-workflow)
3. [Quick Feature Map](#quick-feature-map)
4. [Quick Links: What Are You Working On?](#quick-links-what-are-you-working-on)
5. [FPS / Upscale Enhancer](#fps--upscale-enhancer)
   - [Extract Frames from Video](#1-extract-frames-from-video)
   - [Configure Output Video](#2-configure-output-video)
   - [Set Output Resolution](#3-set-output-resolution)
   - [Set Original FPS](#4-set-original-fps)
   - [Configure FPS Interpolation](#5-configure-fps-interpolation-rife)
   - [Choose Video Codec](#6-choose-video-codec)
   - [ESRGAN Upscaling Settings](#7-esrgan-upscaling-settings)
   - [Choosing a Processing Mode](#choosing-a-processing-mode)
6. [Depth Estimation Tab](#depth-estimation-tab)
   - [Quick Start: Render Your First Depth Map](#quick-start-render-your-first-depth-map)
   - [When Should I Change Settings?](#when-should-i-change-settings)
   - [Adjusting Quality and Performance](#adjusting-quality-and-performance)
   - [Depth Normalization](#depth-normalization)
   - [Output Formats](#output-formats)
   - [Pause, Resume, Cancel](#pause-resume-cancel)
7. [Depth Blender Tab](#depth-blender-tab)
   - [Quick Start](#quick-start)
   - [Blend Parameters](#blend-parameters)
   - [Running a Batch](#running-a-batch)
   - [Output Formats](#output-formats-1)
8. [3D Generator Tab](#3d-generator-tab)
   - [Important: New VisionDepth3D Shift Direction](#important-new-visiondepth3d-shift-direction)
   - [Updating Older Presets](#updating-older-presets)
   - [Getting Started](#getting-started)
   - [3D Assistant Beginner Controls](#3d-assistant-beginner-controls)
   - [Advanced 3D Controls](#advanced-3d-controls)
   - [Layered Depth and Background Depth](#layered-depth-and-background-depth)
   - [Screen Depth and Screen Plane Offset](#screen-depth-and-screen-plane-offset)
   - [Subject Stability and Subject Plane Controls](#subject-stability-and-subject-plane-controls)
   - [Codec Presets](#codec-presets)
   - [VR180 Output Settings](#vr180-output-settings)
   - [Processing Options](#4-configure-processing-options)
   - [Open Preview for Testing](#open-preview-for-testing)
   - [Preview Modes](#preview-modes)
   - [Depth and Parallax Controls](#depth-and-parallax-controls)
   - [Depth Shaping: Pop and Subject Controls](#depth-shaping-pop-and-subject-controls)
   - [Pop-Out vs Depth Layering](#pop-out-vs-depth-layering)
   - [Optional Advanced Controls](#optional-advanced-controls)
   - [Clip Range Rendering](#clip-range-rendering)
   - [Recommended First-Time Workflow](#recommended-first-time-workflow)
   - [Recommended Starting Presets](#recommended-starting-presets)
   - [Troubleshooting](#troubleshooting)
9. [VD3D Live (Real-Time 2D-to-3D)](#vd3d-live-real-time-2d-to-3d)
10. [Recommended Workflow Summary](#recommended-workflow-summary)
11. [Best Practices for High-Quality 3D](#best-practices-for-high-quality-3d)
12. [Hardware / Backend Support](#hardware--backend-support)
13. [Performance Optimization Tips](#performance-optimization-tips)
14. [Common Issues & Fixes](#common-issues--fixes)
15. [When to Use Depth Blending](#when-to-use-depth-blending)
16. [Support & Updates](#support--updates)
17. [End of User Manual](#end-of-user-manual)

---

## FPS / Upscale Enhancer  

The FPS / Upscale Enhancer tab allows you to:  

- Increase video smoothness using AI frame interpolation (RIFE)  
- Enhance resolution using AI upscaling (Real-ESRGAN)  
- Automatically split long videos into manageable scenes using PySceneDetect  
- Rebuild high-quality output videos with hardware-accelerated encoding  

This system is ideal for improving older content, low-resolution sources, and creating ultra-smooth playback for VR and high refresh rate displays.

---  

### 1. Extract Frames from Video  

1. Click **Extract Frames from Video** and select your source video  
2. Click **Select Output Folder** to choose where frames will be saved  
3. Choose an image format:  
   - **JPG** for lower memory usage and faster processing  
   - **PNG** for maximum quality  
4. Once extraction completes, the **Input Frames Folder** will automatically populate with the extracted frames  

---

### 2. Configure Output Video  

1. Select **Output Video File** and choose a format (MP4, MKV, AVI, etc.)  
2. Enable processing options:  
   - **RIFE Interpolation** for FPS enhancement  
   - **ESRGAN Upscaling** for resolution improvement  
   - Enable both if desired  

---

### 3. Set Output Resolution  

Enter your target output resolution (Width × Height).  

**Example:**  
Original: `720 × 480`  
Upscaled Output: `2880 × 2160`  

*(4× upscaling in both dimensions)*  

---

### 4. Set Original FPS  

Enter the original frame rate of the source video.  

**Example:**  
If the original clip is **29.97 FPS**, enter **29.97**  

This ensures proper interpolation timing and smooth output.

---

### 5. Configure FPS Interpolation (RIFE)  

If RIFE is enabled, select the FPS multiplier:  

- **×2** (30 → 60 FPS)  
- **×4** (30 → 120 FPS)  
- **×8** (30 → 240 FPS)  

Higher values create ultra-smooth motion but require more processing time and may introduce more artifacts.

---

### 6. Choose Video Codec  

Select your preferred encoder:  

- **H.264 / H.265 CPU encoding** (universal compatibility)  
- **NVENC GPU encoding** (recommended for NVIDIA GPUs for speed)  

---

### 7. ESRGAN Upscaling Settings  

**AI Blending Strength**  
Controls how much of the AI-enhanced detail is blended with the original frame:  

- Lower values = stronger AI sharpening  
- Higher values = more original texture preserved  

**Input Resolution Scaling**  
Downscales the input frame before AI upscaling to:  

- Reduce memory usage  
- Increase processing speed  
- Still achieve high-quality results  

---

## Choosing a Processing Mode

VisionDepth3D provides two different processing modes for FPS interpolation and upscaling.  
Both produce the same visual results, but differ in how they use system resources and performance flow.

---

### Start Processing Button

The **Merged Pipeline** runs interpolation and upscaling in a single sequential workflow:

1. A frame pair is interpolated using RIFE  
2. The interpolated frames are immediately passed through ESRGAN (if enabled)  
3. Frames are written directly to the output video before moving to the next pair  

#### Key Characteristics:

- Simpler processing flow  
- Very stable and predictable  
- Uses less system memory  
- Ideal for:
  - Lower-end systems  
  - Long videos  
  - Maximum reliability  

#### Recommended when:

- You experience stuttering or memory limits  
- You want guaranteed smooth processing  
- You are running very high resolutions  

---

### Threaded RIFE + ESRGAN Button

The **Threaded Pipeline** runs interpolation, upscaling, and video writing in parallel using multiple worker threads:

• One thread generates interpolated frames (RIFE)  
• One thread upscales frames (ESRGAN)  
• One thread writes frames to the output video  

Frames are buffered and synchronized to maintain correct ordering.

#### Key Characteristics:

- Much higher throughput  
- Better GPU utilization  
- Faster overall render times  
- Slightly higher memory usage  

#### Recommended when:

- You have a strong GPU  
- You want maximum performance  
- You are processing shorter clips or high FPS output  

---

### Which Should I Use?

| Pipeline          | Stability     | Speed        | Memory Use | Best For |
|-------------------|---------------|--------------|------------|----------|
| Merged Pipeline   | ⭐⭐⭐⭐⭐  | ⭐⭐⭐     | Low        | Long renders, reliability |
| Threaded Pipeline | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐ | Medium     | Fast high-performance jobs |

---

### Visual Quality

Both pipelines produce **identical final video quality**.  
The difference is strictly in **processing speed and system resource usage**.

Choose based on your hardware and workload.

---
## Depth Estimation Tab

The Depth Estimation tab generates depth maps from images or videos using AI models.

If this is your first time using VisionDepth3D, follow the steps below to render your first depth map.

---

## Quick Start: Render Your First Depth Map

### Rendering a Depth Map from an Image (Beginner Method)

1. Open the **Depth Estimation** tab.

2. Select a Model  
   - Choose a recommended model such as **Depth Anything V2**.  
   - The first time you load a model, it may take a moment to initialize.

3. Choose an Output Directory  
   - Click **Choose Directory**.  
   - Select a folder where your depth map will be saved.

4. Leave Settings at Default  
   For your first test, keep:
   - Colormap: Default  
   - Invert Depth: Off  
   - Batch Size: Default value  
   - Inference Resolution: A preset like 512×288 or 704×384  

5. Click **Process Image**

6. Select your image file.

VD3D will:
- Generate a depth map
- Display the result in the preview window
- Save the output as:  
  `yourfilename_depth.png`

You have successfully created your first depth map.

---

### Rendering a Depth Video (Beginner Method)

1. Select your model.
2. Choose an Output Directory.
3. Keep default settings for your first run.
4. Click **Process Video**.
5. Select your video file.

VD3D will:
- Process each frame
- Generate a depth video
- Save it as:  
  `yourvideo_depth.mkv`

If you only need depth for 3D conversion, you typically do not need to change colormap or other advanced settings.

---

## When Should I Change Settings?

After your first successful render, you can begin adjusting:

- Increase **Inference Resolution** for more detailed depth.
- Increase **Batch Size** if your GPU has available VRAM.
- Enable **Invert Depth** if near/far values appear reversed.
- Enable **Save Frames** if you need individual depth PNG files.

For most users, default settings work very well.

---

## Adjusting Quality and Performance

After your first successful render, you can fine-tune performance and detail.

### Inference Resolution

Controls internal processing resolution.

- Lower resolution = faster processing
- Higher resolution = more detailed depth

For full movies, many users start at 512×288 and increase if needed.

---

### Batch Size (Frames)

Controls how many frames are processed at once.

- Higher values = faster on strong GPUs
- Lower values = safer if you hit VRAM limits

If you run out of memory, reduce this first.

---

### Invert Depth

Flips near and far values.

Enable this if foreground objects appear darker when they should be closer.

---

### Depth Normalization

Depth Normalization stabilizes the depth range across video frames.

When enabled, VD3D analyzes the depth output and keeps the near/far range more consistent over time. This helps reduce depth flicker, sudden depth jumps, and frame-to-frame breathing during video depth generation.

Recommended behavior:

| Setting | Best For |
|---|---|
| Depth Normalization On | Full movies, smoother depth, better 3D stability |
| Depth Normalization Off | Faster depth rendering, quick tests, users who want maximum speed |

Depth Normalization can cost some render speed depending on the model, resolution, and hardware. If you need the fastest possible depth render, disable it. If you want smoother depth for final 3D conversion, leave it enabled.

---

### Save Frames (Video Mode)

When enabled, VD3D saves individual depth PNG frames in addition to the depth video.

Useful for:
- Manual inspection
- Custom 3D workflows

---

### CPU Offload Mode

`
This only activates when a diffusion based model is selected like Marigold depth model.
`

Reduces VRAM usage by moving parts of the model to CPU.

- None = fastest, highest VRAM usage
- Sequential = balanced
- Full = lowest VRAM usage, slowest

Only adjust this if you encounter memory limits.

---

### Use float16

`
Enable FP16 before loading/reloading DAv2 models.
`

Reduces VRAM usage and can increase speed on supported GPUs.

Recommended for CUDA GPUs.

---

### 3. Choose What You Want to Process

Select one of the following:

- **Process Image** – Single image input  
- **Process Image Folder** – Batch image processing  
- **Process Video** – Generate depth video  
- **Process Video Folder** – Batch video processing  

For your first test, use **Process Image**.

---
## Output Formats  

**Image depth outputs:**  
`filename_depth.png`  

**Video depth outputs:**  
`filename_depth.mkv`  

Depth maps are saved in grayscale format and are ready for use in the 3D Generator tab.

---

## Pause, Resume, Cancel  

- **Pause** temporarily halts processing  
- **Resume** continues where it left off  
- **Cancel** safely stops processing  

> **Note:**  
> For best results in the Depth Blender tab, render two separate depth maps:
> - One using a Depth Anything V1 Base model (white-balanced source)  
> - One using a Depth Anything V2 Large model (Base Source)  
>
> Blending these two depth sources improves edge stability, subject separation, and overall depth consistency.

---

## Depth Blender Tab

The Depth Blender tool lets you merge two different depth sources into one cleaner depth result.

It is designed for cases where:
- One model produces strong subject separation but noisy backgrounds
- Another model produces stable backgrounds but weaker subject edges
- You want to blend both into a single depth map or depth video that behaves better in 3D conversion

You can run it on:
- **Folders of PNG depth frames**
- **Two depth videos**

A live preview panel lets you scrub frames and see adjustments instantly before running a full batch.

---

## Quick Start

### 1. Choose a Mode

In the **Mode** section select one:

- **Folders (frames)** for depth frame sequences (`.png`)
- **Videos** for depth videos (`.mp4`, `.mkv`, `.avi`, `.mov`)

---

### 2. Load Your Inputs

Under **Inputs**:

1. Set **V1 path**
2. Set **V2 path** (this is the “base” depth map)

Notes:
- **V1** is used to contribute extra detail or stronger whites where needed
- **V2** is treated as the main reference depth that the output is normalized to

---

### 3. Choose Output Behavior

#### Frames Mode Output

If you are using **Folders (frames)** you have two options:

- **Overwrite V2**  
  The blended frames replace the original PNGs inside the V2 folder.

- **Output Folder**  
  Turn off overwrite and select an output directory to save blended frames separately.

#### Video Mode Output

If you are using **Videos**, select an output file location such as:
- `blended_depth.mp4`

---

### 4. Optional Final Size

Under **Final Size (optional)**:

- Leave **Width** and **Height** blank to keep the original resolution
- Enter values to force the output size for every frame

Example:
- Width: `1920`
- Height: `1080`

---

### 5. Preview Before Batch

Use the live preview tools to verify your blend:

1. Click **Preview Now**
2. Use the **Preview Frame** slider to scrub
3. Use the arrow keys:
   - Left Arrow goes to the previous frame
   - Right Arrow goes to the next frame

The preview shows:
- **V2 Base** on the left
- **Blended Output** on the right

---

## Blend Parameters

These sliders update the preview live.

### White Strength
Controls how strongly V1 can contribute its high depth whites into V2.

- Lower values keep output closer to V2
- Higher values inject more of V1’s bright depth regions

### Feather Blur (kernel)
Controls the softness of the blending transition.

- Low values create sharper merges
- Higher values create smoother, more gradual blending

### CLAHE Clip Limit
Boosts local contrast in the blended result.

- Higher values can increase depth “punch”
- Too high can increase noise

### CLAHE Tile Grid
Controls how localized the CLAHE contrast enhancement is.

- Lower tile size can increase detail but may look harsher
- Higher tile size is smoother and more global

### Bilateral d
Strength of edge-preserving smoothing.

- Higher values smooth more while keeping edges
- Too high can soften fine detail

### Bilateral sigmaColor
How much intensity difference is allowed during smoothing.

- Higher values smooth more aggressively
- Lower values protect contrast

### Bilateral sigmaSpace
How far smoothing spreads spatially.

- Higher values affect larger areas
- Lower values keep smoothing tighter

---

## Running a Batch

When your preview looks correct:

1. Click **Start Batch**
2. Watch the progress bar and log window
3. Click **Stop** if you need to cancel safely

---

## Output Formats

**Frames mode output:**  
`filename.png` (blended depth frames saved as PNG)

**Video mode output:**  
`blended_depth.mp4` (grayscale depth video output)

The blended results are grayscale depth and are ready to use in the 3D Generator tab.

---

# 3D Generator Tab

The **3D Generator** tab converts a 2D source video and its matching depth map into a stereoscopic 3D video.

This is the final stage of the VisionDepth3D workflow. It takes:

- the original 2D video
- the generated or blended depth map video
- your stereo/parallax settings
- your output and encoding options

and renders a final 3D video using the current **VisionDepth3D Method**.

The current method uses subject-aware depth normalization, pop-control depth shaping, structured near / mid / far disparity weighting, GPU stereo warping, edge-aware repair, dynamic convergence, floating-window protection, cinematic depth sculpting, and optional layered depth-order warping to create a controllable stereo result.

The 3D Generator now has two levels of control:

- **3D Assistant Beginner Controls** for users who want a simpler guided workflow.
- **Advanced 3D Controls** for users who want full manual control over stereo depth, convergence, subject locking, depth shaping, edge repair, and layered background depth.

The goal of the newer 3D system is not only to push subjects forward. It is designed to make the background feel like it sinks behind the screen, while keeping subjects separated so the whole image does not collapse into a flat sheet.

---

## Important: New VisionDepth3D Shift Direction

VisionDepth3D now uses the updated VisionDepth3D Method for stereo generation.

This method uses a different stereo shift convention than older versions of VisionDepth3D.

In the current pipeline:

- **Foreground Shift is usually negative**
- **Midground Shift is usually slightly negative or near zero**
- **Background Shift is usually positive**

This may feel opposite from older VisionDepth3D presets.

Older presets that used positive foreground values may now produce a very different stereo result. If you are updating from an older version, it is recommended to start from the new default presets instead of copying older shift values directly.

### Recommended starting range

| Control | Recommended Range |
|---|---|
| Foreground Shift | `-5.0` to `-10.0` |
| Midground Shift | `-0.5` to `-2.0` |
| Background Shift | `+2.0` to `+5.0` |

### Example natural preset

```text
Foreground Shift: -6.0
Midground Shift:  -0.8
Background Shift: +2.2
```

### Example stronger showcase preset

```text
Foreground Shift: -8.5
Midground Shift:  -1.2
Background Shift: +3.5
```

### Example aggressive pop preset

```text
Foreground Shift: -10.0 to -12.0
Midground Shift:  -2.0
Background Shift: +4.0 to +5.0
```

The 3D effect comes from separation between near, mid, and far depth regions. In the current renderer, negative foreground shift pulls near objects toward the viewer, while positive background shift pushes distant areas deeper behind the screen plane.

A good basic relationship is:

```text
Foreground Shift < Midground Shift < Background Shift
```

Example:

```text
FG -6.0 / MG -0.8 / BG +2.2
```

---

## Updating Older Presets

Presets created for older VisionDepth3D versions may not transfer directly to the new method.

If an older preset used positive foreground values, it may now:

- push depth in the wrong direction
- reduce pop-out
- make the scene feel inverted
- create uncomfortable or flat stereo separation
- produce a very different output than expected

### Old-style approach

```text
Foreground: positive
Background: negative
```

### Current method approach

```text
Foreground: negative
Midground: slightly negative or near zero
Background: positive
```

When converting older presets, do not simply copy the same numbers. Start with one of the new default presets, then tune using Preview Mode, Shift Heatmap, and Anaglyph preview.

---

## Getting Started

### 1. Load Your Inputs

You must provide:

- **Input Video**  
  The original 2D source video.

- **Depth Map Video**  
  The matching depth map video generated from the Depth Engine tab or Depth Blender.

- **Output Path**  
  The location where the final 3D video will be saved.

The source video and depth map video should match in:

- resolution
- frame count
- frame rate
- clip length

If they do not match, the stereo render may drift, desync, or produce incorrect depth alignment.

---

### 2. Check Original Resolution and Aspect Ratio

The 3D Generator displays the original source video size in the preview metadata bar when available.

Example:

```text
Original: 1920×1080 (1.78:1)
```

Use this information to choose the correct output size and aspect ratio.

Common source sizes:

| Source Size | Aspect Ratio | Notes |
|---|---:|---|
| `1920×1080` | `1.78:1` | Standard 16:9 |
| `3840×2160` | `1.78:1` | 4K 16:9 |
| `1920×800` | `2.40:1` | Cinematic widescreen |
| `1440×1080` | `1.33:1` | 4:3 style |
| `1080×1920` | `0.56:1` | Vertical 9:16 |

This helps users avoid accidentally stretching or cropping their video into the wrong output shape.

---

### 3. Choose Output and Encoding Settings

Use **Output & Encoding** to configure how the final 3D video is packaged.

Here you can set:

- **Output Format**
  - Full-SBS
  - Half-SBS
  - VR
  - VR180 Equirect Top-Bottom
  - VR180 Equirect Side-by-Side
  - Red-Cyan Anaglyph
  - Passive Interlaced

- **Stereo Output**
  - SBS
  - Left eye only
  - Right eye only
  - Both eyes separately

- **Aspect Ratio**
  - Default 16:9
  - Classic 4:3
  - Square 1:1
  - Vertical 9:16
  - CinemaScope / Anamorphic / UltraWide formats

- **Codec**
  - H.264 / H.265 CPU encoding
  - NVENC for NVIDIA GPUs
  - AMF for AMD GPUs
  - QSV for Intel GPUs
  - AV1 options where supported

- **Audio Handling**
  - Keep original audio when available
  - Export video-only if audio is not needed

- **HDR10 Preservation**
  - Use when working with compatible HDR source material

NVENC H.264 or NVENC H.265 is recommended for NVIDIA users who want faster encoding.

If you are not sure what codec to choose, use the new **Codec Presets** section instead of manually tuning every codec option. Start with **Fast Preview - Quick Test** for short tests, then use **Balanced Final - NVIDIA**, **High Quality Final - NVIDIA**, **Small File - HEVC**, or **Compatibility Mode - Plays Everywhere** depending on the final goal.

---

## 3D Assistant Beginner Controls

The **3D Assistant** is the recommended starting point for new users. It gives simple controls that adjust the advanced stereo system behind the scenes.

Use the 3D Assistant when:

- you are new to VD3D
- you want fast results without learning every advanced slider
- you want a safe starting point for a movie
- you are tuning preview frames before a full render
- you want to quickly switch between comfort, pop-out, deep background, and clean edge styles

The 3D Assistant does not replace the advanced controls. It gives a cleaner front-end for common tuning choices. If you want more control, open **Advanced 3D Controls** after the beginner settings are close.

---

### 3D Style

The **3D Style** dropdown chooses a preset starting point.

Common styles include:

| Style | What it is for |
|---|---|
| Comfortable Cinema | Balanced, safer movie viewing |
| Strong Pop-Out | More forward subject and foreground depth |
| Deep Background | More background recession and room depth |
| Close-Up Safe | Safer faces, dialogue shots, and subject-heavy scenes |
| VR Comfortable | Reduced strain for headset viewing |
| Wide / Deep Scene | Stronger scene depth for rooms, landscapes, and wide shots |
| Clean Edge | Safer edges with less tearing or halo stress |
| Showcase Mode | Stronger demo-style 3D for short clips |
| Custom | Used when you manually adjust the sliders |

When a user changes the beginner sliders manually, the style can switch to **Custom**. This is normal.

---

### 3D Strength

**3D Strength** controls the overall amount of stereo depth.

Higher values:

- increase the total 3D effect
- increase separation between foreground, midground, and background
- make depth easier to see
- can increase eye strain
- can reveal more edge artifacts if pushed too high

Lower values:

- create a softer 3D effect
- improve comfort
- reduce artifacts
- are better for long movies or difficult depth maps

Recommended use:

```text
40 to 55 = comfortable movie range
55 to 70 = stronger depth
70 to 85 = showcase or testing
```

If a render looks too flat, raise 3D Strength slightly. If the render feels stretched or uncomfortable, lower it.

---

### Pop-Out

**Pop-Out** controls how much the foreground and subjects are allowed to come toward the viewer.

Higher values:

- make close objects feel more forward
- increase subject presence
- create a more dramatic 3D look
- can increase cutout or edge artifacts

Lower values:

- keep the subject closer to the screen plane
- reduce eye strain
- are safer for faces and close-up shots

Use Pop-Out carefully. Strong 3D does not only come from pop-out. A good 3D conversion also needs background depth, midground layering, and a stable screen plane.

---

### Screen Depth

**Screen Depth** controls where the scene sits relative to the screen.

Simple meaning:

```text
Lower value = scene feels closer / more forward
Middle value = neutral
Higher value = scene feels deeper behind the screen
```

Use Screen Depth when:

- the whole scene feels too close
- the whole scene feels too far back
- the background needs to sit deeper
- the subject feels like it is floating too far forward
- the screen plane needs to feel more comfortable

Recommended use:

```text
45 to 55 = neutral
55 to 65 = deeper background feel
35 to 45 = more forward pop-out feel
```

Small changes can make a big difference, so adjust slowly and preview several frames.

---

### Depth Comfort

**Depth Comfort** makes the stereo effect safer.

Higher values:

- reduce extreme parallax
- reduce eye strain
- stabilize the viewing experience
- help longer movies feel more comfortable

Lower values:

- allow stronger 3D
- allow more dramatic depth
- can increase discomfort if pushed too low

Recommended use:

```text
70 to 85 = full movie comfort
55 to 70 = balanced
40 to 55 = stronger test clips
```

For VR or headset viewing, use higher comfort settings.

---

### Subject Stability

**Subject Stability** controls how strongly the main subject is protected from drifting with the background.

Higher values:

- keep faces and bodies more stable
- reduce subject wobble
- prevent the subject from flattening into the background
- are good for dialogue scenes and close-ups

Lower values:

- allow stronger subject depth movement
- can create more pop-out
- may be less stable

Recommended use:

```text
60 to 85 = faces and dialogue
45 to 65 = general scenes
25 to 45 = stronger pop tests
```

If pushing the background deeper makes the subject fall flat with it, increase Subject Stability or use the advanced Subject Plane Lock controls.

---

### Recommended Beginner Workflow

1. Load the source video and depth map.
2. Click **Load Preview Sources**.
3. Select a 3D Style.
4. Preview a frame in **Red-Blue Anaglyph**.
5. Adjust **3D Strength** until the depth is visible but not uncomfortable.
6. Adjust **Screen Depth** until the background sits where you want it.
7. Adjust **Pop-Out** if the foreground needs more presence.
8. Raise **Subject Stability** if people or main objects feel unstable.
9. Use **Shift Heatmap** to check where the strongest stereo stress is.
10. Render a short clip range before rendering the full video.

---

## Advanced 3D Controls

Advanced 3D Controls are for manual tuning. They give direct access to the stereo renderer.

Use Advanced Controls when:

- the beginner controls are close but not perfect
- a scene has difficult depth
- you need exact preset values
- you want to fix flat backgrounds
- you want to reduce artifacts
- you want to tune subject stability, convergence, edge repair, or layered depth

The advanced controls are grouped into several ideas:

1. **Depth and Parallax Controls**  
   Foreground, midground, background, max shift, and parallax balance.

2. **Screen Plane and Convergence**  
   Screen placement, convergence strength, and dynamic convergence.

3. **Subject Controls**  
   Subject tracking, subject lock, subject plane lock, and subject lock width.

4. **Depth Shaping Controls**  
   Depth pop gamma, pop mid, stretch lo, stretch hi, foreground pop, background push, and curvature.

5. **Artifact Controls**  
   Edge masking, feathering, edge repair quality, floating window, and optional smoothing.

6. **Layered Depth Controls**  
   Cinematic depth sculpt and depth-order warp behavior.

---

## Layered Depth and Background Depth

The newer 3D system improves the way VD3D creates background depth.

Older simple stereo shifting could sometimes produce:

```text
flat background
subject pops out
scene does not feel like a room
pushing depth back makes everything move together
```

The updated system is designed to create:

```text
background sinks behind the screen
subjects stay separated from the background
rooms and environments feel deeper
screen feels more like a window into the scene
```

This is done with two major ideas:

- **Cinematic Depth Sculpt**
- **Depth-Order Forward Warp**

---

### Cinematic Depth Sculpt

Cinematic Depth Sculpt reshapes the depth used for stereo rendering.

It helps:

- push far areas deeper
- hold the subject closer to a stable plane
- create stronger background recession
- prevent the subject and background from flattening together
- improve the window-into-the-scene illusion

This works with:

- Depth Pop Gamma
- BG Push
- Subject Plane Lock
- Subject Plane Lock Width
- FG Pop
- Screen Depth

This is why background depth can now feel stronger without destroying the subject placement.

---

### Depth-Order Forward Warp

Depth-Order Forward Warp is an optional layered warp behavior.

Classic stereo warping samples the image smoothly, which is good for stability. Depth-order warping adds a layered placement pass where depth order matters more.

Conceptually:

```text
far pixels are placed first
near pixels are placed after
near objects can sit over far objects
background can separate more naturally
```

This can make the scene feel less flat and more layered.

Best for:

- rooms
- hallways
- landscapes
- shots with clear foreground and background
- clips where the background feels flat

Use carefully for:

- hair
- fast motion
- thin objects
- heavy edge detail
- noisy depth maps

If artifacts increase, lower the layered warp strength or use stronger edge repair.

---

### Layered Depth Warp Strength

If your version exposes this as a control, it adjusts how much depth-order warp is blended into the classic VD3D warp.

Suggested meaning:

```text
0 = classic VD3D warp only
35 = safe layered depth
50 = balanced layered depth
65 = strong room/background depth
80+ = aggressive testing
```

Recommended starting values:

```text
0.35 to 0.50 = safe tests
0.50 to 0.65 = strong but usable
0.65+ = showcase or difficult flat-background shots
```

If the background looks much better but subject edges get stressed, reduce this first.

---

## Screen Depth and Screen Plane Offset

VD3D has both a beginner screen-depth concept and an advanced screen-plane control.

### Screen Depth

Screen Depth is the beginner control.

It answers the user question:

```text
Should the scene feel closer to me, neutral, or deeper behind the screen?
```

Use it for broad tuning.

### Screen Plane Offset

Screen Plane Offset is the advanced fine-tuning control.

It adjusts the underlying stereo plane where the left and right eye line up.

Use it when:

- the whole scene feels shifted forward
- the whole scene feels shifted backward
- the subject is not sitting comfortably
- the stereo field feels offset
- the scene needs a small comfort adjustment

Screen plane should be adjusted slowly. Small changes can have a noticeable effect.

---

## Subject Stability and Subject Plane Controls

Subject controls are important because stronger background depth can make subjects unstable if not managed correctly.

The goal is:

```text
background can go deeper
subject remains readable and stable
```

---

### Subject Lock

Subject Lock anchors the tracked subject depth.

Higher values:

- stabilize the subject
- reduce subject drift
- improve comfort
- help faces and bodies stay readable

Lower values:

- allow more subject movement
- allow stronger pop-out
- may be less stable

Use higher values for dialogue and close-ups. Use lower values for pop-out tests.

---

### Subject Plane Lock

Subject Plane Lock protects the subject depth band from being pulled too aggressively.

Higher values:

- keep subjects from flattening with the background
- keep faces more comfortable
- reduce subject drift

Lower values:

- allow more depth movement
- create stronger separation
- can be less stable

Recommended starting range:

```text
0.20 to 0.35
```

---

### Subject Plane Lock Width

Subject Plane Lock Width controls how wide the protected subject band is.

Recommended range:

```text
0.10 to 0.14
```

Meaning:

```text
0.08 = tight protection
0.12 = good default
0.16+ = safer, but can flatten more of the scene
```

If too much of the scene feels flat, lower the width. If subjects are unstable, increase it slightly.

---

## Codec Presets

Codec presets help users pick output settings without needing to understand every FFmpeg option.

Use presets when:

- you want a quick test
- you want final output
- you want maximum compatibility
- you want smaller files
- you want a high-quality archive
- hardware encoding is not working

A codec preset may change encoder, quality, speed, container behavior, pixel format, and compatibility settings.

---

### Fast Preview - Quick Test

Use this for short previews and setting tests.

Best for:

- checking 10 to 20 seconds
- testing depth settings
- comparing 3D presets
- checking convergence and edge repair

Expected result:

- faster encoding
- smaller test files
- lower quality than final output

Use this before any long render.

---

### Compatibility Mode - Plays Everywhere

Use this when playback compatibility matters most.

Best for:

- sharing with other users
- basic media players
- TVs
- Discord or web uploads
- avoiding playback problems

Expected result:

- H.264-style compatibility
- safer pixel format
- usually larger than HEVC at similar quality
- broad support

---

### Balanced Final - NVIDIA

Recommended default for most NVIDIA users.

Best for:

- final renders
- good speed
- good quality
- general movie clips

Expected result:

- NVENC hardware acceleration
- balanced quality and file size
- much faster than CPU encoding on NVIDIA systems

---

### High Quality Final - NVIDIA

Use this when final quality matters more than speed.

Best for:

- showcase clips
- demo renders
- final archive-quality outputs
- high-detail scenes

Expected result:

- slower than Balanced Final
- higher quality
- larger files

---

### Small File - HEVC

Use this when file size matters.

Best for:

- sharing large videos
- storing many conversions
- reducing upload size
- keeping good quality at lower bitrate

Expected result:

- H.265 / HEVC compression
- smaller files than H.264
- may not play on older devices

---

### 4K / Full-SBS High Quality - HEVC

Use this for high-resolution stereo output.

Best for:

- 4K renders
- Full-SBS output
- VR viewing
- high-quality movie clips
- modern players and headsets

Expected result:

- better compression for high resolution
- high quality
- larger files than small-file presets
- more demanding playback than H.264

---

### Depth Map Output - Fast Safe

Use this for depth videos or utility output.

Best for:

- grayscale depth videos
- intermediate depth files
- quick depth pipeline outputs
- avoiding unnecessary final-video settings

Expected result:

- fast and reliable output
- safe settings for depth video
- not meant as the main final movie preset

---

### Archive Master - Large File

Use this when storage size is less important than preserving quality.

Best for:

- master files
- later editing
- future re-encoding
- avoiding repeated quality loss

Expected result:

- very high quality
- large files
- slower encode
- better source for later exports

---

### CPU Compatibility - Slow

Use this when GPU encoding fails or is unavailable.

Best for:

- systems without NVENC / AMF / QSV
- encoder troubleshooting
- maximum fallback compatibility

Expected result:

- slower render
- reliable CPU encode
- useful fallback when hardware encoders fail

---

### Custom

Use Custom when you want manual control.

Best for users who understand:

- codecs
- CRF / CQ
- bitrate
- presets
- pixel formats
- containers
- HDR settings

---

### Which Codec Preset Should I Use?

| Goal | Recommended Preset |
|---|---|
| First test render | Fast Preview - Quick Test |
| Most NVIDIA final renders | Balanced Final - NVIDIA |
| Best NVIDIA quality | High Quality Final - NVIDIA |
| Smaller file size | Small File - HEVC |
| 4K or Full-SBS output | 4K / Full-SBS High Quality - HEVC |
| Sharing with most devices | Compatibility Mode - Plays Everywhere |
| Depth video output | Depth Map Output - Fast Safe |
| High-quality master file | Archive Master - Large File |
| Hardware encoding fails | CPU Compatibility - Slow |
| Manual encoder tuning | Custom |

---

### Codec Troubleshooting

If a render fails with a hardware encoder:

1. Try **Compatibility Mode - Plays Everywhere**.
2. Try **CPU Compatibility - Slow**.
3. Update GPU drivers.
4. Try H.264 before HEVC or AV1.
5. Try MKV if MP4 fails.
6. Confirm your GPU supports the selected encoder.

If playback fails:

1. Try H.264 Compatibility Mode.
2. Avoid AV1 unless your player supports it.
3. Use MP4 for broad compatibility.
4. Use MKV for more flexible advanced output.

---

## VR180 Output Settings

VR180 output is used when creating stereoscopic video for VR headsets.

VisionDepth3D supports VR180-style output modes such as:

- **VR180 Equirect Top-Bottom**
- **VR180 Equirect Side-by-Side**

Use these modes when you want the final render to be viewed as immersive VR180 content instead of a normal flat SBS video.

### VR180 Equirect Presets

VR180 equirect presets control the final per-eye equirectangular output size.

Common presets include:

| Preset | Use Case |
|---|---|
| 2048×1024 per eye | Faster tests, lower VRAM use |
| 3072×1536 per eye | Balanced VR output |
| 3840×1920 per eye | High-quality VR output |
| 4096×2048 per eye | Very high quality |
| 5760×2880 per eye | Heavy showcase renders |

Higher VR180 resolutions create sharper headset output, but require more GPU memory, longer render time, and larger files.

### VR180 Flat Working Presets

Flat working presets control the internal flat stereo render before it is warped into VR180.

Common working presets include:

| Preset | Use Case |
|---|---|
| 1280×720 | Fast tests |
| 1920×1080 | Balanced quality |
| 2560×1440 | Cleaner source before VR warp |

If the VR180 output looks soft, increase the flat working size first. If rendering is too slow, reduce the flat working size or choose a lower VR180 equirect preset.

### VR180 Tips

- Start with a lower preset for testing.
- Use short clip ranges before full VR180 renders.
- Keep Max Pixel Shift moderate for headset comfort.
- Avoid extreme pop-out close to the frame edges.
- Use Floating Window and Edge Repair when strong foreground objects approach the side borders.

---

### 4. Configure Processing Options

Use **Processing Options** to control stereo stability, edge behavior, and render safety.

Common recommended options:

- **Preserve Original Aspect Ratio**  
  Keeps the source framing from being stretched.

- **Auto Crop Black Bars**  
  Detects and removes letterbox bars before stereo generation when appropriate.

- **Stabilize Screen Plane**  
  Helps keep the subject or dominant depth region closer to the screen plane.

- **Skip Blank / White Frames**  
  Avoids rendering empty frames that can appear in some sources.

- **Enable Edge Masking**  
  Reduces harsh stereo artifacts around strong depth edges.

- **Enable Feathering**  
  Softens transitions between shifted regions.

- **Edge Repair Quality**  
  Lets you choose Off, Fast, Balanced, High, or Showcase edge repair depending on whether you want faster rendering or cleaner disocclusion repair.

- **Enable Dynamic Convergence**  
  Smooths convergence changes across scenes.

- **Enable Floating Window**  
  Adds cinematic edge protection when strong pop-out approaches frame borders.

- **Clip Range**  
  Allows short test renders before committing to a full video.

These settings directly affect visual comfort, stereo stability, and artifact control.

---

## Open Preview for Testing

Click **Load Preview Sources** to open the source video and depth map for preview.

The preview system lets you:

- scrub through frames
- test different preview modes
- inspect stereo direction
- check depth alignment
- tune shift values before rendering
- save preview images for comparison

Testing preview frames is strongly recommended before starting a full render.

---

## Preview Modes

### Red-Blue Anaglyph

Use this for a quick stereo check.

It helps you inspect:

- stereo direction
- subject placement
- edge ghosting
- convergence comfort
- whether the scene feels pushed forward or backward

If the image feels inverted, uncomfortable, or backwards, check depth inversion, shift direction, and eye order.

---

### Passive Interlaced

Useful for displays that support interlaced stereo or for checking alternating-line stereo separation.

---

### HSBS

Shows a half side-by-side stereo preview.

Use this when checking headset-style or SBS-based output.

---

### Shift Heatmap

The Shift Heatmap visualizes stereo displacement.

Use it to check:

- whether foreground, midground, and background are separating correctly
- whether foreground is receiving enough negative shift
- whether background is receiving positive push
- whether extreme shift is being clamped

This is one of the best modes for tuning the new method.

---

### Shift Heatmap (Abs)

Shows the strength of displacement without focusing on direction.

Use this to see where the strongest stereo stress exists.

---

### Shift Heatmap (Clipped ±5px)

Shows a clipped range of shift values to make smaller displacement differences easier to inspect.

Useful when tuning subtle scenes.

---

### Overlay Arrows

Displays shift direction visually.

Use this to confirm whether near and far regions are moving in the expected directions.

---

### Left-Right Diff

Shows differences between the two generated eye views.

Useful for spotting:

- excessive disparity
- edge tearing
- ghosting
- overly aggressive stereo separation

---

### Feather Mask

Shows the feathering mask used to soften depth transitions.

Useful when diagnosing harsh cutout edges.

---

### Feather Blend

Shows the blended feathering result.

Useful when checking whether stereo transitions are too sharp or too soft.

---

## Depth and Parallax Controls

### Foreground Shift

Controls how strongly near objects are pulled toward the viewer.

In the current VisionDepth3D Method, foreground pop is usually created with **negative values**.

More negative values:

- increase foreground pop-out
- pull close subjects and objects forward
- create stronger stereo separation

Less negative values:

- create a more subtle 3D effect
- reduce eye strain
- keep subjects closer to the screen plane

Recommended range:

| Style | Range |
|---|---|
| Natural | `-5.0` to `-7.0` |
| Strong | `-8.0` to `-10.0` |
| Aggressive | `-10.0` to `-12.0` |

If the foreground looks too flat, make the value more negative.

If the foreground feels uncomfortable, stretched, or too separated, move it closer to zero.

---

### Midground Shift

Controls the depth position of objects between foreground and background.

In the current method, midground shift is usually **slightly negative or near zero**.

Typical values:

| Style | Value |
|---|---:|
| Subtle mid-depth | `-0.5` |
| Natural layering | `-0.8` to `-1.2` |
| Strong layering | `-1.5` to `-2.0` |
| Neutral screen-plane feel | `0.0` |

Midground shift helps connect the foreground and background so the scene does not feel like only two flat layers.

If the image looks like cardboard cutouts, reduce the gap between foreground and midground values.

Example:

```text
Too separated:
FG -12.0 / MG 0.0 / BG +5.0

More natural:
FG -6.0 / MG -0.8 / BG +2.2
```

---

### Background Shift

Controls how far distant scene elements are pushed behind the screen plane.

In the current VisionDepth3D Method, background depth is usually created with **positive values**.

Higher positive values:

- push backgrounds deeper
- increase cinematic depth scale
- make environments feel larger

Lower positive values:

- keep backgrounds closer
- reduce eye strain
- create a more natural stereo effect

Recommended range:

| Style | Range |
|---|---|
| Subtle | `+1.0` to `+2.0` |
| Natural | `+2.0` to `+3.0` |
| Strong | `+3.5` to `+5.0` |

Avoid pushing the background too far if the foreground is already very negative, because the scene can start to look stretched, separated, or uncomfortable.

---

### Convergence Strength

Controls how strongly the convergence plane is adjusted.

Higher values:

- move the perceived focus plane more aggressively
- increase perceived depth movement between shots
- can make scene transitions more dramatic

Lower values:

- produce more stable convergence
- reduce eye strain
- keep long-form content more comfortable

Use smaller values for full-length videos.

For aggressive pop-out testing, reduce convergence strength or disable dynamic convergence temporarily so the pipeline does not pull the foreground back toward the screen plane.

---

### Screen Plane Offset (formerly Zero Parallax)

Screen Plane Offset replaces the older **Zero Parallax** naming.

This control fine-tunes where the stereo screen plane sits. The screen plane is the depth position where the left and right eyes line up with no perceived pop-out or recession.

Use this when:

- the whole scene feels too far forward
- the whole scene feels pushed too far backward
- subjects are not sitting where expected
- the stereo field feels offset
- you want to move the perceived screen surface without fully changing FG / MG / BG shift values

Small changes can have a noticeable effect. For comfort, adjust this slowly and preview several frames before rendering.

If the scene already feels comfortable, leave Screen Plane Offset near the default value.

---

### Parallax Balance

Controls the overall stereo balance between foreground and background.

Higher values:

- increase overall stereo strength
- make foreground and background separation stronger
- may increase eye strain

Lower values:

- create a gentler stereo effect
- improve comfort
- reduce extreme parallax

Recommended starting range:

```text
0.70 to 1.00
```

For comfort, start around `0.70`.

For stronger showcase depth, try `0.90` to `1.05`.

---

### Max Pixel Shift (%)

Limits the maximum allowed parallax displacement.

This is a safety clamp.

Lower values:

- reduce extreme stereo separation
- improve comfort
- help prevent eye strain
- may reduce pop-out

Higher values:

- allow stronger depth
- allow more foreground pop
- can increase artifacts or discomfort

Recommended range:

```text
0.020 to 0.050
```

For subtle or VR-friendly output, use lower values.

For stronger pop tests, temporarily try higher values such as `0.050`, then reduce if the image becomes uncomfortable.

---

### Stereo Scaling (IPD)

Controls overall stereo separation intensity, similar to virtual eye distance.

Higher values:

- stronger 3D effect
- larger parallax
- more separation between eyes

Lower values:

- more comfortable viewing
- softer depth
- less eye strain

Use this as a global depth strength control after your FG / MG / BG balance feels correct.

---

### Sharpness Factor

Enhances perceived edge clarity in the stereo output.

Higher values:

- make depth edges look crisper
- emphasize fine detail
- may also emphasize halos or edge artifacts

Lower values:

- produce softer transitions
- reduce harsh edge behavior

Use moderately.

---

### Depth of Field Strength

Applies optional focus blur based on depth.

This can add cinematic realism, but should be used lightly.

Too much DOF can make the stereo output feel artificial or reduce depth readability.

---

## Depth Shaping: Pop and Subject Controls

The current VisionDepth3D Method does not rely only on direct shift amounts.

It uses a separate shaped depth representation for stereo design. This lets the renderer tune how near, mid, and far regions are emphasized without corrupting the underlying subject-tracking depth.

Depth shaping controls affect how the depth map is redistributed before the near / mid / far weighting system builds the final stereo shift field.

---

### Depth Pop Gamma

Controls how aggressively depth values are reshaped around the stereo midpoint.

Lower values:

- increase near/mid separation
- create stronger perceived pop
- can make scenes more dramatic

Higher values:

- soften the depth curve
- reduce cutout-like separation
- can make live or difficult scenes more natural

Recommended range:

```text
0.75 to 1.15
```

For stronger pop, try `0.75` to `0.90`.

For a more natural or less cardboard look, try `1.05` to `1.20`.

---

### Pop Mid

Controls where the shaping curve focuses its strongest separation.

Lower values:

- emphasize foreground depth
- help subjects stand forward
- increase near-object separation

Higher values:

- shift emphasis toward midground and background
- help environments feel deeper
- reduce aggressive foreground pop

Recommended starting value:

```text
0.45 to 0.50
```

---

### Stretch Lo

Controls how the near-depth range is stretched.

Lower values:

- keep foreground tighter
- reduce over-expansion of near subjects

Higher values:

- expand foreground separation
- can make near objects feel stronger

Recommended starting range:

```text
0.02 to 0.06
```

---

### Stretch Hi

Controls how the far-depth range is stretched.

Lower values:

- keep background closer
- reduce background exaggeration

Higher values:

- push distant elements farther back
- increase scene scale

Recommended starting range:

```text
0.94 to 0.98
```

---

### FG Pop ×

Multiplies foreground depth strength after curve shaping.

Higher values:

- exaggerate subject separation
- increase near-object presence
- can create stronger pop-out

Lower values:

- keep subjects more natural
- reduce sticker-like foreground separation

Recommended range:

```text
1.00 to 1.45
```

For aggressive testing, values around `1.50` to `1.60` may be useful, but should be reduced for final comfort.

---

### BG Push ×

Multiplies background depth recession after curve shaping.

Higher values:

- push environments deeper
- increase cinematic scale

Lower values:

- keep backgrounds closer
- reduce excessive depth spread
- help prevent cardboard separation

Recommended range:

```text
0.85 to 1.15
```

---

### Foreground Curvature

Foreground Curvature adds a subtle rounded depth shape to near foreground regions.

This helps foreground subjects feel less flat by gently pushing the center of near objects forward while keeping the edges softer. It is useful for faces, bodies, and larger foreground objects that can otherwise look like flat cardboard cutouts.

Higher values:

- make foreground subjects feel more rounded
- increase perceived volume
- can improve face/body depth
- may exaggerate foreground shape if pushed too far

Lower values:

- keep the original depth map closer to unchanged
- reduce artificial rounding
- are safer for difficult depth maps or thin objects

Recommended use:

```text
Low to moderate values for final renders
Higher values only for testing or stylized depth
```

Foreground Curvature works best when the depth map already has clear foreground separation. It should be used as a shape enhancement, not as a replacement for a good depth map.

---

### Subject Lock

Controls how strongly the pipeline anchors the detected subject depth.

Higher values:

- keep subjects stable
- reduce subject drift
- improve comfort
- may reduce pop-out if too strong

Lower values:

- allow more foreground movement
- can help pop-out testing
- may increase depth instability

Recommended range:

```text
0.00 to 0.25
```

For strong pop-out testing, keep Subject Lock very low.

For long-form viewing, use light subject locking for stability.

---

## Pop-Out vs Depth Layering

Strong 3D does not come only from increasing shift values.

VisionDepth3D separates stereo design into multiple stages:

- normalized depth
- tracked subject depth
- shaped disparity depth
- near / mid / far weighting
- subject-aware screen plane
- dynamic convergence
- edge-aware repair
- floating-window safety

Because of this, pop-out is controlled by more than Foreground Shift alone.

If foreground objects do not pop forward enough, check:

- Foreground Shift is negative enough
- Max Pixel Shift is not too low
- Parallax Balance is not too low
- Subject Lock is not anchoring the subject too strongly
- Dynamic Convergence is not pulling the scene back to the screen plane
- Floating Window is not limiting aggressive pop near frame edges
- Edge Masking is not suppressing too much shift around the foreground
- The depth map is not inverted
- The depth map has enough near-depth contrast

### Stronger pop-out test preset

Use this only for testing, not as a final comfort preset:

```text
Foreground Shift: -10.0
Midground Shift:  -1.0
Background Shift: +2.5

Max Pixel Shift: 0.050
Parallax Balance: 1.00
Subject Lock: 0.00 to 0.05
Floating Window: Off for testing
Dynamic Convergence: Off for testing
Edge Masking: Off for testing
Feathering: Off for testing
```

Once the pop direction is confirmed, re-enable comfort and repair settings for final renders.

---

## Optional Advanced Controls

### Floating Window

Adds cinematic edge protection to prevent objects from breaking the screen border.

This is useful when strong foreground pop approaches the left or right frame edge.

Benefits:

- reduces window violations
- improves viewing comfort
- protects aggressive stereo shots
- makes the render feel more professionally composed

For strong pop-out testing, disable Floating Window temporarily. For final renders, re-enable it if edge violations appear.

---

### Dynamic Convergence

Automatically adjusts convergence based on the tracked subject path and smooths transitions over time.

When enabled:

- tracks subject depth more naturally
- smooths frame-to-frame convergence
- reduces sudden depth jumps
- improves comfort for full-length content

Recommended for long renders.

For aggressive pop-out testing, disable Dynamic Convergence temporarily to make sure it is not pulling the foreground back toward the screen plane.

---

### Stabilize Zero-Parallax

Keeps the screen plane aligned with the dominant or tracked depth range.

When enabled:

- prevents depth drift
- keeps subjects more stable
- reduces eye strain during scene changes

This can improve comfort, but high subject locking can reduce strong pop-out.

---

### Edge Repair Quality

Edge Repair Quality controls how much disocclusion and edge cleanup the 3D Generator applies around shifted stereo edges.

This dropdown lets users balance render speed against cleaner edges.

| Mode | Behavior |
|---|---|
| Off | Fastest. No extra edge repair. More edge artifacts may appear. |
| Fast | Lighter repair for faster rendering. Good for tests or slower GPUs. |
| Balanced | Recommended default. Good speed and quality balance. |
| High | Stronger repair with slower render speed. Better for final clips. |
| Showcase | Strongest edge cleanup. Slowest mode. Best for demos or difficult scenes. |

Use **Fast** or **Balanced** for most full-length renders. Use **High** or **Showcase** when edge artifacts are very noticeable and render speed is less important.

If you are testing maximum FPS, set Edge Repair Quality to **Off** or **Fast**.

---

### Edge-Aware Masking

Suppresses unstable stereo shift near hard depth edges such as:

- hair
- fingers
- shoulders
- thin foreground objects
- high-contrast silhouettes

Benefits:

- reduces halos
- reduces edge tearing
- improves contour cleanliness

If the scene lacks pop, test with Edge Masking off temporarily to see whether it is suppressing too much foreground shift. Re-enable it for final renders if edge artifacts appear.

---

### Feathering

Softens transitions between shifted regions.

Benefits:

- smoother depth blending
- fewer harsh stereo edges
- less cutout-like transitions

Too much feathering may make the scene feel softer or reduce perceived sharpness.

---

### Disable Shift EMA

Disables temporal shift smoothing for debugging.

Use this only for testing.

When enabled:

- raw shift changes are easier to inspect
- pop direction can be tested more directly
- motion may look less stable

For final renders, shift smoothing is usually recommended.

---

### Depth of Field Simulation

Applies subtle focus blur based on depth.

Use sparingly.

This can help cinematic presentation, but too much blur can make depth harder to read.

---

## Clip Range Rendering

Set start and end timecodes to render only a portion of the video.

Useful for:

- testing settings quickly
- tuning difficult scenes
- checking pop-out behavior
- checking edge artifacts
- avoiding long re-renders

Recommended before full-length renders.

Example:

```text
Start: 00:01:20
End:   00:01:35
```

---

## Recommended First-Time Workflow

1. Load source video and depth map video.
2. Confirm the original resolution and aspect ratio.
3. Choose output format and codec.
4. Open preview sources.
5. Test a few frames in Anaglyph and Shift Heatmap mode.
6. Start with the new negative-foreground shift convention.
7. Tune FG / MG / BG shift.
8. Tune Max Pixel Shift and Parallax Balance.
9. Adjust Depth Pop Gamma and FG Pop × if the scene feels flat.
10. Render a short clip range.
11. Re-enable comfort tools such as Dynamic Convergence, Edge Masking, Feathering, and Floating Window.
12. Render the full video.

---

## Recommended Starting Presets

### Natural Comfortable 3D

```text
Foreground Shift: -6.0
Midground Shift:  -0.8
Background Shift: +2.2

Max Pixel Shift: 0.022
Parallax Balance: 0.70
Depth Pop Gamma: 1.05
FG Pop ×: 1.00
BG Push ×: 0.95
Subject Lock: 0.15
Dynamic Convergence: On
Edge Masking: On
Feathering: On
Floating Window: On if needed
```

Best for:

- full-length movies
- dialogue scenes
- comfortable VR viewing
- natural depth layering

---

### Strong Showcase 3D

```text
Foreground Shift: -8.5
Midground Shift:  -1.2
Background Shift: +3.5

Max Pixel Shift: 0.035
Parallax Balance: 0.90
Depth Pop Gamma: 0.85
FG Pop ×: 1.20
BG Push ×: 1.05
Subject Lock: 0.10
Dynamic Convergence: On
Edge Masking: On
Feathering: On
Floating Window: On if needed
```

Best for:

- demo clips
- trailers
- scenes with clear subjects
- stronger depth presentation

---

### Aggressive Pop-Out Test

```text
Foreground Shift: -10.0 to -12.0
Midground Shift:  -2.0
Background Shift: +4.0 to +5.0

Max Pixel Shift: 0.050
Parallax Balance: 1.00
Depth Pop Gamma: 0.75
FG Pop ×: 1.45
BG Push ×: 0.85
Subject Lock: 0.00 to 0.05
Dynamic Convergence: Off for testing
Edge Masking: Off for testing
Feathering: Off for testing
Floating Window: Off for testing
```

Best for:

- confirming pop direction
- diagnosing whether the foreground can move forward
- testing depth map strength
- checking if stabilization is suppressing pop

Not recommended as a final full-length preset without comfort adjustments.

---

## Troubleshooting

### The scene looks inverted

Try checking:

- depth inversion
- eye order
- preview mode
- whether the depth map uses white-near or black-near convention
- whether old presets are being reused incorrectly

---

### The scene has depth but no pop-out

Check:

- Foreground Shift is negative enough
- Max Pixel Shift is high enough
- Parallax Balance is not too low
- Subject Lock is not too strong
- Dynamic Convergence is not over-stabilizing the subject
- Floating Window is not suppressing aggressive foreground depth
- the depth map has enough near-depth contrast

Try the Aggressive Pop-Out Test preset to confirm whether the renderer can produce forward disparity.

---

### The scene looks like cardboard cutouts

Try:

- reducing Foreground Shift strength
- moving Midground Shift closer to Foreground Shift
- increasing Depth Pop Gamma above `1.00`
- lowering FG Pop ×
- lowering Subject Lock
- enabling Feathering
- using a smoother depth map
- blending depth maps in Depth Blender

Example adjustment:

```text
From:
FG -12.0 / MG 0.0 / BG +5.0

To:
FG -6.0 / MG -0.8 / BG +2.2
```

---

### There are halos around subjects

Try:

- enabling Edge Masking
- enabling Feathering
- lowering Max Pixel Shift
- reducing Foreground Shift strength
- checking depth map edge quality
- using Depth Blender to smooth or refine the depth map

---

### There is too much eye strain

Try:

- reducing Max Pixel Shift
- reducing Parallax Balance
- moving Foreground Shift closer to zero
- lowering Background Shift
- enabling Dynamic Convergence
- enabling Floating Window
- using a shorter clip range for testing

---

### The background is too flat

Try:

- increasing Background Shift
- increasing BG Push ×
- lowering Pop Mid slightly
- increasing Stretch Hi
- increasing Parallax Balance carefully

---

### The foreground is too flat

Try:

- making Foreground Shift more negative
- increasing FG Pop ×
- lowering Depth Pop Gamma
- lowering Pop Mid slightly
- increasing Max Pixel Shift
- lowering Subject Lock

---

## Final Notes

The current VisionDepth3D Method is designed around a full stereo pipeline rather than simple positive/negative pixel shifting.

Foreground, midground, and background controls now work together with:

- depth normalization
- pop-control depth shaping
- structured near / mid / far weighting
- subject-aware screen plane
- dynamic convergence
- edge-aware shift limiting
- contour-safe repair
- floating-window control
- temporal stabilization

For the best results, start from the new presets, preview several frames, render short clip ranges, and tune gradually.

---

## VD3D Live (Real-Time 2D-to-3D)

VD3D Live is a real-time 2D-to-3D pipeline designed for live sources like:
- Screen capture (desktop / games / video players)
- Cameras and capture cards

It captures frames, runs a Depth Anything model, then generates a stereoscopic SBS output using the Pixel Shift CUDA pipeline.

You can use it for:
- Live 3D preview while watching content
- Real-time depth tuning
- External output to other apps (HTTP stream or Virtual Camera)

---

## Quick Start: Live Screen 3D

### 1) Open VD3D Live
Launch **VD3D Live – GUI** from inside VD3D (or run the live script if you use it standalone).

---

### 2) Set Capture Source to Screen
In the **Capture** section:

- **Source:** `screen:1` (primary monitor)  
  - `screen:2` for a second monitor  
  - `screen:0` captures the full bounding box across all monitors (not recommended unless you need it)

- **Capture FPS:** `30` is a good default for stability  
  - Raise if you want smoother motion and your GPU can keep up

Tip:
If you are screen capturing the same monitor the preview is on, you can create a feedback loop. Use one of these:
- Put preview on a different monitor  
- Enable **Mask preview region in screen capture**  
- Or disable preview and use external output instead

---

### 3) Configure the Depth Model
In **Depth / Model**:

- **Model ID:** choose a Depth model from hugging face or use one already in input field  
  Example: `depth-anything/Depth-Anything-V2-Large-hf`

- **Use FP16 (if CUDA):** enable this on NVIDIA GPUs  
  - Reduces VRAM usage and improves speed

- **Infer W / Infer H:** depth inference resolution  
  Example: `320 × 180` for speed  
  - Higher values = better depth detail, slower performance

- **Depth FPS:** how often depth is updated  
  Example: `5.0`  
  - Lower = faster overall performance  
  - Higher = more responsive depth changes

Optional:
- **Smooth (EMA + median):** reduces depth jitter and flicker  
- **EMA α:** smoothing strength (higher = smoother but more lag)

---

### 4) Enable SBS 3D Output
In **3D / Pixel Shift**:

- Enable **Enable SBS 3D**
- Set your shifts:
  - **FG shift** (foreground pop)
  - **MG shift** (mid depth layering)
  - **BG shift** (background push)

Typical starter values using the current VD3D shift direction:
- FG shift: `-5 to -10`
- MG shift: `-0.5 to -2`
- BG shift: `+2 to +5`

These are live controls, so you can tune while watching.

---

### 5) Preview and Start
In **Preview / Output**:

- Enable **Show preview window** if you want an on-screen preview
- If your source is screen capture:
  - Leave **Force preview (screen src)** OFF unless you know what you are doing
  - Use **Mask preview region in screen capture** if the preview is on the same monitor you’re capturing

Then press **Start**.

---

## Live Preview Controls (Hotkeys)

When the preview window is visible:

- `m` cycles view mode  
  Passthrough → Depth → 3D-SBS

- `f` toggles fullscreen

- `q` or `ESC` quits

---

## What Each Capture Setting Does

### Backend (device capture only)
Controls the OpenCV capture backend:
- `msmf` is usually best on Windows
- `dshow` can work better for some capture cards
- `ffmpeg` can help with certain device formats
If a device won’t open or drops frames, try changing this first.

### Device index (device capture only)
Selects which camera or capture device you are using.
If you have multiple devices, try index `0`, then `1`, then `2`.

### FOURCC (device capture only)
Requests a specific camera/capture format (example: `YUY2`).
Only use this if you know your device needs it.

### Force BGR swap / Disable auto swap
Some capture devices output color channels differently.
- **Force BGR swap** manually flips channels (fixes weird colors)
- **Disable auto swap** prevents automatic guessing
If your colors look wrong, toggle these.

---

## External Output Options

### HTTP Stream (MJPEG)
Lets you view the live output in another app over your network.

1. Set **HTTP stream (host:port)**  
   Example: `127.0.0.1:8080`

2. Click **Start**

Then open:
- `http://127.0.0.1:8080/video.mjpg`

Use this when:
- You want external viewing without a local preview window
- You want to capture the stream in another tool

---

### Virtual Camera Output
Outputs the live SBS feed as a virtual webcam device (requires `pyvirtualcam`).

1. Enable **Virtual camera**
2. Set **VCam FPS** (example: `30`)
3. Click **Start**

Use this when:
- You want to feed live SBS output into OBS, VR tools, or other software that accepts webcams
- You want an output pipeline without relying on the preview window

Note:
The virtual camera resolution matches the current output frame size.

---

## Audio Device (Optional Monitor)
The **Audio device** field can start an audio monitor using ffplay.

- **Audio device:** your system audio capture name (Windows uses DirectShow naming)
- **Audio delay ms:** applies a delay if your video processing introduces lag

Use this when:
- You need audio while viewing live output
- You need to compensate for processing latency

---

## Recommended Settings for Screen Live 3D

Comfort + stability preset:
- Capture FPS: `30`
- Infer: `320 × 180`
- Depth FPS: `5`
- Smooth: ON
- EMA α: `0.35`
- FG/MG/BG: `-6 / -0.8 / +2.2`

If you need more depth detail:
- Raise Infer size first (example: `512 × 288`)
- Keep Depth FPS modest to avoid GPU overload

---

## Troubleshooting

### Preview feedback loop (infinite recursion)
If you see repeated “screen within screen” or performance tanks:
- Disable preview and use HTTP stream / Virtual camera
- Or enable **Mask preview region in screen capture**
- Or move preview to a different monitor than the one being captured

### Black screen or no frames
- For screen capture: make sure `mss` is installed
- For device capture: try a different backend (`msmf` ↔ `dshow`) and check device index

### Low FPS
- Lower Infer resolution
- Lower Depth FPS
- Turn Smooth OFF
- Reduce shift strength slightly
- Make sure FP16 is enabled on CUDA

---

## Recommended Workflow Summary

For best quality and efficiency, follow this proven VD3D workflow:

1. Generate depth maps in the **Depth Estimation Tab**  
2. (Optional) Blend two depth sources in the **Depth Blender Tab**  
3. Load source + depth video in the **3D Generator Tab**  
4. Choose a 3D Assistant preset or open Advanced 3D Controls  
5. Configure Encoder Settings or choose a Codec Preset  
6. Open Live Preview and tune depth using Shift Heatmap + Anaglyph  
7. Test short Clip Range  
8. Render final full-length 3D video  

This approach prevents wasted long renders and ensures optimal depth quality.

---

## Best Practices for High-Quality 3D

- Start with built-in presets and refine from there  
- Use **Shift Heatmap view** to keep parallax within comfortable ranges  
- Increase depth gradually rather than maxing sliders  
- Enable **Dynamic Convergence** for long content  
- Use **Edge Masking + Feathering** for clean depth edges  
- Test short clip ranges before full renders  
- Avoid extreme pixel shift values (eye strain risk)  

Balanced depth always looks more cinematic than aggressive depth.

---

# Hardware / Backend Support

VisionDepth3D is designed for GPU acceleration, but supported features depend on your hardware and installed backend.

---

## NVIDIA CUDA, Recommended

NVIDIA CUDA is the recommended setup for VisionDepth3D.

Best for:

- Depth estimation
- 3D stereo rendering
- Live 3D preview
- RIFE interpolation
- Real-ESRGAN upscaling
- NVENC video encoding

Recommended install path:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Use the official PyTorch install selector if your system needs a different CUDA version:

```text
https://pytorch.org/get-started/locally/
```

---

## AMD / Intel on Windows, DirectML

AMD and Intel GPU users on Windows can use DirectML through:

```bash
pip install torch-directml
```

DirectML can provide GPU acceleration on supported AMD Radeon, Intel Arc, and some integrated GPUs.

Notes:

- DirectML is usually slower than NVIDIA CUDA.
- Some models or operations may fall back to CPU.
- If DirectML causes issues, use CPU mode as a fallback.
- Do not install CUDA PyTorch for AMD GPUs on Windows.

---

## AMD on Linux, ROCm

AMD users on Linux may be able to use ROCm if their GPU and driver stack are supported.

ROCm support depends heavily on:

- GPU model
- Linux distribution
- installed ROCm version
- PyTorch ROCm compatibility

Use the official PyTorch install selector for ROCm setup:

```text
https://pytorch.org/get-started/locally/
```

---

## CPU Fallback

VisionDepth3D can fall back to CPU when no supported GPU backend is available.

CPU mode works, but it is much slower for:

- Depth map generation
- Video processing
- Live 3D
- Upscaling
- Frame interpolation
- Full 3D renders

CPU mode is best for testing, small images, or fallback compatibility.

---

## FFmpeg Hardware Encoders

VisionDepth3D can use different FFmpeg encoders depending on your GPU.

| GPU / Backend | Encoder Options |
|---|---|
| NVIDIA | `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` |
| AMD | `h264_amf`, `hevc_amf`, `av1_amf` |
| Intel | `h264_qsv`, `hevc_qsv`, `av1_qsv` |
| CPU | `libx264`, `libx265`, `libaom-av1`, `libsvtav1` |

If a hardware encoder fails, try a CPU encoder for compatibility.

---

## Recommended Setup by User Type

| User Type | Recommended Backend |
|---|---|
| NVIDIA GPU user | CUDA PyTorch + NVENC |
| AMD GPU on Windows | DirectML + AMF encoder |
| Intel GPU on Windows | DirectML + QSV encoder |
| AMD GPU on Linux | ROCm if supported |
| No supported GPU | CPU PyTorch |

---

## Backend Troubleshooting

## CUDA is not detected

Try:

- Check NVIDIA driver installation.
- Run `nvidia-smi`.
- Reinstall CUDA PyTorch using the official PyTorch selector.
- Make sure `torch`, `torchvision`, and `torchaudio` use matching CUDA builds.
- Restart VisionDepth3D after reinstalling PyTorch.

---

## DirectML is not detected

Try:

- Install `torch-directml`.
- Update AMD / Intel GPU drivers.
- Confirm you are on Windows.
- Restart VisionDepth3D after installation.
- Use CPU mode if DirectML is unstable.

---

## ROCm is not detected

Try:

- Confirm your AMD GPU supports ROCm.
- Confirm your Linux distribution is supported by ROCm.
- Install the correct PyTorch ROCm build.
- Check that your ROCm driver/runtime is installed correctly.

---

## FFmpeg hardware encoding fails

Try:

- Switch from NVENC / AMF / QSV to CPU encoding.
- Update GPU drivers.
- Use H.264 before trying H.265 or AV1.
- Confirm your GPU supports the selected encoder.
- Try another container such as `.mkv` if `.mp4` fails.

---

## VD3D Live v4.0 Shift Value Update

The VD3D should follow the new v4.0 shift convention.

Older 3D values may have used:

```text
FG shift: 6 to 10
MG shift: 1 to 3
BG shift: -3 to -6
```

For VisionDepth3D v4.0, use the new convention:

```text
FG shift: -5 to -10
MG shift: -0.5 to -2
BG shift: +2 to +5
```

Recommended Live 3D starter preset:

```text
FG/MG/BG: -6 / -0.8 / +2.2
```

Stronger Live 3D test preset:

```text
FG/MG/BG: -8.5 / -1.2 / +3.5
```

If 3D looks inverted, check:

- depth inversion
- eye order
- foreground shift direction
- whether an older preset was loaded
- whether the depth model uses the opposite near/far convention

For comfortable realtime Live 3D, start with softer values and increase strength slowly.

Recommended comfort settings:

```text
Capture FPS: 30
Inference: 384x384 or 518x518
Depth FPS: 4 to 6
Smooth Depth: On
Foreground Shift: -6.0
Midground Shift: -0.8
Background Shift: +2.2
Max Pixel Shift: 0.020 to 0.030
Parallax Balance: 0.70
Depth Pop Gamma: 1.05 to 1.15
Subject Tracking: Off for testing, On for stability
Dynamic Convergence: On
Edge Masking: On
Feathering: Off for speed, On for cleaner edges
Floating Window: Off for testing, On if edge violations appear
```

---

## Performance Optimization Tips

### Depth Estimation

- Start at **512×288** or **704×384** for movies  
- Increase only if depth lacks detail  
- Raise batch size until VRAM limit is reached  

### FPS / Upscale Enhancer

- Use Threaded Pipeline on strong GPUs  
- Use Merged Pipeline for long videos or lower-end systems  

### 3D Generator

- NVENC encoding is much faster on NVIDIA GPUs  
- Moderate Max Pixel Shift improves comfort and speed  
- Avoid excessive feather + masking strength  

---

## Common Issues & Fixes

### Depth Looks Flat

- Use **Deep Background** or **Wide / Deep Scene** in the 3D Assistant  
- Increase **BG Push ×** slightly  
- Enable or raise **Layered Depth Warp** if available  
- Lower **Depth Pop Gamma** slightly for stronger depth separation  
- Increase **Background Shift** carefully  
- Use **Shift Heatmap** to confirm the background is receiving usable shift  
- Check that the depth map actually contains background depth detail  

---

### Seeing Halos or Ghosting

- Enable **Edge-Aware Masking**  
- Increase or Decrease MG Shift to eliminate Edge Tearing 
- Reduce Sharpness Factor  

---

### Jitter Between Scenes

- Enable **Dynamic Convergence**  
- Enable **Stabilize Screen Plane**  
- Reduce Convergence Strength  

---

### Eye Strain or Discomfort

- Lower Max Pixel Shift  
- Reduce Foreground Shift  
- Decrease Stereo Scaling (IPD)  

---

### Slow Performance

- Lower inference resolution  
- Reduce batch size  
- Use NVENC encoder  
- Disable unnecessary preview modes  

---

## When to Use Depth Blending

Use the Depth Blender when:

- Subject edges shimmer or break  
- Background depth is noisy  
- One model looks strong in subjects but weak in environment  

Blending V1 + V2 depth sources often produces the cleanest results.

---

## Support & Updates

For updates, documentation, and new releases:

- GitHub repository (VisionDepth3D)  
- Community feedback and issues welcome  

Regular updates continue improving depth quality, speed, and stability.

---

End of User Manual
