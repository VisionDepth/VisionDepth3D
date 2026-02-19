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

## Table of Contents

1. [Overview](#overview)
2. [FPS / Upscale Enhancer](#fps--upscale-enhancer)
   - [Extract Frames from Video](#1-extract-frames-from-video)
   - [Configure Output Video](#2-configure-output-video)
   - [Set Output Resolution](#3-set-output-resolution)
   - [Set Original FPS](#4-set-original-fps)
   - [Configure FPS Interpolation (RIFE)](#5-configure-fps-interpolation-rife)
   - [Choose Video Codec](#6-choose-video-codec)
   - [ESRGAN Upscaling Settings](#7-esrgan-upscaling-settings)
   - [Choosing a Processing Mode](#choosing-a-processing-mode)

3. [Depth Estimation Tab](#depth-estimation-tab)
   - [Quick Start: Render Your First Depth Map](#quick-start-render-your-first-depth-map)
   - [Adjusting Quality and Performance](#adjusting-quality-and-performance)
   - [Output Formats](#output-formats)

4. [Depth Blender Tab](#depth-blender-tab)
   - [Quick Start](#quick-start)
   - [Blend Parameters](#blend-parameters)
   - [Running a Batch](#running-a-batch)

5. [3D Generator Tab](#3d-generator-tab)
   - [Getting Started](#getting-started-1)
   - [Using Preview Modes for Tuning](#using-preview-modes-for-tuning)
   - [Sliders and Settings](#sliders-and-settings)
   - [Optional Advanced Controls](#optional-advanced-controls)
   - [Clip Range (Optional)](#clip-range-optional)

6. [Audio Tool (Ripper & Attacher)](#audio-tool-audio-ripper--attacher)
   - [What This Tool Can Do](#what-this-tool-can-do)
   - [Step 1 — Add Your Source Videos](#step-1--add-your-source-videos)
   - [Step 2 — Choose Your Operation](#step-2--choose-your-operation)
   - [Audio Source Modes (Attach / Attach+Stitch)](#audio-source-modes-attach--attachstitch)
   - [Audio Offset (Sync Fix)](#audio-offset-sync-fix)
   - [Force Re-encode (Per-clip Attach)](#force-re-encode-per-clip-attach)
   - [Rip Settings (Extract Audio)](#rip-settings-extract-audio)
   - [Final Stitch Settings (Attach+Stitch)](#final-stitch-settings-attachstitch)
   - [Step 3 — Output Settings](#step-3--output-settings)
   - [Preview Button (Recommended)](#preview-button-recommended)
   - [Run Button](#run-button)
   - [Common Workflows](#common-workflows)
   - [Notes](#notes)

7. [VD3D Live (Real-Time 2D-to-3D)](#vd3d-live-real-time-2d-to-3d)
   - [Quick Start: Live Screen 3D](#quick-start-live-screen-3d)
   - [Live Preview Controls (Hotkeys)](#live-preview-controls-hotkeys)
   - [What Each Capture Setting Does](#what-each-capture-setting-does)
   - [External Output Options](#external-output-options)
   - [Audio Device (Optional Monitor)](#audio-device-optional-monitor)
   - [Recommended Settings for Screen Live 3D](#recommended-settings-for-screen-live-3d)
   - [Troubleshooting](#troubleshooting)

8. [Recommended Workflow Summary](#recommended-workflow-summary)
9. [Best Practices for High-Quality 3D](#best-practices-for-high-quality-3d)
10. [Performance Optimization Tips](#performance-optimization-tips)
11. [Common Issues & Fixes](#common-issues--fixes)
12. [When to Use Depth Blending](#when-to-use-depth-blending)
13. [Support & Updates](#support--updates)
14. [End of User Manual](#end-of-user-manual)


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

### Save Frames (Video Mode)

When enabled, VD3D saves individual depth PNG frames in addition to the depth video.

Useful for:
- Manual inspection
- Custom 3D workflows

---

### CPU Offload Mode

Reduces VRAM usage by moving parts of the model to CPU.

- None = fastest, highest VRAM usage
- Sequential = balanced
- Full = lowest VRAM usage, slowest

Only adjust this if you encounter memory limits.

---

### Use float16

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

## 3D Generator Tab

The 3D Generator tab converts a 2D video and its corresponding depth map into a stereoscopic 3D video.

It uses depth-based pixel shifting to create left and right eye views, with advanced control over convergence, parallax, floating windows, and stereo stabilization.

This is the final stage of the VisionDepth3D workflow.

---

## Getting Started

### 1. Load Your Inputs

You must provide:

- **Source Video** (original 2D video)
- **Depth Map Video** (generated from the Depth Engine tab or Depth Blender)

Both videos must match in:
- Resolution  
- Frame count  
- Frame rate  

If these do not match, the stereo render will not align correctly.

---

### 2. Choose Output File

Select your output file path and format (MP4, MKV, etc.).

This is where your final 3D video will be saved.

---

### 3. Configure Encoder Settings

Click **Open Encoder Settings** to configure how the 3D video is packaged.

Here you can set:

- **3D Format**
  - Full Side-by-Side (recommended for VR)
  - Other stereo layouts if needed

- **Aspect Ratio**
  - Classic (4:3)
  - Widescreen (16:9)
  - Or custom formats depending on your source

- **Codec**
  - H.264 or H.265 CPU encoding (most compatible)
  - NVENC GPU encoding (recommended for NVIDIA GPUs)

- **Audio Handling**
  - Keep original audio (recommended)
  - Optional cleanup of intermediate SBS files

- **HDR10 Preservation** (if working with HDR content)

Once configured, close the window to apply the settings.

---

### 4. Configure Processing Options

Click **Open Processing Options** to control how depth is stabilized and refined.

Common recommended options:

- Enable **Dynamic Convergence** for smoother depth transitions  
- Enable **Edge Masking** to reduce halo artifacts  
- Enable **Feathering** for softer depth edges  
- Enable **Stereo Scaling (IPD)** for adaptive depth strength  

Optional tools:

- Auto crop black bars  
- Skip blank frames  
- Floating window (cinematic edge protection)  
- Clip range rendering for testing short sections  

These settings directly affect visual stability and comfort.

---

### 5. Open Preview Quick Testing

Click **Open Preview** to test your depth before rendering.

The preview window lets you:

- Scrub frame-by-frame through the video  
- Test different 3D visualization modes  
- Tune depth strength in real time  

This step is highly recommended before starting a full render.

---

## Using Preview Modes for Tuning

### Shift Heatmap (Best for Depth Tuning)

The **Shift Heatmap** view visualizes stereo displacement:

- **Dark blue** = closest to the viewer (strong pop-out)
- **Green/yellow** = mid-depth
- **Red** = far background

Use this mode to:

- Balance foreground, midground, and background shift
- Avoid extreme parallax that may cause discomfort
- Ensure depth layers are clean and separated

---

### Red–Blue Anaglyph (Quick Stereo Check)

Switch to **Red–Blue Anaglyph** mode to:

- Visually inspect stereo alignment
- Spot edge artifacts and ghosting
- Verify convergence placement

If you see halos or misalignment:

- Reduce foreground shift
- Adjust convergence strength
- Adjust MG Shift to reduce edge tearing

---

### Other Preview Tools

Additional preview modes allow:

- Shift intensity visualization  
- Parallax direction overlays  
- Feather mask inspection  
- Convergence Guide Overlay

These are useful for diagnosing depth artifacts and fine-tuning advanced scenes.

---

### 6. Tune Depth & Parallax Controls

Back in the main 3D Generator tab, adjust:

- Foreground Shift  
- Midground Shift  
- Background Shift  
- Convergence Strength  
- Parallax Balance  
- Stereo Scaling (IPD)  
- Depth Pop Gamma  

Begin with a built-in preset and refine depth using the Live Preview for real-time feedback.

---

### 7. Start Rendering

Once satisfied:

- Click **Generate 3D** for a single render  
- Or **Start Batch Render** for multiple inputs  

Monitor progress, FPS, and ETA in the Video Info panel.

---

## Recommended First-Time Workflow

1. Load source + depth video  
2. Configure Encoder Settings  
3. Enable basic Processing Options  
4. Open Live Preview  
5. Tune using Shift Heatmap + Anaglyph view  
6. Render a short clip range (optional)  
7. Render full video  

This prevents wasted long renders and ensures optimal depth quality.

---

### Sliders and Settings
### Depth Shaping (Pop Curve Controls)

This section remaps depth values to enhance separation between foreground, midground, and background.

It redistributes depth intensity rather than simply increasing shift amounts.

Used to:
- Add stronger 3D “pop”  
- Prevent flat-looking scenes  
- Maintain smooth depth transitions  

---

#### Depth Pop Gamma  

Controls how aggressively depth values are redistributed across the scene.

- Lower values create softer, flatter depth  
- Higher values increase separation between depth layers for stronger 3D impact  

Main control for overall depth strength.

---

#### Pop Mid  

Controls where the curve focuses its strongest depth separation.

- Lower values emphasize foreground depth  
- Higher values emphasize midground and background  

Useful when subjects feel flat but backgrounds already have depth.

---

#### Stretch Lo  

Controls how much detail is expanded in the near-depth (foreground) range.

- Lower values keep foreground depth tighter and more subtle  
- Higher values expand foreground depth separation for stronger subject pop  

Best used when close objects feel compressed or lack depth clarity.

---

#### Stretch Hi  

Controls how much detail is expanded in the far-depth (background) range.

- Lower values keep background depth softer and closer  
- Higher values push distant elements farther back for greater scene scale  

Best used when environments feel flat or lack depth range.

---

#### FG Pop ×  

Multiplies foreground depth strength after curve shaping.

- Lower values keep subjects natural  
- Higher values exaggerate subject separation  

Use for fine-tuning how strongly subjects stand out.

---

#### BG Push ×  

Multiplies background depth recession after curve shaping.

- Lower values keep environments closer  
- Higher values increase cinematic depth scale  

Use to enhance scene size without flattening foreground.

---

#### Subject Lock  

Biases depth shaping toward detected subject depth.

When enabled:
- Keeps main subjects visually dominant  
- Prevents flattening during strong depth enhancement  

Best for character-focused scenes and dialogue shots.

---

#### Foreground Shift  
Controls how strongly foreground objects are pushed toward the viewer.

Higher values:
- Increase 3D “pop-out” effect  
- Make characters and close objects appear closer  

Lower values:
- Create more subtle depth  
- Reduce eye strain during long viewing sessions  

Use this to define how aggressive the 3D effect feels.

---

#### Midground Shift  
Controls depth separation for objects between foreground and background.

Higher values:
- Increase depth layering across the scene  
- Improve sense of spatial depth  

Lower values:
- Keep mid-depth areas flatter  

Helps prevent scenes from feeling like only foreground and background exist.

---

#### Background Shift  
Controls how far distant elements recede into depth.

Higher negative values:
- Push backgrounds deeper  
- Increase cinematic depth scale  

Lower values:
- Keep backgrounds closer to the screen plane  

Useful for adding scale without over-popping subjects.

---

#### Convergence Strength  
Controls how strongly the zero-parallax plane is adjusted.

Higher values:
- Move focus plane faster between scenes  
- Increase perceived depth shifts  

Lower values:
- More stable and subtle convergence  

Use smaller values for comfort on long content.

---

#### Zero Parallax Strength  
Fine-tunes the exact depth level that sits at the screen surface.

Adjust this when:
- Objects feel too far forward  
- Or everything feels pushed backward  

This is your precision convergence control.

---

#### Stereo Scaling (IPD)  
Controls overall stereo separation intensity.

Higher values:
- Stronger depth effect  
- Larger parallax  

Lower values:
- More comfortable viewing  
- Subtle depth  

Acts like virtual eye separation.

---

#### Sharpness Factor  
Enhances perceived edge clarity in stereo output.

Higher values:
- Crisper depth edges  
- Emphasizes fine detail  

Lower values:
- Softer transitions  

Use moderately to avoid halo artifacts.

---

#### Parallax Balance  
Balances depth emphasis between foreground and background.

Toward foreground:
- Stronger pop-out effect  

Toward background:
- Deeper environments  

Useful for scene-specific tuning.

---

#### Max Pixel Shift (%)  
Limits the maximum allowed parallax displacement.

Lower values:
- Safer for VR  
- Prevent extreme eye strain  

Higher values:
- Allow more aggressive depth  

Acts as a safety clamp.

---

## Optional Advanced Controls

#### Floating Window (DFW)  
Adds cinematic edge protection to prevent objects from breaking the screen border.
Helps maintain professional stereo composition.

---

#### Dynamic Convergence (Stabilization)

Automatically adjusts convergence based on scene depth while smoothing transitions over time.

When enabled:
- Tracks subject depth naturally  
- Smooths frame-to-frame convergence  
- Reduces flicker and sudden depth jumps  
- Improves long-term viewing comfort  

Recommended for full-length videos.

---

#### Stabilize Zero-Parallax (Center-Depth)

Keeps the zero-parallax plane aligned with the scene’s dominant depth range.

When enabled:
- Prevents depth drift  
- Keeps subjects consistently at screen depth  
- Reduces eye strain during transitions  

Works alongside Dynamic Convergence for stable depth behavior.

Recommended for long content and fast scene changes.

---

#### Edge-Aware Masking  
Detects strong depth edges and suppresses stereo artifacts.

Benefits:
- Reduces halos  
- Improves depth cleanliness  

Enable when ghosting appears around subjects.

---

#### Feathering  
Softens depth transitions between layers.

Benefits:
- More natural blending  
- Less harsh stereo edges  

Pairs well with Edge Masking.

---

#### Depth of Field Simulation  
Applies subtle focus blur based on depth.

Adds cinematic realism when used sparingly.

---

## Clip Range (Optional)

Set start and end timecodes to render only a portion of the video.

Useful for:
- Testing settings quickly  
- Tuning difficult scenes  
- Avoiding long re-renders  

Recommended before full-length renders.

## Audio Tool (Audio Ripper & Attacher)

The Audio Tool lets you extract audio from videos, attach external audio tracks to videos, or attach audio and then stitch multiple clips into one final file.

This is useful when:
- Your source has no audio after processing
- You rendered clips in batches and need the original audio back
- You need to fix audio sync with an offset
- You have per-scene outputs and want one seamless final export

---

## What This Tool Can Do

### 1) Rip (Extract) Audio
Extracts the main audio track from one or more video files and saves it as a separate audio file.

Best for:
- Saving original audio before heavy processing
- Creating audio files you can re-attach later
- Archiving multiple language tracks (if you select the right source file)

---

### 2) Attach (Mux) Audio
Takes an external audio file and adds it to a video file (without re-encoding by default).

Best for:
- Restoring audio after a render
- Replacing audio with a clean track
- Adding a different language track

---

### 3) Attach + Stitch
Attaches audio to multiple clips and then stitches them into one final continuous video.

Best for:
- Batch renders that output multiple clips
- Scene-split workflows where you want a single final movie

---

## Step 1 — Add Your Source Videos

Use the **Step 1 — Sources** panel:

- **Add Files** to select multiple videos
- **Add Folder** to load every supported video in a folder
- **Up / Down** to reorder clips (important for stitching)
- **Remove / Clear** to clean the list

Supported video formats include MP4, MKV, MOV, AVI, WEBM, and more.

---

## Step 2 — Choose Your Operation

In **Step 2 — Operation & Options**, set **Operation** to one of:

- **rip**  
- **attach**  
- **attach_stitch**

---

## Audio Source Modes (Attach / Attach+Stitch)

### Auto-match from folder
Uses a folder of audio files and automatically matches each audio track to each video.

Matching rules:
- Exact name match works best  
  Example:  
  `scene_001.mp4` matches `scene_001.wav`

- If names are similar, it tries common patterns  
  Example:  
  `scene_001.mp4` matches `scene_001_audio.wav`

- If needed, it can match by the last number in the filename  
  Example:  
  `clip12.mp4` matches `audio12.m4a`

Best for:
- Batch clip workflows
- Scene split outputs

---

### Single audio for all
Uses one audio file and applies it to every video in the list.

Best for:
- One continuous audio track
- Short test clips that all share the same audio

---

## Audio Offset (Sync Fix)

**Audio offset (sec)** shifts the audio forward or backward.

- Positive offset: audio starts later (delays audio)
- Negative offset: audio starts earlier (pulls audio forward)

Use this if:
- Lip sync is slightly off
- Your pipeline introduced a delay
- Your stitched output drifts out of sync

Tip:
Start with small adjustments like ±0.05 to ±0.20 seconds.

---

## Force Re-encode (Per-clip Attach)

By default, attaching audio uses **copy** mode (fast, no quality loss).

Enable **Force re-encode** only when needed.

Use it when:
- The video or audio won’t mux cleanly
- A container doesn’t support the stream format
- You want to convert codecs for compatibility

Options include:
- Video: copy, libx264, libx265, h264_nvenc, hevc_nvenc
- Audio: copy, aac, mp3, opus, flac, ac3, eac3

---

## Rip Settings (Extract Audio)

### Rip codec
Controls how audio is extracted:

- **copy**  
  Fastest, no quality loss, keeps original codec when possible

- **aac / mp3 / opus / flac / wav / ac3 / eac3**  
  Re-encodes audio into the selected codec

### Bitrate (kbps)
Used when re-encoding audio (example: 192 kbps).

Higher bitrate:
- Better quality
- Larger files

---

## Final Stitch Settings (Attach+Stitch)

When using **attach_stitch**, the tool re-exports a final stitched file to ensure:
- All clips match size and FPS
- Pixel format is compatible
- Audio is gapless when possible

Key options:
- **Final vcodec**: auto, libx264, libx265, NVENC options
- **CRF/CQ**: quality level (lower = higher quality)
- **Preset**: speed vs compression efficiency
- **acodec / bitrate**: output audio format and bitrate

---

## Step 3 — Output Settings

### Rip output folder
Required for **rip** mode.
This is where extracted audio files will be saved.

---

### Per-clip output folder
Required for **attach** and **attach_stitch**.
This is where the muxed video files will be saved.

---

### Final stitched output
Required only for **attach_stitch**.
This is the final single stitched video file.

---

## Preview Button (Recommended)

Click **Preview** before running.

It will show:
- Your selected mode
- What audio source is being used
- Video-to-audio pairing results (including missing matches)

This helps prevent wasted runs.

---

## Run Button

Click **Run** to start processing.

A progress window will appear showing FFmpeg output logs.
If something fails, the logs usually show which codec or file caused the issue.

---

## Common Workflows

### Restore audio after a 3D render
1. Add your rendered video
2. Set mode to **attach**
3. Choose **Single audio for all**
4. Pick the original audio track
5. Set offset if needed
6. Run

---

### Batch clips + stitch into one final movie
1. Add clips in correct order
2. Set mode to **attach_stitch**
3. Use **Auto-match from folder** or **Single audio**
4. Set per-clip output folder
5. Set final stitched output file
6. Run

---

## Notes

- FFmpeg must be installed and available in PATH for this tool to work.
- For best matching results, keep audio filenames close to the video filenames.
- If a video has no audio when ripping, it will be skipped automatically.

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

Typical starter values:
- FG shift: `6 to 10`
- MG shift: `1 to 3`
- BG shift: `-3 to -6`

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
- FG/MG/BG: `8 / 2 / -4`

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
4. Configure Encoder Settings and Processing Options  
5. Open Live Preview and tune depth using Shift Heatmap + Anaglyph  
6. Test short Clip Range (optional)  
7. Render final full-length 3D video  

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

- Increase **Depth Pop Gamma**  
- Raise **Foreground Shift** slightly  
- Adjust **Pop Mid** toward subject depth  

---

### Seeing Halos or Ghosting

- Enable **Edge-Aware Masking**  
- Increase or Decrease MG Shift to eliminate Edge Tearing 
- Reduce Sharpness Factor  

---

### Jitter Between Scenes

- Enable **Dynamic Convergence**  
- Enable **Stabilize Zero-Parallax**  
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

