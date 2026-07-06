# VisionDepth3D User Guide

## Overview

VisionDepth3D, also called VD3D, is a 2D-to-3D conversion suite for depth generation, depth blending, stereoscopic rendering, VR output, FPS interpolation, AI upscaling, and live 3D preview.

This updated guide reflects the newer 3D workflow, including the beginner **3D Assistant**, the expanded **Advanced 3D Controls**, the new layered depth behavior, and the new **Output & Encoding Presets**.

VD3D is designed for two types of users:

- **Beginner users** can use the 3D Assistant and codec presets without understanding every technical setting.
- **Advanced users** can open the full stereo controls and tune exact shift, convergence, depth shaping, subject lock, edge repair, and output settings.

---

## Recommended VD3D Workflow

1. **Prepare your source video**
   - Check resolution, frame rate, aspect ratio, black bars, and audio.
   - If the movie changes aspect ratio, split it into separate sections first.

2. **Generate a depth map**
   - Use the **Depth Engine** tab.
   - Start with a Depth Anything V2 model or Video Depth Anything model.

3. **Check depth direction**
   - Make sure near objects are closer and backgrounds are farther.
   - If the result feels inverted, enable **Invert Depth**.

4. **Optionally blend depth maps**
   - Use **Depth Blender** when one model gives better subjects and another gives better backgrounds.

5. **Load source and depth in 3D Generator**
   - Load the original video.
   - Load the matching depth video.
   - Select an output path.

6. **Start with the 3D Assistant**
   - Choose a 3D Style.
   - Adjust 3D Strength, Pop-Out, Screen Depth, Depth Comfort, and Subject Stability.

7. **Preview before rendering**
   - Use Red-Blue Anaglyph, HSBS, Shift Heatmap, Overlay Arrows, and convergence guides.

8. **Render a short test clip**
   - Use Clip Range for 10 to 20 seconds before a full render.

9. **Choose an encoding preset**
   - Use Fast Preview for testing.
   - Use Balanced Final or High Quality Final for finished renders.

10. **Render the final video**

---

## Quick Feature Map

| Goal | Start Here |
|---|---|
| Generate a depth map | Depth Engine |
| Blend two depth maps | Depth Blender |
| Convert 2D video to 3D | 3D Generator |
| Use beginner 3D controls | 3D Assistant |
| Fine tune stereo manually | Advanced 3D Controls |
| Move the screen plane | Screen Depth or Screen Plane Offset |
| Create deeper backgrounds | BG Push and Layered Depth Warp |
| Keep subjects stable | Subject Stability, Subject Lock, Subject Plane Lock |
| Reduce edge artifacts | Edge Repair Quality, Edge Masking, Feathering |
| Export VR180 | VR180 Output Settings |
| Improve FPS | FPS / Upscale |
| Upscale video | FPS / Upscale |
| Use live 3D | VD3D Live |

---

# Depth Engine

The **Depth Engine** creates depth maps from images or videos.

A depth map tells VD3D what should appear near, mid, or far when generating stereo 3D.

## Quick Start: Image Depth

1. Open **Depth Engine**.
2. Select a depth model.
3. Choose an output directory.
4. Keep default settings for your first test.
5. Click **Process Image**.
6. Select your image.

VD3D saves an output such as:

```text
yourfilename_depth.png
```

## Quick Start: Video Depth

1. Open **Depth Engine**.
2. Select a depth model.
3. Choose an output directory.
4. Click **Process Video**.
5. Select your video.

VD3D saves an output such as:

```text
yourvideo_depth.mkv
```

## Inference Resolution

Controls the internal size used by the depth model.

Lower values:

- faster
- less VRAM
- less detail

Higher values:

- more detail
- more VRAM
- slower

Good starting values:

```text
512x288
704x384
768x432
```

## Batch Size

Controls how many frames process at once.

- Higher batch size can be faster on strong GPUs.
- Lower batch size is safer if VRAM is limited.

If you run out of memory, lower batch size first.

## Invert Depth

Enable **Invert Depth** if near and far values are reversed.

VD3D usually expects:

```text
brighter = closer
darker = farther
```

## Depth Normalization

Depth Normalization stabilizes near and far depth range across frames.

Use it for:

- full movie depth maps
- smoother 3D conversion
- reducing depth flicker
- reducing frame-to-frame breathing

Disable it for:

- fast tests
- maximum speed
- models that are already stable

## Save Frames

Enable Save Frames if you want PNG depth frames in addition to a depth video.

Useful for:

- inspection
- debugging
- external editing
- frame-based workflows

---

# Depth Blender

The **Depth Blender** combines two depth sources into one cleaner result.

Use it when:

- one model has better subject edges
- another model has better backgrounds
- depth flickers
- subject edges shimmer
- a single model does not produce enough usable depth separation

## Inputs

Depth Blender can use:

- folders of PNG depth frames
- two depth videos

## Preview

1. Load both depth sources.
2. Click **Preview Now**.
3. Scrub frames.
4. Adjust blend controls.
5. Compare the base depth and blended output.

## Important Blend Controls

### White Strength

Controls how strongly the second depth source can add bright near-depth information.

### Feather Blur

Controls softness between blended regions.

### CLAHE Clip Limit

Boosts local depth contrast.

### Bilateral Filtering

Smooths noisy depth while trying to preserve edges.

---

# 3D Generator

The **3D Generator** converts a source video and matching depth video into stereoscopic 3D.

It uses:

- source RGB video
- depth video
- 3D Assistant settings
- advanced stereo controls
- subject tracking
- depth shaping
- depth sculpting
- optional depth-order warp
- edge repair
- output and codec settings

The newest 3D updates focus on creating a stronger **window into the scene** effect, where backgrounds sink inward and subjects stay separated instead of the image looking flat with only foreground pop-out.

---

## Important Shift Direction

The current VD3D shift convention is:

```text
Foreground Shift: usually negative
Midground Shift: usually slightly negative or near zero
Background Shift: usually positive
```

Recommended range:

| Control | Range |
|---|---|
| Foreground Shift | -5.0 to -10.0 |
| Midground Shift | -0.5 to -3.0 |
| Background Shift | +2.0 to +5.0 |

A good relationship is:

```text
Foreground Shift < Midground Shift < Background Shift
```

Example:

```text
FG -8.9 / MG -2.5 / BG +2.9
```

Older presets may not transfer correctly if they used positive foreground and negative background values.

---

# 3D Assistant Beginner Controls

The **3D Assistant** is the recommended starting point for most users.

It gives simple controls that map into the advanced render system.

Beginner controls:

- 3D Style
- 3D Strength
- Pop-Out
- Screen Depth
- Depth Comfort
- Subject Stability

## 3D Style

The 3D Style dropdown provides preset starting points.

Common styles:

| Style | Use Case |
|---|---|
| Comfortable Cinema | Balanced movie viewing |
| Strong Pop-Out | More forward foreground effect |
| Deep Background | More background recession |
| Close-Up Safe | Safer faces and dialogue scenes |
| VR Comfortable | Lower strain for headset viewing |
| Wide / Deep Scene | Stronger depth for rooms and scenery |
| Clean Edge | Safer edge handling |
| Showcase Mode | Strong demo-style depth |
| Custom | Manual adjusted settings |

When you move a beginner slider manually, the style becomes **Custom**.

## 3D Strength

Controls the overall stereo separation.

Higher values:

- stronger 3D
- larger foreground/background separation
- deeper environments when the depth map supports it
- more artifact and comfort risk

Lower values:

- softer 3D
- less strain
- fewer artifacts

## Pop-Out

Controls how much foreground objects feel like they come toward the viewer.

Higher values:

- stronger foreground presence
- stronger subject pop
- more dramatic 3D

Lower values:

- more natural foreground
- less eye strain

Pop-Out is not the main background-depth control. Use **Screen Depth**, **BG Push**, and **Layered Depth Warp** for deeper environments.

## Screen Depth

Controls where the neutral screen plane sits.

Meaning:

```text
0 = closer / more pop-out feeling
50 = neutral
100 = deeper behind the screen
```

Use this when:

- the whole scene feels too close
- the whole scene feels too far back
- the subject needs to sit closer to screen level
- backgrounds need to sit deeper behind the screen

Adjust slowly. Small changes can strongly affect comfort.

## Depth Comfort

Controls safety and strain reduction.

Higher values:

- reduce extreme shift
- improve comfort
- soften aggressive stereo

Lower values:

- allow stronger depth
- allow more pop
- increase artifact risk

## Subject Stability

Controls how strongly the subject or main depth region is stabilized.

Higher values:

- keeps subjects stable
- reduces subject drift
- helps faces and dialogue scenes

Lower values:

- allows more subject movement
- can create stronger pop
- may be less stable

---

## Beginner Starting Points

### Comfortable Movie

```text
3D Strength: 50
Pop-Out: 35
Screen Depth: 50
Depth Comfort: 70
Subject Stability: 60
```

### Deep Background

```text
3D Strength: 65
Pop-Out: 35
Screen Depth: 58 to 65
Depth Comfort: 55
Subject Stability: 55
```

### Strong Demo

```text
3D Strength: 75
Pop-Out: 55
Screen Depth: 45 to 55
Depth Comfort: 45
Subject Stability: 45
```

### Close-Up Safe

```text
3D Strength: 45 to 55
Pop-Out: 25 to 35
Screen Depth: 48 to 52
Depth Comfort: 80
Subject Stability: 80 to 90
```

---

# Advanced 3D Controls

Advanced 3D Controls are for exact stereo tuning.

Use them when:

- a scene is difficult
- the beginner controls are close but not perfect
- you are creating presets
- you need exact control over screen plane, convergence, edge repair, or depth shaping

## Foreground Shift

Controls how strongly near objects are pulled toward the viewer.

Recommended:

```text
Natural: -5.0 to -7.0
Strong: -8.0 to -10.0
Aggressive: -10.0 to -12.0
```

## Midground Shift

Controls objects between foreground and background.

Recommended:

```text
Subtle: -0.5
Natural: -0.8 to -2.0
Strong: -2.0 to -3.0
```

Midground helps prevent the render from looking like only two flat layers.

## Background Shift

Controls how far backgrounds sink behind the screen.

Recommended:

```text
Subtle: +1.0 to +2.0
Natural: +2.0 to +3.0
Strong: +3.0 to +5.0
```

If the background still looks flat, do not only raise Background Shift. Also check BG Push, Depth Pop Gamma, Layered Depth Warp, and depth map quality.

## Max Pixel Shift

Limits maximum allowed stereo displacement.

Recommended:

```text
Comfort: 0.020 to 0.045
Strong: 0.050 to 0.071
```

This is a safety clamp. Higher values allow stronger 3D but increase artifact and comfort risk.

## Parallax Balance

Controls overall stereo balance.

Recommended:

```text
Comfort: 0.35 to 0.55
Strong: 0.55 to 0.75
```

## Screen Plane Offset

Advanced version of Screen Depth.

This controls where the zero-parallax screen plane sits.

Use it to fine tune the scene after the beginner Screen Depth slider gets close.

## Convergence Strength

Controls how strongly convergence is applied.

Convergence is a placement and comfort tool, not the main depth creator.

Recommended:

```text
0.000 to 0.030 for stable cinematic output
0.030 to 0.120 for stronger tests
```

Too much convergence can make the whole stereo volume feel like it is sliding around.

## Dynamic Convergence

Automatically adjusts convergence based on tracked subject depth.

Use it for:

- full movies
- smoother scene transitions
- subject comfort
- fewer sudden depth jumps

Disable it temporarily when testing raw pop-out.

## Subject Lock

Controls how strongly the tracked subject is pinned near the screen plane.

Higher values:

- more subject stability
- safer faces
- less subject drift
- can reduce strong pop-out

Recommended:

```text
0.25 to 0.85
```

## Subject Plane Lock

Locally reduces disparity around the subject depth.

This keeps the subject stable while allowing background depth to remain strong.

Recommended:

```text
0.20 to 0.35
```

## Subject Plane Lock Width

Controls how wide the protected subject depth band is.

Recommended:

```text
0.10 to 0.14
```

Meaning:

```text
0.08 = tight protection
0.12 = good default
0.18 = wide protection, safer but flatter
```

If subjects feel unstable, increase width slightly. If too much of the scene feels protected and flat, reduce width.

## Depth Pop Gamma

Controls how depth is reshaped before stereo shift.

Lower values:

- stronger depth punch
- stronger near and mid separation
- stronger background separation when used with BG Push

Higher values:

- softer depth
- more natural scenes
- less cutout effect

Recommended:

```text
0.86 to 0.94 for stronger cinematic depth
1.00 to 1.15 for softer natural depth
```

## Pop Mid

Controls where the depth shaping curve focuses.

Recommended:

```text
0.45 to 0.50
```

## Stretch Lo and Stretch Hi

Stretch Lo controls near-depth stretch.

Recommended:

```text
0.02 to 0.08
```

Stretch Hi controls far-depth stretch.

Recommended:

```text
0.94 to 1.00
```

## FG Pop Multiplier

Boosts foreground contribution.

Recommended:

```text
1.00 to 1.20
```

## BG Push Multiplier

Boosts background recession.

Recommended:

```text
1.03 to 1.12
```

This is one of the most important controls for the new deeper background look.

## Foreground Curvature

Adds a subtle rounded shape to near foreground objects.

Use it for:

- faces
- bodies
- large foreground objects
- reducing cardboard-flat subjects

Recommended:

```text
0.03 to 0.08
```

Too much can look artificial.

## Sharpness Factor

Adds sharpening after stereo generation.

Use lightly. Too much can increase halos.

## Depth of Field

Adds optional depth-based blur.

Use sparingly. Too much can hide depth detail.

---

# Layered Depth and New 3D Illusion Features

The new 3D update improves the illusion that the screen is a window into a scene.

Older behavior could sometimes look like:

```text
flat background
subject popping forward
limited background recession
```

The new behavior is designed to create:

```text
subjects separated from background
backgrounds sinking inward
stronger room and environment depth
less flat-cardboard feeling
```

This is mainly done by:

1. **Cinematic Depth Sculpt**
2. **Depth-Order Forward Warp**

## Cinematic Depth Sculpt

Cinematic Depth Sculpt reshapes the depth map before stereo shift.

It protects the tracked subject while pushing far background areas deeper.

It improves:

- background recession
- subject/background separation
- room depth
- wide scene depth
- screen-window illusion

Depth Sculpt is driven by existing controls:

- Depth Pop Gamma
- BG Push
- Subject Plane Lock
- Subject Plane Lock Width
- FG Pop

## Depth-Order Forward Warp

Depth-Order Forward Warp is an experimental layered warp mode.

Classic VD3D warp is smooth and fast, but it samples the image backward. Depth-order warp adds a forward placement pass where pixels are placed by depth order.

Conceptually:

```text
far pixels are placed first
near pixels are placed last
foreground can sit over background
background can separate more believably
```

This helps create a stronger layered scene instead of a flat displacement effect.

## Layered Depth Warp Strength

If exposed in the UI, this controls how much depth-order warp is blended into the classic render.

Suggested meaning:

```text
0 = Classic VD3D warp only
50 = balanced layered warp
100 = strong depth-order warp
```

Recommended values:

```text
0.35 = safer for faces and motion
0.50 = good default
0.65 = strong room and showcase depth
0.80 = aggressive testing
```

If background depth improves but subject edges become stressed, lower Layered Depth Warp Strength first.

## Recommended Layered Depth Test

```text
FG Shift: -8.9
MG Shift: -2.5
BG Shift: +2.9
Max Pixel Shift: 0.071
Parallax Balance: 0.40
Depth Pop Gamma: 0.88
BG Push: 1.08
Subject Lock: 0.65
Subject Plane Lock: 0.25
Subject Plane Lock Width: 0.12
Convergence Strength: 0.015
Dynamic Convergence: On
Screen Plane Offset: 0.000
Floating Window: Off
Edge Repair: High
Layered Depth Warp: 0.50 to 0.65
```

---

# Preview Modes

Use preview before rendering full video.

## Red-Blue Anaglyph

Good for checking:

- stereo direction
- pop-out
- screen plane
- background recession
- edge artifacts

## HSBS

Shows a half side-by-side preview.

## Passive Interlaced

Useful for passive interlaced 3D displays.

## Shift Heatmap

Shows stereo displacement as a heatmap.

Use it to check:

- where shift is strongest
- whether foreground and background separate correctly
- whether shift is being clamped
- whether settings are too aggressive

## Shift Heatmap Abs

Shows shift strength without direction.

## Shift Heatmap Clipped

Shows a clipped range to inspect subtle shifts.

## Overlay Arrows

Shows shift direction with arrows.

## Left-Right Diff

Shows differences between eye views.

Good for spotting:

- excessive disparity
- ghosting
- edge tearing
- unstable depth

## Feather Mask and Feather Blend

Used for inspecting feathering and softened transitions.

## Convergence Guides

Convergence guides add registration marks in preview only.

They help show:

- screen level
- convergence comfort
- depth direction
- how far subjects and backgrounds sit relative to the screen

They are not meant for final output. Toggle them off when saving clean previews.

---

# Output and Encoding Presets

The **Output & Encoding** section controls final output format, stereo layout, codec, quality, HDR behavior, and file compatibility.

The new codec presets make encoding easier by matching settings to common goals.

## Output Format

| Output Format | Use Case |
|---|---|
| Full-SBS | Best side-by-side quality, full width per eye |
| Half-SBS | Smaller file, common TV/player support |
| VR / SBS | VR-style side-by-side output |
| VR180 Equirect Top-Bottom | VR180 headset output |
| VR180 Equirect Side-by-Side | VR180 headset output |
| Red-Cyan Anaglyph | Preview or red/cyan glasses |
| Passive Interlaced | Passive 3D displays |

## Stereo Output

| Stereo Output | Meaning |
|---|---|
| SBS | Normal stereo pair |
| Left eye only | Export only left eye |
| Right eye only | Export only right eye |
| Both eyes separately | Export left and right as separate files |

## Encoding Presets

### Compatibility Mode - Plays Everywhere

Best for maximum playback compatibility.

Use when:

- sharing with users
- playing on TVs
- uploading to common platforms
- avoiding codec issues

Usually uses safe H.264 style output.

### Fast Preview - Quick Test

Best for quick test renders.

Use when:

- tuning settings
- checking a short clip
- comparing presets
- testing screen plane or convergence

### Balanced Final - NVIDIA

Recommended default for NVIDIA users.

Use when:

- rendering final clips
- you want good quality and speed
- NVENC is available

### High Quality Final - NVIDIA

Best for higher quality NVIDIA renders.

Use when:

- making showcase clips
- quality matters more than speed
- larger files are acceptable

### Small File - HEVC

Best when file size matters.

Use when:

- storing many renders
- uploading large clips
- reducing file size
- your player supports HEVC

### 4K / Full-SBS High Quality - HEVC

Best for high-resolution Full-SBS renders.

Use when:

- rendering 4K
- rendering Full-SBS
- making high-quality headset or archive output

### Depth Map Output - Fast Safe

Best for depth-only or utility outputs.

Use when:

- exporting grayscale depth videos
- creating intermediate files
- testing depth pipeline output

### Archive Master - Large File

Best for high-quality masters.

Use when:

- preserving quality
- creating edit-friendly masters
- storage size is not a concern

### CPU Compatibility - Slow

Best fallback when hardware encoding fails.

Use when:

- NVENC fails
- AMF fails
- QSV fails
- no hardware encoder is available

### Custom

Use Custom when you want manual codec control.

Advanced users can tune:

- codec
- CQ / CRF
- bitrate
- preset
- container
- HDR
- audio
- pixel format

## Which Preset Should I Use?

| Goal | Recommended Preset |
|---|---|
| First test | Fast Preview - Quick Test |
| Most final NVIDIA renders | Balanced Final - NVIDIA |
| Best NVIDIA quality | High Quality Final - NVIDIA |
| Small file size | Small File - HEVC |
| 4K Full-SBS | 4K / Full-SBS High Quality - HEVC |
| Maximum compatibility | Compatibility Mode - Plays Everywhere |
| Depth utility output | Depth Map Output - Fast Safe |
| Archive quality | Archive Master - Large File |
| Hardware encoder fails | CPU Compatibility - Slow |
| Manual tuning | Custom |

## Codec Tips

- H.264 is safest for compatibility.
- H.265 / HEVC gives smaller files but may not play on older devices.
- AV1 is efficient but requires newer hardware and player support.
- NVENC is recommended for NVIDIA GPUs.
- AMF is used for supported AMD GPUs.
- QSV is used for supported Intel GPUs.
- CPU encoding is slower but reliable.

## HDR10 Preservation

Enable HDR10 preservation only when:

- your source is HDR10
- your output codec supports it
- your playback device supports HDR

Disable it when:

- source is SDR
- output looks washed out
- playback device has poor HDR support

---

# VR180 Output Settings

VR180 output is for stereoscopic headset viewing.

Supported modes can include:

- VR180 Equirect Top-Bottom
- VR180 Equirect Side-by-Side

## VR180 Equirect Presets

| Preset | Use Case |
|---|---|
| 2048x1024 per eye | Fast tests |
| 3072x1536 per eye | Balanced VR |
| 3840x1920 per eye | High-quality VR |
| 4096x2048 per eye | Very high quality |
| 5760x2880 per eye | Heavy showcase renders |

## VR180 Flat Working Presets

| Preset | Use Case |
|---|---|
| 1280x720 | Fast working render |
| 1920x1080 | Balanced quality |
| 2560x1440 | Cleaner source before VR warp |

## VR180 Tips

- Use short test clips.
- Keep pop-out moderate.
- Avoid strong edge violations.
- Use Edge Repair High for final clips.
- Test in a headset before full render.

---

# Clip Range Rendering

Clip Range lets you render only part of a video.

Use it for:

- testing settings
- checking difficult scenes
- comparing presets
- avoiding long failed renders

Example:

```text
Start: 00:01:20
End:   00:01:35
```

---

# FPS / Upscale

The **FPS / Upscale** tab can improve smoothness and resolution.

It can:

- extract frames
- run RIFE interpolation
- run Real-ESRGAN upscaling
- rebuild video
- optionally restore audio

## RIFE Interpolation

| Multiplier | Example |
|---|---|
| x2 | 30 FPS to 60 FPS |
| x4 | 30 FPS to 120 FPS |
| x8 | 30 FPS to 240 FPS |

Higher multipliers are smoother but require more processing.

## Real-ESRGAN Upscaling

Use it for:

- old movies
- low-resolution clips
- DVD sources
- cleaner 3D output

## Merged Pipeline

Best for:

- long videos
- lower memory use
- reliability

## Threaded Pipeline

Best for:

- strong GPUs
- faster processing
- shorter clips
- high-performance workflows

---

# VD3D Live

VD3D Live is a real-time 2D-to-3D system.

It can work with:

- screen capture
- video players
- games
- cameras
- capture cards

## Live Shift Convention

Use the current VD3D shift direction:

```text
FG shift: -5 to -10
MG shift: -0.5 to -2
BG shift: +2 to +5
```

Starter:

```text
FG/MG/BG: -6 / -0.8 / +2.2
```

Stronger:

```text
FG/MG/BG: -8.5 / -1.2 / +3.5
```

## Live Performance Tips

- Lower inference resolution for speed.
- Lower Depth FPS for stability.
- Use FP16 on CUDA.
- Use smoothing to reduce flicker.
- Avoid capturing the preview window on the same monitor.

---

# Hardware and Backend Support

## NVIDIA CUDA

Recommended backend for VD3D.

Best for:

- depth estimation
- 3D rendering
- interpolation
- upscaling
- NVENC encoding

## AMD / Intel DirectML

Useful on Windows for AMD and Intel GPUs.

DirectML may be slower than CUDA and some operations may fall back to CPU.

## AMD ROCm

Available on supported Linux systems and supported AMD GPUs.

## CPU

CPU fallback works but is much slower.

## FFmpeg Encoders

| GPU / Backend | Encoder Options |
|---|---|
| NVIDIA | h264_nvenc, hevc_nvenc, av1_nvenc |
| AMD | h264_amf, hevc_amf, av1_amf |
| Intel | h264_qsv, hevc_qsv, av1_qsv |
| CPU | libx264, libx265, libaom-av1, libsvtav1 |

---

# Troubleshooting

## The scene looks inverted

Check:

- depth inversion
- eye order
- foreground shift direction
- old presets
- depth map convention

## The background is too flat

Try:

- enable Layered Depth Warp
- increase BG Push
- lower Depth Pop Gamma slightly
- increase Background Shift carefully
- increase Screen Depth slightly
- use Depth Blender
- try a different depth model

Do not only increase 3D Strength. A flat depth map needs better depth shaping or blending.

## The subject pops out but the background is flat

Try:

- increase BG Push
- enable Depth-Order Warp
- use Wide / Deep Scene style
- use Deep Background style
- check depth map background detail
- reduce Subject Plane Lock Width if it protects too much

## The subject falls flat with the background

Try:

- increase Subject Lock slightly
- increase Subject Plane Lock
- use Subject Plane Lock Width around 0.12
- keep Screen Depth closer to neutral
- lower convergence if the stereo volume slides too much

## Edges tear or smear

Try:

- Edge Repair High
- Edge Masking On
- Feathering On
- lower Layered Depth Warp Strength
- lower Max Pixel Shift
- reduce Foreground Shift
- reduce BG Push

## Eye strain

Try:

- reduce Max Pixel Shift
- reduce 3D Strength
- increase Depth Comfort
- lower Parallax Balance
- reduce convergence
- use VR Comfortable style
- render a short test before final

## Cardboard cutout look

Try:

- lower FG Pop
- increase Depth Pop Gamma
- use Foreground Curvature lightly
- adjust Midground Shift closer to Foreground Shift
- use Depth Blender
- reduce overly aggressive Background Shift

## Preview text or language does not update

Check:

- labels are registered for translation
- combo boxes rebuild translated display text
- combo itemData stores English preset keys
- PreviewPanel has refresh_labels
- JobQueueDock stores status keys instead of translated status text
- dynamic log history does not need to retranslate after it is printed

---

# Recommended Presets

## Comfortable Cinema

```text
FG Shift: -6.4
MG Shift: -2.1
BG Shift: +2.2
Max Pixel Shift: 0.053
Parallax Balance: 0.35 to 0.45
Depth Pop Gamma: 0.94 to 1.05
BG Push: 1.03
Subject Lock: 0.40 to 0.65
Subject Plane Lock: 0.20 to 0.30
Subject Plane Lock Width: 0.12
Layered Depth Warp: 0.35 to 0.50
Edge Repair: Balanced or High
```

## Wide / Deep Scene

```text
FG Shift: -8.9
MG Shift: -2.5
BG Shift: +2.9
Max Pixel Shift: 0.071
Parallax Balance: 0.40
Depth Pop Gamma: 0.88
BG Push: 1.08
Subject Lock: 0.65
Subject Plane Lock: 0.25
Subject Plane Lock Width: 0.12
Convergence Strength: 0.015
Dynamic Convergence: On
Layered Depth Warp: 0.50 to 0.65
Edge Repair: High
```

## Clean Edge

```text
FG Shift: -9.0
MG Shift: -3.0
BG Shift: +2.4
Max Pixel Shift: 0.071
Parallax Balance: 0.35
Depth Pop Gamma: 1.00
Subject Lock: 0.80 to 0.95
Subject Plane Lock Width: 0.12 to 0.15
Layered Depth Warp: 0.35 to 0.50
Edge Repair: High
Edge Masking: On
Feathering: On
```

## Showcase

```text
FG Shift: -9.9
MG Shift: -3.0
BG Shift: +3.3
Max Pixel Shift: 0.071
Parallax Balance: 0.70 to 1.00
Depth Pop Gamma: 0.87
BG Push: 1.08 to 1.12
Subject Lock: 0.70 to 0.90
Subject Plane Lock Width: 0.12
Layered Depth Warp: 0.65
Edge Repair: High or Showcase
```

---

# Best Practices

- Start with beginner controls.
- Use preview before rendering.
- Render short clips first.
- Use Shift Heatmap to inspect stereo stress.
- Use Anaglyph to check real depth feeling.
- Use Layered Depth Warp when backgrounds are flat.
- Keep Subject Plane Lock Width around 0.10 to 0.14.
- Use Edge Repair High for final showcase clips.
- Use Balanced Final or High Quality Final for NVIDIA output.
- Use Compatibility Mode when sharing widely.
- Do not overuse convergence to create depth.
- Use depth maps with strong subject and background separation.
- Blend depth maps when one model is not enough.

Balanced depth usually looks more cinematic than extreme pop-out.

---

# Support and Updates

For updates, documentation, and releases:

- GitHub repository: VisionDepth3D
- Official VisionDepth3D website
- Community feedback and issue reports are welcome

VD3D continues to improve depth quality, stereo stability, render speed, preview tools, language support, encoding presets, and user workflow.

---

# End of User Manual
