# The VisionDepth3D Method

### v4.0 Backend rewrite

An advanced, real-time stereo rendering pipeline for 2D-to-3D conversion across stereoscopic, VR, and cinematic display workflows.

VisionDepth3D combines subject-aware depth shaping, structured disparity weighting, GPU stereo warping, contour-safe repair logic, floating-window control, and temporal stabilization into a single render pipeline designed for controllable, believable 3D structure.

---

## Overview

The current VisionDepth3D method is built around a consistent mathematical workflow:

1. Normalize the incoming depth map into a stable working depth field.
2. Estimate the tracked subject depth from normalized raw depth.
3. Create a second shaped depth representation specifically for disparity design.
4. Build stereo structure from normalized near, mid, and far weighting bands.
5. Anchor zero parallax using the same weighting model that drives the stereo shift field.
6. Apply dynamic convergence and window control from the tracked subject path.
7. Render both eyes through GPU warp grids.
8. Detect and repair contour stress, disocclusion strips, and frame-edge violations using stereo-aware repair logic.
9. Apply optional DOF, color processing, formatting, and export.

This separation between **tracking depth**, **disparity depth**, and **repair analysis** is a defining part of the current pipeline.

---

## Core Method Components

### 1. Subject-Tracked Depth Normalization
VisionDepth3D begins from a normalized working depth field and estimates the subject from the **raw normalized depth**, before stylized stereo shaping is applied.

This subject estimate is temporally stabilized and used as the primary anchor for:
- subject locking
- convergence behavior
- floating-window control
- focal-depth decisions

This keeps subject placement more stable across motion and scene changes.

---

### 2. Pop-Control Depth Shaping
The `shape_depth_for_pop` stage creates a separate disparity-design depth copy used to build stereo structure.

This stage performs:
1. percentile stretching of the working depth range
2. subject-aware recentering
3. gamma-based reshaping around a configurable stereo midpoint

This allows VisionDepth3D to tune how near, mid, and far structure are emphasized without corrupting the underlying subject-tracking depth used for stability.

---

### 3. Structured Near / Mid / Far Disparity Weighting
Instead of relying on a simple linear depth-to-shift mapping, the current pipeline builds stereo structure from normalized near, mid, and far weighting bands.

These bands are blended into a structured disparity field using:
- foreground emphasis
- mid-depth continuity
- controlled far-depth push

This makes the stereo result read more like layered spatial structure instead of a flatter or overly heuristic depth push.

---

### 4. Subject-Aware Zero-Parallax Anchoring
The zero-parallax plane is not treated as an isolated offset.

VisionDepth3D derives subject anchoring from the **same near / mid / far weighting model** used to compute the main stereo shift field. This keeps subject locking and rendered disparity in the same mathematical space.

The result is a more coherent screen-plane lock with less mismatch between:
- perceived subject position
- convergence behavior
- actual stereo structure

---

### 5. Dynamic Convergence from the Tracked Subject Path
Dynamic convergence is applied from the tracked subject estimate rather than from a separately reinterpreted shaped-depth estimate.

This reduces semantic mismatch between:
- subject tracking
- zero-parallax anchoring
- convergence bias
- final disparity behavior

It also improves stability when scenes contain strong contrast, complex edges, or fast depth transitions.

---

### 6. Dynamic Parallax Scaling by Scene Variance
VisionDepth3D includes scene-aware parallax scaling based on normalized depth variance in the central view.

This allows the system to adapt stereo strength based on shot complexity:
- gentler scaling for flatter scenes
- stronger scaling for more spacious or layered scenes

This helps preserve comfort while still allowing the pipeline to open up on shots that can support stronger depth separation.

---

### 7. GPU Stereo Warp Rendering
Both eyes are generated through GPU tensor warping using per-eye shift maps and `grid_sample`-based resampling.

This allows:
- smooth depth-weighted stereo construction
- real-time or near-real-time GPU execution
- integration with HDR, VR180, and multi-format export paths

The stereo renderer is designed to work as a continuous shift field, not a discrete layered cutout system.

---

### 8. Edge-Aware Shift Limiting
VisionDepth3D uses edge-aware suppression to reduce unstable disparity near hard depth transitions such as:
- hairlines
- fingers
- shoulders
- thin foreground contours

This acts as a pre-warp safety layer that helps reduce contour splitting, halos, and harsh edge drag before stereo repair is applied.

---

### 9. Contour-Safe Repair and Protect Masking
The updated method introduces a more explicit contour-repair stage built around:
- raw intended shift analysis
- one-sided repair masks
- contour protection barriers
- directional background fill

A key design choice is that repair/protect analysis is derived from the **intended geometric shift**, not only the fully smoothed or edge-suppressed result.

This makes the repair system more responsive to the real disparity conditions that create stereo tears and slivers.

---

### 10. Visual Stress Detection After Warp
In addition to geometry-based mask building, VisionDepth3D analyzes the warped eyes themselves for visible contour stress.

This post-warp visual stress cue helps detect:
- stretched silhouette strips
- stressed high-contrast edges
- contour-adjacent instability that pure shift gradients may under-report

That signal is fused into the repair stage so the system can respond to what is actually visible in the rendered eye.

---

### 11. Measured Frame-Edge Violation for Floating Window Control
The floating-window stage now responds to measured frame-edge violation instead of relying only on inferred zero-parallax magnitude.

VisionDepth3D evaluates edge-risk directly from the shift field near the left and right frame borders, then uses that measurement to drive:
- floating-window side choice
- window width
- temporal easing of the window response

This makes the floating-window system more tightly linked to real stereo risk at the frame edges.

---

### 12. Temporal Stabilization Layers
VisionDepth3D uses multiple stabilization layers across the render path, including:
- subject-depth EMA
- convergence smoothing
- floating-window easing
- optional shift EMA
- depth normalization EMA

These controls are used to reduce shimmer, subject drift, and unstable stereo jitter while preserving enough responsiveness for motion-heavy shots.

The current pipeline also exposes debug-oriented controls so temporal behavior can be tested directly against raw geometry.

---

### 13. Motion-Aware Focal Depth Tracking
Depth-of-field behavior is driven by a focal-depth tracker that responds to both subject estimation and scene motion.

This allows VisionDepth3D to shift between:
- more stable focus in calm shots
- faster focus adaptation in active scenes

The result is a DOF system that behaves more like a dynamic cinematic effect than a fixed blur pass.

---

### 14. Matte Sculpting and Rounded Subject Depth
VisionDepth3D includes optional depth-roto sculpting that can round subject depth using:
- distance transforms
- feathered matte blending
- temporal matte stabilization

This is used to improve the volume and curvature of subjects without requiring a heavy neural reconstruction step.

---

### 15. Debugable Stereo Telemetry
A notable part of the current method is that the stereo pipeline is instrumented for analysis.

VisionDepth3D can expose internal stereo metrics such as:
- tracked subject depth
- zero-parallax offset
- edge-window violation
- repair-mask strength
- contour-protection strength
- validity statistics

This allows the pipeline to be tuned as an engineering system rather than only by visual guesswork.

---

## Supporting Features

The VisionDepth3D method also integrates a number of supporting systems that extend the core stereo workflow:

- temporal percentile EMA depth normalization
- GPU color grading and sharpening
- automatic black bar detection and aspect preservation
- VR180 flat-to-equirect projection workflow
- HDR-capable frame handling paths
- multiple stereoscopic output formats
- FFmpeg-based encode and export pipeline
- CPU and GPU codec support across NVENC, AMF, QSV, and software encoders

---

## Summary Table

| Category | Component |
| --- | --- |
| **Core Method** | Subject-tracked depth normalization |
|  | Pop-control depth shaping |
|  | Structured near / mid / far disparity weighting |
|  | Subject-aware zero-parallax anchoring |
|  | Dynamic convergence from tracked subject depth |
|  | Dynamic parallax scaling by scene variance |
|  | GPU stereo warp rendering |
|  | Edge-aware shift limiting |
|  | Contour-safe repair and protect masking |
|  | Visual stress detection after warp |
|  | Measured frame-edge violation for floating window control |
|  | Temporal stabilization layers |
|  | Motion-aware focal depth tracking |
|  | Matte sculpting and rounded subject depth |
|  | Debugable stereo telemetry |
| **Supporting Systems** | HDR-ready frame handling |
|  | VR180 projection workflow |
|  | Black bar detection and aspect handling |
|  | GPU color grading and sharpening |
|  | Multi-format stereo export |
|  | FFmpeg codec and container pipeline |

---

## Closing

VisionDepth3D is built as a holistic GPU-accelerated 2D-to-3D rendering method where depth interpretation, subject anchoring, stereo structure, contour protection, floating-window response, and post-warp cleanup are all part of a unified pipeline.

The current method places strong emphasis on:
- controllable stereo structure
- subject stability
- contour safety
- practical real-time performance
- direct debug visibility for iterative tuning

📄 Licensed under: VisionDepth3D Custom Use License (No Derivatives)  
🔗 Project: [https://github.com/VisionDepth/VisionDepth3D](https://github.com/VisionDepth/VisionDepth3D)
