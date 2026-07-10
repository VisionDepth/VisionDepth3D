# The VD3D Hybrid3D Method

### v5.0 Hybrid Stereo Backend

The **VD3D Hybrid3D Method** is VisionDepth3D's GPU-accelerated 2D-to-3D rendering architecture for stereoscopic video, still images, VR180, and cinematic display workflows.

It combines subject-aware depth interpretation, shaped disparity design, continuous GPU eye warping, depth-ordered forward projection, temporal stabilization, contour protection, disocclusion repair, and floating-window control in one coordinated render path.

The method is designed to create controllable stereo structure while preserving subject stability, depth continuity, and practical render performance.

---

## What “Hybrid3D” Means

The **Hybrid3D** name describes the way VD3D combines multiple complementary forms of stereo synthesis rather than relying on a single depth-to-offset operation.

The two primary eye-generation paths are:

1. **Continuous inverse GPU warping** using a normalized per-pixel shift field and `grid_sample` resampling.
2. **Depth-ordered forward warping** that moves source pixels into each target eye while writing farther surfaces first and nearer surfaces afterward.

The continuous warp preserves smooth, subpixel stereo motion. The depth-ordered path adds explicit foreground-over-background ordering and stronger layer separation. VD3D then blends the forward-warp result only where its validity mask confirms that the forward path owns the target pixel.

This validity-aware combination is the central rendering feature behind the Hybrid3D method: smooth continuous disparity and depth-ordered spatial structure are used together instead of forcing either method to handle the full image alone.

Hybrid3D also combines:

- raw normalized depth for tracking and stability
- shaped depth for stereo design
- subject-relative masks for local popout
- geometry-derived masks for edge safety and repair
- temporal controls for reducing shimmer and breathing
- post-warp cleanup for artifacts visible only after the eyes are generated

---

## Render Workflow

The active Hybrid3D video path follows this general sequence:

1. Decode and prepare the source frame and matching depth frame.
2. Temporally smooth and percentile-normalize the working depth map.
3. Estimate a tracked subject depth from normalized depth before stylized disparity shaping.
4. Optionally add rounded foreground curvature and quality-dependent RGB-guided depth refinement.
5. Create a separate shaped depth field for disparity design.
6. Sculpt the shaped field into a stronger cinematic window layout when the selected controls request it.
7. Build normalized near, mid, and far weighting bands and combine them into a continuous disparity field.
8. Anchor zero parallax and locally reduce subject disparity through Subject Plane Lock.
9. Apply tracked-subject convergence control.
10. Add a protected, subject-relative Close Popout influence to nearer details and objects.
11. Smooth shift noise in flat depth regions while preserving hard depth boundaries.
12. Apply temporally stabilized edge and occlusion suppression.
13. Limit frame-to-frame shift velocity, then apply the final shift EMA.
14. Generate both eyes with continuous inverse GPU warping.
15. Generate depth-ordered forward-warp eyes and blend their valid regions into the continuous eyes.
16. Suppress high-stress smear zones and construct one-sided repair and contour-protection masks.
17. Repair disocclusion strips with directional background fill.
18. Run post-eye-generation halo cleanup for strong close-popout and subject-lock cases.
19. Measure frame-edge pressure and drive the dynamic floating window.
20. Apply optional DOF, color processing, stereo formatting, audio attachment, and export.

A defining part of the method is the separation between **tracking depth**, **disparity-design depth**, **the final shift field**, and **repair analysis**. Each representation has a different purpose and is not treated as interchangeable.

---

## Core Depth Representations

| Representation | Primary Role |
| --- | --- |
| **Normalized tracking depth** | Subject estimation, convergence, focal tracking, and stable scene interpretation |
| **Shaped disparity depth** | Near/mid/far stereo design, cinematic depth sculpting, and depth-order decisions |
| **Per-pixel shift field** | Final horizontal stereo displacement for both eyes |
| **Edge and repair masks** | Occlusion suppression, contour protection, disocclusion fill, and halo cleanup |

This separation allows VD3D to stylize stereo depth without forcing subject tracking or repair logic to depend on the same altered depth values.

---

## Core Method Components

### 1. Temporal Depth Normalization and Subject Tracking

VD3D begins with a temporally smoothed working depth map and normalizes its active range through a percentile-based EMA process.

The subject is estimated from normalized depth before pop shaping or cinematic depth sculpting. For performance, large depth maps are reduced for the statistical subject estimate while the full-resolution depth remains available for rendering.

The estimate uses center weighting and depth-gradient weighting to reduce the influence of unstable borders and unrelated background regions. A separate subject-depth EMA stabilizes this tracked value across frames.

The tracked subject depth becomes a shared anchor for:

- zero-parallax placement
- Subject Plane Lock
- dynamic convergence
- Close Popout classification
- floating-window behavior
- focal-depth tracking

---

### 2. Foreground Curvature Enhancement

Before stereo shaping, VD3D can add subtle curvature to near or foreground regions.

The curvature stage:

- creates a soft foreground mask from depth
- estimates the weighted center and spread of the detected foreground
- builds an elliptical interior bump
- feathers the influence to avoid hard boundaries
- moves the interior slightly relative to its outer contour

This gives faces, bodies, and foreground objects more internal volume without requiring the entire scene to be reconstructed as explicit geometry.

---

### 3. Pop-Control Depth Shaping

The `shape_depth_for_pop` stage creates a dedicated disparity-design copy of the normalized depth field.

It performs:

1. percentile stretching of the active depth range
2. subject-aware recentering toward a configurable stereo midpoint
3. signed gamma reshaping around that midpoint

This stage changes how the available depth range is distributed for stereo rendering while leaving the original normalized depth available for tracking and stability decisions.

---

### 4. Cinematic Window Depth Sculpting

Hybrid3D can further reshape the disparity-design depth into a stronger **window into the scene** layout.

The sculpt is relative to the tracked subject plane and is designed to:

- push background regions farther behind the subject
- pull nearer foreground regions slightly forward
- preserve a protected depth band around the subject
- avoid shifting the hero subject as aggressively as the surrounding scene

The current implementation derives this effect from the active pop gamma, background push, foreground multiplier, and subject-lock controls. This produces stronger scene depth without simply multiplying every disparity value equally.

---

### 5. Structured Near / Mid / Far Disparity Weighting

VD3D does not use a single linear depth-to-shift equation for the whole frame.

Instead, it builds overlapping near, mid, and far weighting bands. A lightly smoothed copy of the shaped depth is used for these layer weights so pixels do not rapidly jump between stereo zones from frame to frame.

The bands are normalized and combined with independent foreground, midground, and background shift controls. Foreground and background multipliers can then emphasize popout or scene depth while the mid band maintains continuity between them.

This produces a continuous disparity field with layered spatial behavior rather than a set of hard cutout planes.

---

### 6. Subject-Aware Zero-Parallax Anchoring

The tracked subject is evaluated through the same near, mid, and far weighting model used to build the main shift field.

VD3D uses that weighted subject shift to calculate the zero-parallax correction. This keeps subject anchoring and scene disparity in the same mathematical space, reducing mismatch between:

- perceived subject placement
- the selected screen plane
- convergence behavior
- the surrounding stereo volume

The correction is smoothed and safety-clamped when floating-window control is active.

---

### 7. Local Subject Plane Lock and Silhouette Fill

Subject Plane Lock is a local disparity correction rather than a global movement of the entire stereo scene.

The stage:

1. builds a soft depth band around the tracked subject
2. applies center weighting to reduce accidental locking of unrelated background objects
3. expands and softly closes the subject region to fill face and body gaps
4. measures the subject's current average shift
5. subtracts that shift only within the eased subject mask

The advanced plane-lock strength and the simpler Subject Zero Lock control are combined without directly stacking both at full strength.

This allows the subject to remain closer to the screen plane while preserving most of the foreground and background depth around it.

---

### 8. Dynamic Convergence from the Tracked Subject Path

Dynamic convergence uses the temporally stabilized subject estimate instead of a separately interpreted shaped-depth value.

The convergence bias is smoothed, given a backend gain appropriate to the active floating-window mode, converted into normalized shift units, and clamped for safety.

Using the tracked subject path keeps convergence, subject locking, and zero-parallax placement semantically aligned.

---

### 9. Subject-Relative Close Popout

The v5 renderer adds a dedicated **Close Subject Popout** stage for details that lie closer than the tracked subject plane, such as hands, tools, faces, weapons, and other foreground objects.

This is a local detail and shape booster, not a second global 3D-strength multiplier.

Its safety system includes:

- a subject-relative activation threshold
- a configurable popout range
- horizontal and vertical frame-edge falloff
- depth-edge stress protection
- soft mask feathering
- nonlinear suppression of weak mask noise
- a local extra-shift cap
- independence from the main foreground pop multiplier

The temporal mask uses **fast attack and slow release**. New close regions can respond immediately, while disappearing regions fade more gradually. This prevents repeated preview refreshes from slowly building the effect and reduces mask flicker during motion.

Close Popout is applied after the normal global shift has been clamped. Only protected local regions are allowed a small amount of additional disparity beyond the normal cap.

---

### 10. Edge-Stress Analysis and Flat-Region Shift Smoothing

VD3D derives a soft depth-edge stress map from horizontal and vertical depth gradients.

The stress map distinguishes:

- flatter regions where small shift noise can be safely smoothed
- hard depth boundaries where smoothing could smear a silhouette

A local average is blended into the shift field primarily in flat regions. This reduces shimmer and breathing inside foreground, midground, and background surfaces while preserving stronger object boundaries.

The same edge-stress information is reused by Close Popout protection, occlusion masking, and the temporal shift limiter.

---

### 11. Temporally Stabilized Edge and Occlusion Suppression

Hybrid3D builds an occlusion-risk mask from shift gradients and supplements it with depth-edge stress when the shift field alone under-reports a silhouette.

A small safety layer remains active whenever edge masking is enabled, even if the general feather control is set very low.

The occlusion mask has its own EMA. Stabilizing the mask itself prevents the protection zone from repeatedly appearing and disappearing around moving contours, which directly reduces edge breathing.

The final shift is blended toward an edge-suppressed version only where this mask indicates risk.

---

### 12. Temporal Shift Velocity Limiting

Before the final shift EMA, VD3D limits how quickly the per-pixel disparity field can change between frames.

This stage is distinct from ordinary smoothing:

- the **velocity limiter** caps sudden frame-to-frame jumps
- the **EMA** smooths the motion that remains

The allowed shift delta is reduced near hard depth edges and remains more responsive in flatter regions. This prevents noisy depth changes from pumping stereo strength at silhouettes while still allowing genuine scene motion to progress.

Together with normalized depth smoothing, subject tracking, mask EMA, and convergence smoothing, this forms a multi-layer anti-breathing system.

---

### 13. Continuous Inverse GPU Eye Warp

The first eye-generation path uses a cached normalized sampling grid and per-pixel horizontal shift values.

The left and right eyes are sampled in opposite directions through GPU `grid_sample` operations. This provides:

- continuous depth-weighted stereo movement
- smooth subpixel resampling
- efficient GPU execution
- compatibility with HDR, VR180, and multiple stereo formats

This path remains the stable base image for the Hybrid3D blend.

---

### 14. Depth-Ordered Forward Warp

The second eye-generation path forward-projects source pixels into each target eye.

For each eye, VD3D:

1. converts normalized shift values into pixel destinations
2. sorts pixels by shaped depth order
3. writes farther surfaces first
4. writes nearer surfaces afterward so foreground can occupy the visible target location
5. fills remaining holes horizontally from the appropriate direction
6. returns both the eye image and a validity mask

This helps preserve foreground-over-background ordering and can create stronger, more stable layer separation than inverse sampling alone.

---

### 15. Validity-Aware Hybrid Warp Blending

The depth-ordered result is not blended uniformly over the continuous eye warp.

Unresolved forward-warp holes can contain source-frame fallback pixels. Mixing those fallback pixels indiscriminately can produce bright or dark contour halos, especially around close subjects.

VD3D therefore multiplies the forward-warp blend by the path's validity mask. Only pixels genuinely owned by the depth-order projection are mixed into the continuous eyes.

The blend amount and horizontal fill effort are adaptive:

- **Off, Fast, Balanced, High, and Showcase** repair modes use different depth-order strengths and fill limits
- high-resolution outputs reduce the maximum fill work and blend level to control render cost

This validity-aware fusion is the central stereo synthesis step of Hybrid3D.

---

### 16. Smear Suppression and One-Sided Contour Repair

After eye generation, VD3D detects high shift-gradient zones that are likely to stretch or smear. These zones are selectively blended back toward the original frame before the main repair pass.

The repair system then builds separate masks for each eye from:

- signed horizontal shift gradients
- warp validity loss
- the expected trailing side of each stereo displacement
- a foreground contour-protection barrier

Repair is directional. Background pixels are pulled horizontally into newly exposed strips from the side appropriate to each eye.

The contour barrier prevents repair from crossing the preserved foreground edge, while a softened safety buffer reduces repair collision with hairlines, shoulders, faces, and thin objects.

---

### 17. Quality-Adaptive Disocclusion Repair

The edge-repair quality controls change both repair accuracy and runtime behavior.

- **Off** disables the repair path.
- **Fast** and **Balanced** use a lightweight directional fill and reduced mask strength.
- **High** and **Showcase** use the fuller silhouette-safe repair path with larger fill reach and stronger protection.

Heavy RGB-guided depth refinement is also reserved for High and Showcase modes. This prevents the most expensive edge-aware preprocessing from running on every normal video frame.

The result is a repair system that can scale from practical everyday rendering to slower showcase-oriented cleanup.

---

### 18. Post-Warp Close-Popout Halo Cleanup

Some artifacts only become visible after the final eyes have been generated and blended. Close faces, strong subject lock, and local popout can expose wider horizontal strips than the standard repair mask captures.

VD3D therefore runs a second post-eye-generation cleanup pass.

This stage:

- measures the actual final shift magnitude
- builds one-sided edge seeds for the current eye
- combines shift-gradient and depth-edge stress cues
- expands the repair area primarily in the horizontal direction
- preserves the hard foreground core
- directionally fills the exposed halo strip
- applies slightly stronger correction to the right eye where the current path can show more visible tearing

Because this pass operates on the completed eye image, it targets the artifact zone created by the final warp rather than modifying the source frame pre-emptively.

---

### 19. Measured Frame-Edge Violation and Dynamic Floating Window

VD3D measures likely window pressure directly from the final shift field near the left and right edges of the frame.

The floating-window controller combines:

- measured left and right edge violation
- zero-parallax magnitude
- subject depth distance from the midpoint
- a maximum bar-width safety limit
- temporal easing of bar width and side selection

The side with greater measured pressure is preferred. When neither side dominates, the zero-parallax direction is used as the fallback.

This ties the floating window to actual stereo risk at the image boundaries instead of relying only on an abstract convergence value.

---

### 20. Motion-Aware Focal Depth and Depth of Field

The DOF system tracks a candidate focal depth and adjusts its responsiveness from scene motion.

Calmer shots use a steadier focal response. More active shots allow faster adaptation. A deadband and maximum step limit reduce unnecessary focus drift.

The DOF mask is feathered to prevent a hard cutout boundary between the focused subject and the blurred background.

---

### 21. Optional Matte Sculpting

VD3D can load per-frame subject mattes and use distance transforms to create rounded subject depth.

The matte path provides:

- temporally stabilized masks
- feathered subject blending
- adjustable near and far subject values
- rounded center-to-edge depth falloff

The GPU-to-CPU depth transfer is avoided unless a matte file actually exists for the current frame, preserving performance when the feature is not in use.

---

### 22. Debuggable Stereo Telemetry and Profiling

The render path exposes internal values for engineering and tuning, including:

- tracked subject depth
- zero-parallax and screen-plane offsets
- Subject Plane Lock values
- Close Popout state and parameters
- measured left and right edge violation
- repair and protect mask statistics in debug mode
- forward-warp validity statistics

A render-stage profiler can also report average per-frame timing for major pipeline stages when debugging is enabled.

This makes Hybrid3D tunable as an observable rendering system rather than only through visual trial and error.

---

## Still-Image Scene-Variance Scaling

The still-image render path can scale foreground, midground, and background shift values from normalized depth variance in the central image region.

This lets a single image open up more strongly when it contains clear spatial separation and remain gentler when its depth field is comparatively flat.

The main video loop currently keeps this dynamic scale neutral and relies on its temporal stabilization, user-selected shift values, and scene/keyframe settings instead.

---

## Supporting Systems

The Hybrid3D method is integrated with the wider VisionDepth3D render stack, including:

- HDR10 RGB48/PQ decode and P010 encode paths
- source HDR metadata handling for compatible x265 export
- GPU color grading and sharpening
- automatic black-bar detection and aspect preservation
- cached GPU sampling grids
- VR180 flat-to-equirect projection
- Half-SBS, Full-SBS, top-bottom, anaglyph, passive, and VR output workflows
- FFmpeg-based video encoding and source-audio attachment
- audio copy and AAC encode modes
- software, NVENC, AMF, and QSV encoder support
- CUDA, ROCm, DirectML, MPS, and CPU device selection paths

---

## Summary Table

| Category | Component |
| --- | --- |
| **Depth Interpretation** | Temporal and percentile depth normalization |
|  | Subject estimation from pre-shaped normalized depth |
|  | Foreground curvature enhancement |
|  | Pop-control depth shaping |
|  | Cinematic window depth sculpting |
| **Stereo Structure** | Near / mid / far disparity weighting |
|  | Subject-aware zero-parallax anchoring |
|  | Local Subject Plane Lock with silhouette fill |
|  | Dynamic convergence from tracked subject depth |
|  | Subject-relative Close Popout |
| **Temporal Stability** | Flat-region shift smoothing |
|  | Edge and occlusion mask EMA |
|  | Per-pixel shift velocity limiting |
|  | Final shift EMA |
| **Hybrid Eye Generation** | Continuous inverse GPU warp |
|  | Depth-ordered forward warp |
|  | Validity-aware hybrid blending |
| **Artifact Control** | Smear suppression |
|  | One-sided repair and contour protection |
|  | Quality-adaptive directional disocclusion fill |
|  | Post-warp close-popout halo cleanup |
| **Presentation** | Measured dynamic floating window |
|  | Motion-aware DOF |
|  | Matte depth sculpting |
|  | HDR, VR180, and multi-format export |
| **Engineering** | Stereo telemetry and render-stage profiling |

---

## Closing

The VD3D Hybrid3D Method is a unified 2D-to-3D rendering architecture in which depth interpretation, subject anchoring, local popout, continuous stereo displacement, depth-ordered projection, temporal stability, contour protection, post-warp repair, and presentation safety are designed to operate together.

Its defining characteristics are:

- **hybrid eye generation** rather than a single warp strategy
- **subject-relative stereo control** rather than only global depth strength
- **continuous depth structure** rather than hard cutout layers
- **depth-order awareness** for foreground-over-background placement
- **multi-stage temporal stabilization** to reduce breathing and shimmer
- **geometry-aware and post-warp repair** for difficult contours
- **adaptive performance controls** across quality levels and output resolutions
- **direct telemetry** for iterative engineering and tuning

The result is a controllable stereo method that can create strong spatial separation while maintaining subject stability, edge safety, and practical performance across a wide range of source material.

📄 Licensed under: VisionDepth3D Custom Use License (No Derivatives)  
🔗 Project: [https://github.com/VisionDepth/VisionDepth3D](https://github.com/VisionDepth/VisionDepth3D)
