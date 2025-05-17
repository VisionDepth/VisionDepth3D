# The VisionDepth3D Method 

An advanced, real-time stereo rendering engine for 2D-to-3D conversion in VR and stereoscopic displays.

## ✅ Core Innovations (Proprietary to VisionDepth3D)

### 1. Depth-Weighted Continuous Parallax Shifting

* Avoids depth slicing or zone segmentation.
* Uses soft weights from the depth map to mix foreground, midground, and background contributions:

```python
raw_shift = (fg_weight * fg_shift +
             mg_weight * mg_shift +
             bg_weight * bg_shift)
```

* This enables smooth and natural parallax gradients across the image with no seams or pop-ins.

---

### 2. Subject-Aware Zero-Parallax Plane Tracking

* Calculates the convergence plane dynamically by analyzing the mode of a center-weighted histogram from the depth map:

```python
subject_depth = estimate_subject_depth(depth_tensor)
```

* Produces a subject-anchored stereo window to keep the scene's focus point at screen depth.

* Final zero-parallax shift is calculated as:

```python
zero_parallax_offset = ((-subject_depth * fg) + (-subject_depth * mg) + (subject_depth * bg)) / (resized_width / 2)
```

* Replaces the original convergence formula, improving tracking accuracy.

---

### 3. Edge-Aware Shift Masking (No Inpainting Required)

* Uses gradient-based masking to suppress parallax near high-contrast edges:

```python
edge_mask = torch.sigmoid((grad_mag - edge_threshold) * feather_strength * 5)
smooth_mask = 1.0 - edge_mask
total_shift = total_shift * smooth_mask
```

* Prevents hard ghosting or edge bleed with no pre-processing.

---

### 4. Floating Window Stabilization

* Smooths convergence shifts over time with adaptive momentum tracking:

```python
zero_parallax_offset = floating_window_tracker.smooth_offset(zero_parallax_offset)
```

* Clamps offset within bounds and applies side masking for viewer comfort.

---

### 5. Scene-Aware Parallax Dampening

* Dynamically adjusts stereo strength based on scene flatness:

```python
parallax_scale = compute_dynamic_parallax_scale(depth_tensor)
```

* Ensures stable 3D across both action scenes and low-contrast shots.

---

### 6. Real-Time GPU-Optimized Pixel Warping

* CUDA-accelerated `grid_sample` warping of the frame based on shift values:

```python
grid_left[..., 0] += shift_vals
grid_right[..., 0] -= shift_vals
```

* Full left/right stereo generation without latency using PyTorch tensor operations.

---

### 7. Depth-Based DOF and Healing

* Adaptive DOF: Multiple Gaussian blurred versions composited using per-pixel depth weight:

```python
blur_idx = blur_weights * (len(levels) - 1)
```

* Pixel Healing: Smart fill of stereo occlusion zones using gradient-based mask blending.

---

## ✅ Summary Table

| Component                       | Description                                         |
| ------------------------------- | --------------------------------------------------- |
| Depth-Weighted Shift            | Smooth pixel displacement without discrete zones    |
| Subject-Aware Parallax Tracking | Dynamically centers stereo plane on dominant object |
| Edge-Aware Feathering           | Prevents ghosting without segmentation              |
| Floating Window Tracker         | Stabilizes convergence across frames                |
| Scene-Aware Parallax Dampening  | Auto-tunes 3D intensity by depth stats              |
| DOF + Healing                   | Simulated depth of field with gap healing           |
| GPU Tensor Grid Warping         | High-performance stereo image warping               |

---

* All features above are **original** to VisionDepth3D.
* Designed and optimized in-house for real-time VR stereo authoring.
* No external segmentation, warping, or AI fill-inpainting required.

For citations or inquiries:
[https://github.com/VisionDepth/VisionDepth3D](https://github.com/VisionDepth/VisionDepth3D)

📄 Licensed under: VisionDepth3D Custom Use License (No Derivatives)
