# The VisionDepth3D Method  

An advanced, real-time stereo rendering engine for 2D-to-3D conversion in VR and stereoscopic displays.  
VisionDepth3D introduces multiple **first-of-their-kind techniques** in the 3D conversion community, while also integrating established practices into a single GPU-optimized pipeline.

---

## 🚀 Original Innovations (Introduced by VisionDepth3D)

### 1. Depth Shaping for “Pop Control”  
* The **`shape_depth_for_pop` algorithm** is unique to VisionDepth3D:  
  1. Percentile stretch of depth range.  
  2. Recentered on subject depth.  
  3. Symmetric gamma curve for controlled “pop”.  
* Enables tunable cinematic or VR-style stereo with consistent subject emphasis.  

---

### 2. Subject-Aware Zero-Parallax Plane with EMA Stabilization  
* Dynamically estimates subject depth (center-weighted histograms + percentiles).  
* Locks the zero-parallax plane to the subject with exponential smoothing.  
* Prevents subject “drift” and stabilizes the screen plane without manual keyframing.  

---

### 3. Dynamic Parallax Scaling by Scene Variance  
* Computes normalized variance of depth in the central region:  
```python
parallax_scale = compute_dynamic_parallax_scale(depth_tensor)
```  
* Adapts stereo strength automatically — gentle for flat shots, expansive for landscapes.  

---

### 4. Edge-Aware Shift Suppression with Gradient Sigmoid Masking  
* Novel use of depth gradients to suppress parallax shifts near thin, detailed edges:  
```python
edge_mask = torch.sigmoid((grad_mag - edge_threshold) * feather_strength * 5)
```  
* Prevents ghosting and halos without AI inpainting.  

---

### 5. Matte Sculpting with Temporal Stabilization  
* Combines distance transforms with EMA smoothing to “round” depth on subjects.  
* Prevents shimmer in hair, fingers, and soft edges.  
* Creates natural curvature without requiring heavy segmentation pipelines.  

---

### 6. Floating Window with Temporal Easing  
* Subject-aware floating window automatically detects window violations.  
* `FloatingWindowTracker` + `FloatingBarEaser` smooth jitter and clamp drift.  
* First real-time system that applies cinematic floating windows automatically.  

---

### 7. Motion-Aware Focal Depth Tracking for DOF  
* Depth-of-field blur controlled by a **focal depth tracker** that adapts to scene motion.  
* Busy shots → faster focus shifts; still shots → stable focus.  
* Simulates real cinematography rules dynamically during 3D rendering.  

---

### 8. Gradient-Based Healing of Stereo Occlusion Gaps  
* Detects warping gaps via gradient magnitude and fills with blended original + blurred content.  
* Lightweight alternative to neural inpainting — seamless and invisible.  

---

## ⚙️ Supporting Features (Implemented, Not Unique)

* **Depth-weighted continuous parallax shifting** – smooth stereo gradients instead of discrete layers.  
* **GPU tensor grid warping** – CUDA-optimized `grid_sample` per-eye rendering.  
* **Scene-aware dampening** – adjusts disparity for flat vs. complex scenes.  
* **Temporal percentile EMA normalization** – stabilizes depth scale across frames.  
* **Depth-based DOF (multi-level Gaussian pyramid)** – established technique, enhanced in VD3D with motion-aware focus.  
* **Black bar detection & aspect handling** – auto-crop and cinematic aspect preservation.  
* **Color grading & sharpening** – GPU-accelerated saturation, contrast, brightness, and safe sharpening.  
* **Multi-format 3D output** – Half-SBS, Full-SBS, VR, anaglyph, interlaced.  
* **FFmpeg streaming codec pipeline** – CPU (libx264/x265/AV1), NVIDIA NVENC, AMD AMF, Intel QSV with CRF/CQ control.  

---

## ✅ Summary Table

| Category             | Component                                         |
| -------------------- | ------------------------------------------------- |
| **Original**         | Pop Control Depth Shaping                         |
|                      | Subject-Aware Zero-Parallax Plane (EMA stabilized)|
|                      | Dynamic Parallax Scaling by Scene Variance        |
|                      | Edge-Aware Gradient Shift Suppression             |
|                      | Matte Sculpting + Temporal Stabilization          |
|                      | Floating Window with Temporal Easing              |
|                      | Motion-Aware DOF Focal Tracking                   |
|                      | Gradient-Based Healing of Occlusion Gaps          |
| **Supporting**       | Depth-weighted Parallax Shifting                  |
|                      | Temporal Percentile EMA Depth Normalization       |
|                      | GPU Tensor Grid Warping                           |
|                      | Scene-Aware Dampening                             |
|                      | DOF via Gaussian Pyramids                         |
|                      | Aspect Handling + Black Bar Detection             |
|                      | GPU Color Grading + Sharpening                    |
|                      | Multi-Codec Pipeline + Multi-Format Export        |

---

VisionDepth3D combines these original contributions with proven practices to form a **holistic, real-time, GPU-accelerated 2D→3D pipeline**.  

📄 Licensed under: VisionDepth3D Custom Use License (No Derivatives)  
🔗 Project: [https://github.com/VisionDepth/VisionDepth3D](https://github.com/VisionDepth/VisionDepth3D)  
