## VisionDepth3D v4.2.2 Changelog

### Depth Anything V2 Performance Fix

- Fixed a major DA-V2 performance issue when using the `Original` inference resolution.
- DA-V2 no longer treats `Original` as full-resolution model inference by default.
- `Original` now preserves the source aspect ratio while capping the DA-V2 model input size to a safer default.
- This prevents accidental 1080p/4K DA-V2 inference, which could severely reduce render speed.
- Added environment override for users who intentionally want true original-resolution DA-V2 inference:

```bash
VD3D_DAV2_DEFAULT_MAX_SIDE=0
```

- Default DA-V2 max side is now:

```bash
VD3D_DAV2_DEFAULT_MAX_SIDE=518
```

---

### DA-V2 Adapter Optimization

- Reduced unnecessary DA-V2 output resizing inside the adapter.
- DA-V2 now returns model-resolution depth output and lets the main VisionDepth3D postprocess handle final video/image resizing.
- Removed duplicate DA-V2 normalization before main pipeline normalization.
- Reduced GPU work and GPU-to-CPU transfer size during DA-V2 video rendering.
- Improved DA-V2 handling for `Original` resolution while keeping aspect ratio.

---

### Diagnostics

- Added clearer DA-V2 load diagnostics showing:
  - selected weight file
  - backend
  - device
  - dtype
  - FP16 state
- Added main load-setting diagnostics showing:
  - selected model
  - checkpoint
  - UI FP16 state
  - resolved FP16 state
  - active backend
  - torch device

---

### Notes

- CUDA FP16 still requires enabling FP16 before loading/reloading the model.
- DA-V2 `Original` mode is now optimized for practical render speed by default.
- True original-resolution DA-V2 inference remains available through the environment override, but may be very slow on high-resolution video.
