# VisionDepth3D v3.2.6 – Summary

- **Batch & Single Video 3D Rendering**
  - Added dual input modes: single video or batch of video/depth map pairs.
  - Dynamic UI updates to show relevant fields for each mode.
  - Improved label system and streamlined 3D render logic for both workflows.

- **IPD Factor for Parallax Scaling**
  - Introduced `ipd_factor` to control stereo separation intensity.
  - Applies uniform scaling to foreground, midground, and background shift values.
  - Synthetic (non-metric) adjustment for aesthetic tuning.

- **Audio Offset Support in Audio Attacher**
  - Added GUI slider to set custom audio offset (positive or negative).
  - Updated FFmpeg integration to shift audio relative to video duration.

- **Japanese Language Support**
  - Added complete Japanese localization.
