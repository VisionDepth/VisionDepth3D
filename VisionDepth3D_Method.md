# The VisionDepth3D Method

An original real-time stereo rendering framework for VR-ready 2D-to-3D conversion

## Overview

The VisionDepth3D Method is a novel algorithmic approach to generating stereoscopic 3D from 2D
video using AI depth estimation. It was developed independently as part of the VisionDepth3D
project and is specifically designed to produce natural, comfortable depth for VR and stereoscopic
displays - without requiring segmentation masks, inpainting, or warping.

## Core Innovations

1. **Depth-Weighted Parallax Shifting**
Rather than segmenting a frame into discrete depth zones, this method uses continuous depth
blending to determine pixel shift:

    `raw_shift = fg_weight * fg_shift + mg_weight * mg_shift + bg_weight * bg_shift`

- Each weight is derived dynamically from the normalized depth map.
- This creates smooth parallax transitions across the entire frame.
- Prevents visible seams or pop artifacts at depth boundaries.

2. **Subject-Aware Zero-Parallax Tracking**
The method dynamically calculates the convergence plane using the histogram mode of the depth
map's center region. This provides a subject-centered stereo anchor, reducing eye strain:

    `zero_parallax_offset = (-adjusted_depth * fg_shift + -adjusted_depth * mg_shift + adjusted_depth * bg_shift) / width - convergence_offset`

This allows dynamic parallax tuning that responds to scene motion.

3. **Edge-Aware Shift Masking (No Inpainting Required)**

To suppress stereo artifacts near depth discontinuities (e.g., hair, edges), the method uses a
gradient-based edge mask derived from the depth map:

    `edge_mask = sigmoid(gradient_magnitude - threshold)`

This mask blends between shifted and original pixels, preserving clean outlines without the need for
warping, segmentation, or generative fill.

4. **Scene-Aware Parallax Dampening**

The system calculates the variance of the scene's center-depth to modulate stereo intensity. Flat
scenes automatically receive lower parallax values to prevent false depth amplification:

    `parallax_scale = min(1.0, depth_variance * sensitivity)`

This provides a more consistent experience across shot types and lighting conditions.

## Summary

The VisionDepth3D Method combines:
Component                         | Function
----------------------------------|-----------------------------------------------------------
`Depth-Weighted Pixel Shift`       | Smooth, gradient-based 3D parallax
`Subject-Based Convergence`        | Dynamic, zero-parallax plane adjustment
`Gradient Edge Masking`            | Stereo cleanup without external masks or AI inpainting
`Parallax Scaling via Depth Stats` | Adaptive stereo strength per shot

- The VisionDepth3D Method is original to the VisionDepth3D project.
- Designed and optimized for real-time VR stereo authoring.
- No external segmentation, warping, or masking required.

For inquiries or citations, please reference:
https://github.com/VisionDepth/VisionDepth3D

📄 Licensed under: VisionDepth3D Custom Use License (No Derivatives)

