# VisionDepth3D v5.3

**Unreleased · CUDA build**

Live 3D gains Half-SBS output for compatible displays and players. The Depth Engine can load supported local models and process a folder of videos. The Hybrid 3D Engine gives you finer control over where subjects sit relative to the screen.

## Live 3D

- **Half-SBS output (#134).** Select **Live 3D → Preview / Output → Half-SBS output** before starting a session. Preview and stream use the selected layout. Full-SBS is the default.

## Depth Engine

- **Video folder processing.** Fixed a folder being rejected with “Input video does not exist.” Supported clips are processed in natural filename order. Each gets its own output name, and a failure identifies the affected clip.
- **Output filenames.** Set a custom name for a single render or keep the source-based default. Existing outputs are protected with numbered suffixes such as `_2` and `_3`.
- **Local model import (Pro).** Load supported Depth Anything V2 checkpoints, compatible ONNX files, and supported model folders. Refresh Models rescans available models. Open Weights Folder takes you to the weights directory. The native DA-V2 Large checkpoint was verified through the import workflow.
- **Model readiness and preview (#121).** Generate Preview and Start Processing wait for the selected model to finish loading. ZoeDepth NK now reports a failed warm-up and was verified to generate a preview after a successful load. Clearing the model selection no longer opens a Pro upgrade prompt.
- **Thin letterbox detection.** The automatic detector recognizes narrow, matching black bars at the top and bottom of a video. It detected 20-pixel bars on a 1080p test clip during normal processing.

## Hybrid 3D Engine

- **Sharpening and brightness (#124).** Corrected an SDR sharpening filter that brightened flat areas. Zero sharpness now bypasses preview sharpening. Preview and export use matching sharpening behavior.
- **Stereo depth controls.** Corrected foreground curvature and Close Pop-Out direction. Manual Screen Depth works with tracking disabled. Pixel-based adjustments scale with render width, and zero displacement produces matching eye views.
- **3D Assistant and presets.** Named styles coordinate Close Pop-Out and advanced stereo settings. Custom adjustments use the saved slider baseline. Slider readouts and renderer settings stay synchronized when preset values reach control limits. Reselect a named style to apply its updated settings.
- **Render controls.** Inputs & Render is organized into tabs. Close Pop-Out exposes amount, threshold, range, and border safety.
- **SBS render diagnostics (#132).** Input, frame-decoding, range, and encoder failures now report the step that failed. A render crash can no longer be masked by returning a filename during cleanup. Completion is reported after encoder finalization. The original reporter’s failure still needs confirmation.

## Depth Blender

- **Legacy presets.** The preset picker is available from Edge-Aware mode. Choosing a preset switches to Legacy, applies the settings, and refreshes the preview. Selecting the same preset again restores its values after manual edits.
- **Edge-Aware blending.** Refined Edge Protection where the two maps disagree near a silhouette. Testing showed a smaller halo on a synthetic offset edge. A visible outline persists on a real test frame and remains under investigation.

## Build coverage

These notes describe the CUDA release candidate. Depth-output naming and overwrite protection have also been ported to DirectML. The remaining DirectML changes, cross-build validation, and release packaging are pending.
