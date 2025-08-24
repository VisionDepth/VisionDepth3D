
# VisionDepth3D v3.5 – Changelog

## 1) Depth of Field (DOF) – Rewritten & Stabilized
- Fully rewritten as a **GPU-accelerated, multi-level Gaussian pipeline**.
- Uses **per-pixel interpolation** between blur levels for smooth transitions.
- Added **motion-adaptive focal tracking**:
  - Exponential moving average (EMA) for stable focus.
  - Deadband to ignore micro noise.
  - Max-step limiter to prevent “focus pops.”
- DOF now applies **after stereo warp** using warped per-eye depth.
- DOF slider maps directly to `max blur`; setting it to `0` cleanly disables DOF.
- **Result:** smoother bokeh, no ring artifacts, and much more natural focal transitions.

## 2) Audio Tool – Revamp & Codec Control
- Added **progress bar** for encoding/attaching audio.
- Users can now select **codec and bitrate** before muxing:
  - `aac` (default) and `libmp3lame` supported.
  - Configurable bitrate (e.g. 128k, 192k, 320k).
- **Offset slider** added for real-time sync adjustment when attaching audio.
- Audio attachment now clearly distinguishes between **copy vs. re-encode**:
  - If codec/bitrate unchanged → fast copy (`-c copy`).
  - If codec/bitrate changed → re-encode.
- UI fields now properly populate when files are chosen.
- Safe handling of long videos (2+ hours) with progress feedback.

## 3) Color Grading – GPU Accelerated & Fully Integrated
- Introduced **GPU-accelerated color grading pipeline** (`apply_color_grade`) with:
  - **Saturation**
  - **Contrast**
  - **Brightness**
- Color grading now applies **after stereo warp & DOF, before packing/formatting**.
- Added **Preview GUI sync**:
  - Sliders update live in the preview with **debounced re-rendering**.
  - Two-way binding with main UI — values set in Preview transfer to main UI controls and vice versa.
- Preset/save/load support extended to include color grading.
- Tooltips and i18n refreshed for new controls.
- **Result:** creators can now fine-tune the image directly inside VD3D without round-tripping into external grading tools.

## 4) Stereo Separation (IPD Adjustment) – New 3D Control
- Added **Interpupillary Distance (IPD) adjustment slider** for fine-tuning stereo separation.
- Works as a **global scale factor** on pixel shifts (foreground, midground, background).
- Allows creators to:
  - Increase IPD for stronger 3D “pop” on large screens / VR headsets.
  - Reduce IPD for comfortable viewing on smaller displays.
- Fully integrated into:
  - **Preset system** (save/load).
  - **Preview GUI** with real-time feedback.
  - **Tooltip and i18n system** for clarity.
- **Result:** users can now match stereo depth strength to their **display environment and audience comfort**.

## 5) General Fixes & Stability
- Fixed tensor size mismatch crash in DOF when depth/resolution didn’t match warp output.
- Preview GUI sliders now wire correctly to main GUI sliders for seamless testing.
- Minor UI consistency fixes across tools.
- **Language Files Clean-Up:**  
  - Removed duplicate keys and aligned all translations with `en.json`.  
  - Verified **FR/DE/ES/JA** language packs — all tooltips and UI labels now update correctly when switching languages.  
  - Added missing entries (`Apply Entries`, `Start Batch Render`, scene detection, etc.) to ensure full coverage.

## 6) New Session Additions – ONNX Pipeline & UI Enhancements
- **ONNX Integration**
  - Converted **Video Depth Anything (pth → onnx)** for faster inference.
  - Optimized ONNX pipeline path to run converted models efficiently.
- **UI Enhancements**
  - New **start time / end time controls** inside Encoding Settings:
    - Users can render **short clips or preview segments** without full video runs.
  - Inputs section refactored into its own **dedicated frame** for clarity.
- **Result:** streamlined workflow for experimenting with models, and flexible render ranges for testing.

---

**Summary:**  
v3.4 gave creators fine-grained depth & subject control.  
v3.5 brings **cinematic polish** with stabilized DOF, a **true audio tool** with sync + codec options, a **GPU color grading suite**, a **stereo separation (IPD) adjustment** for display comfort, and now **ONNX pipeline + clip rendering support** for faster experimentation.
