# VisionDepth3D v5.0 Changelog

## Free/Pro Release Foundation, 3D Pipeline Improvements, Depth Engine Expansion, and Release Build Hardening

VisionDepth3D v5.0 is a major stability, licensing, depth-estimation, and 3D-rendering update. This release lays the foundation for the new Free and Pro tier system while improving the 3D conversion pipeline, depth model handling, stereo controls, watermark workflow, model licensing safety, and release packaging.

This update focuses on making VisionDepth3D more reliable as a full desktop 2D-to-3D suite, with better Free/Pro behavior, cleaner model support, improved depth processing, stronger 3D pipeline control, and a more production-ready build process.

---

## Free and Pro Modes

VisionDepth3D v5.0 introduces a new Free and Pro structure.

The Free mode is designed to give users access to the core VisionDepth3D experience so they can test the workflow, generate shorter 3D conversions, try supported depth tools, and evaluate the software before upgrading.

The Pro mode is designed for creators who want the full production workflow, including longer renders, watermark-free exports, batch processing, advanced stereo controls, keyframe workflows, and faster depth-generation options.

### Free Mode

Free mode includes the core VisionDepth3D conversion workflow with limits intended for testing, previewing, and evaluating the software.

Free mode includes:

* Core 2D-to-3D conversion workflow.
* Shorter video conversion.
* Standard depth-generation options.
* Basic stereo/depth controls.
* Access to supported Free-mode depth models.
* Watermarked exports.
* Upgrade prompts for Pro-only tools.

### Pro Mode

Pro mode unlocks the full VisionDepth3D creator workflow.

Pro mode includes:

* Longer video renders.
* Watermark-free exports.
* Batch video processing.
* Advanced stereo and depth controls.
* Advanced keyframe workflows.
* Faster Pro depth-generation options.
* Pro model access where licensing allows.
* Full production-focused conversion workflow.
* Unlocked Depth Blender processing.
* Unlocked Live3D Viewer.

The Free and Pro structure helps support continued VisionDepth3D development, including model testing, performance optimization, hardware compatibility work, AMD / Intel / NVIDIA backend testing, future Mac and Linux testing, and long-term maintenance.

---

## License Activation

VisionDepth3D v5.0 adds Pro license activation support.

Users can activate Pro mode with a valid license key, and VisionDepth3D will automatically load the correct Free or Pro mode on startup.

If no valid license is active, VisionDepth3D runs in Free mode. If a valid Pro license is active, Pro features become available without needing a separate installer.

### Offline Pro License Cache

VisionDepth3D now supports cached Pro license access for offline use.

After a successful Pro activation, the activated license is saved to the user AppData license location and can be loaded on startup without requiring an internet connection every time the app opens.

Offline Pro cache behavior includes:

* Local AppData license loading on startup.
* Cached Pro access after successful activation.
* Offline startup support for activated Pro users.
* Startup license logs showing the license path, cached license state, and active tier decision.
* Lemon Squeezy validation remaining part of the online activation and refresh flow.
* No separate Pro installer required.

## 3D Pipeline Updates

VisionDepth3D v5.0 includes several 3D pipeline improvements focused on stereo stability, subject separation, depth pop, background pushback, Depth of Field behavior, style persistence, and cleaner Free/Pro behavior.

### 3D Style and Assistant Updates

* Fixed 3D Style saving and restore behavior.
* Added persistent 3D Assistant state saving.
* Added saved values for:

  * selected 3D Style
  * simple 3D strength
  * pop strength
  * comfort
  * stability
  * screen depth
  * subject plane control

* Fixed startup behavior so saved 3D values reload correctly.
* Prevented 3D Style presets from overwriting unrelated processing toggles.
* Separated stereo/depth style changes from processing workflow settings.
* Prevented simple 3D controls from changing advanced processing options such as:

  * subject tracking
  * dynamic convergence
  * edge masking
  * feathering
  * shift EMA behavior
  * floating window state
  * edge repair quality

* Improved “Custom” style behavior so manual tuning is preserved.
* Added close-event saving so current 3D UI values are written before shutdown.
* Improved consistency between simple 3D controls, advanced controls, app state, and saved settings.

### Advanced Stereo Control Cleanup

* Cleaned up how simple and advanced 3D presets interact.
* Kept simple 3D presets focused on stereo/depth appearance only.
* Preserved advanced processing toggles independently from style selection.
* Improved consistency between UI sliders, app state, and saved settings.
* Reduced unexpected jumps when switching styles or reopening the app.
* Improved behavior around manual 3D tuning so custom values are not unintentionally overwritten.
* Improved separation between “style” controls and “render processing” controls.

### Depth Pop and Background Pushback

* Improved the 3D depth-pop workflow for stronger subject/background separation.
* Preserved the cinematic “window into the scene” look where backgrounds appear pushed back from the subject.
* Improved background separation behavior without relying on blur alone.
* Continued tuning around depth-order rendering, foreground placement, and background pullback.
* Improved the balance between subject depth, mid-ground depth, and background depth.
* Kept Depth Pop / Background Push behavior separate from Depth of Field blur so users can control 3D separation and cinematic softness independently.

### Depth of Field Improvements

* Reworked Depth of Field behavior so it is safer for 2D-to-3D conversion.
* Changed DoF direction toward background-only blur instead of full depth-plane blur.
* Improved subject protection so faces, characters, foreground objects, and important near-depth areas stay sharper.
* Reduced cases where DoF accidentally softens the subject.
* Moved DoF processing to happen before stereo eye warping.
* Prevented the DoF mask from being misaligned after left/right stereo pixel shifting.
* Disabled the older post-warp DoF pass to avoid double blur and subject blur artifacts.

* Added safer DoF mask behavior using:

  * subject focus protection
  * foreground protection
  * background-only blur masking
  * feathered blur transitions
  * adjustable background start and fade controls

* Improved DoF behavior for both video conversion and still-image conversion.

* Kept the cinematic background-push effect intact while making DoF function more like an optional background softness layer.

### Still Image 3D Rendering Updates

* Updated still-image DoF placement so blur is applied before stereo warping.
* Improved still-image subject protection during DoF processing.
* Prevented post-warp still-image DoF from blurring shifted subject pixels incorrectly.
* Improved consistency between still-image and video 3D render behavior.
* Kept still-image output compatible with the same Depth Pop, Background Push, and stereo shift pipeline used by video renders.

### Video 3D Rendering Updates

* Updated video DoF placement so DoF is applied after active keyframe settings are resolved but before stereo pixel shifting.
* Fixed DoF timing so keyframed DoF strength can be used correctly.
* Prevented DoF from referencing per-frame render settings before they are created.
* Improved video render order for:

  * depth tensor preparation
  * active 3D settings
  * keyframe-adjusted DoF strength
  * pre-warp DoF
  * stereo pixel shifting
  * final eye formatting

* Left the older post-warp DoF path disabled to avoid subject blur and stereo mask mismatch.

### 3D Render Stability

* Improved Free/Pro-aware render flow.
* Pro renders do not apply Free watermark.
* Free renders keep watermark behavior active.
* Improved final render handling so watermark/audio workflows are more predictable.
* Improved behavior around final output replacement after FFmpeg processing.
* Reduced unnecessary extra encode passes for Free watermark/audio workflows.
* Improved render pipeline ordering for cleaner output handling.
* Reduced risk of unexpected final-output replacement issues after audio merge or watermark processing.

---

## Depth Blender Updates

VisionDepth3D v5.0 includes Depth Blender fixes and cleanup for the newer PySide6 interface.

### Depth Blender Slider Fixes

* Fixed Depth Blender sliders not affecting preview or batch output.
* Fixed PySide6 slider state not syncing back into the actual blend parameter values.
* Restored expected behavior from the older Tkinter version, where slider variables updated automatically.

* Added proper syncing for:

  * White Strength
  * Feather Blur
  * CLAHE Clip Limit
  * CLAHE Tile Grid
  * Bilateral Filter Diameter
  * Bilateral sigmaColor
  * Bilateral sigmaSpace

* Improved live preview behavior so slider movement updates the actual processing parameters.
* Improved batch output behavior so Start Batch uses the latest slider values.
* Added a safety sync step before preview generation.
* Added a safety sync step before batch processing starts.
* Improved consistency between displayed slider values, preview output, and final blended output.

### Depth Blender Preview and Processing Reliability

* Improved Depth Blender live-preview reliability.
* Reduced cases where the UI appeared to update while the backend continued using default values.
* Improved parameter handling between the PySide6 Depth Blender page and the core blending function.
* Preserved GPU/CPU Depth Blender processing while improving UI-to-backend parameter flow.
* Improved confidence that preview output matches final batch output.

---

## Updated Known Notes

* Depth Pop / Background Push is separate from Depth of Field.
* Depth Pop controls the perceived 3D separation and “window into the scene” look.
* DoF is now treated as an optional background-softness layer.
* Strong DoF values can still soften edges if the depth map is inaccurate, but the new background-only DoF behavior should better protect subjects.
* DoF should be applied before stereo warping to avoid blur-mask misalignment.
* Depth-order warp and background push can create strong subject/background separation even without DoF.
* Depth Blender sliders now need explicit state syncing in the PySide6 interface because PySide sliders do not behave like Tkinter `DoubleVar` / `IntVar` variables automatically.

---

## Depth Estimation Pipeline Updates

VisionDepth3D v5.0 includes major depth-estimation pipeline work, including new adapter handling, ONNX testing, ZoeDepth support, model registry cleanup, normalization checks, and backend compatibility fixes.

### Depth Model Registry Cleanup

* Cleaned model registry for Free/Pro release.
* Removed or hid non-commercial/research-risk models from official public builds.
* Added public/internal model registry separation.
* Added `official_build` style filtering logic for safer release builds.
* Cleaned tier labels from “Free Tier” / “Pro Tier” to simpler “Free” / “Pro”.
* Removed risky model entries from public-facing registry.
* Kept internal/research model tracking separate from official release models.
* Added better model licensing awareness for:

  * MIT
  * Apache-2.0
  * CC-BY-NC
  * research-only
  * unclear-license models
* Removed ViGeo from official builds after discovering the Hugging Face weights were CC-BY-NC-4.0, despite permissive source code licensing.

---

## ZoeDepth Adapter Integration

VisionDepth3D v5.0 adds a dedicated ZoeDepth adapter path.

### ZoeDepth Model Support

Added ZoeDepth model entries:

* ZoeDepth NK Recommended
* ZoeDepth N Indoor
* ZoeDepth K Outdoor

ZoeDepth NK is now the recommended ZoeDepth option because it produced significantly cleaner results than ZoeDepth N during testing.

---

## ONNX Depth Pipeline Updates

VisionDepth3D v5.0 includes new ONNX depth testing and pipeline verification.

### ONNX Model Support

* Added Depth Anything v2 Small ONNX test entry.
* Tested ONNXRuntime execution path with DA-v2 Small ONNX.
* ONNX models use the ONNX adapter path.
* Improved ONNX model resolution handling.
* Ensured ONNX outputs return standard `predicted_depth` format for the main render loop.
* Standard ONNX depth models flow through the same depth normalization path as non-VDA models.

### ONNX Normalization Verification

* Verified that normal ONNX models should use `FixedPercentileNormalizer`.
* VDA-style ONNX models are treated separately only when `_is_vda_onnx` is set.
* Added recommended debug logs for normalizer mode:

  * normalizer enabled
  * normalizer disabled
  * VDA skip path
* ONNX flicker may come from model output instability even when range normalization is enabled.
* Added clearer distinction between:

  * depth range breathing
  * structural model flicker
  * per-frame ONNX prediction instability

---

## Depth Normalization and Flicker Control

VisionDepth3D v5.0 includes a cleaner depth-stability workflow that separates normal depth normalization from optional experimental flicker-control tools.

This update keeps the normal depth normalization path as the recommended default while giving advanced users optional controls for testing stabilization behavior.

### Normalizer Behavior

* `FixedPercentileNormalizer` remains the recommended default for regular video depth generation.
* Fast per-frame normalization remains available when scene normalization is disabled.
* Normal ONNX models continue through the main depth normalization path unless specifically marked as VDA-style ONNX.
* VDA-style models keep their separate sequence-aware handling.
* Debug logging now makes the selected depth normalization route easier to inspect during testing.
* The depth pipeline now separates:

  * normal depth range normalization
  * optional temporal range stabilization
  * optional motion-aware depth smoothing
  * model-level structural flicker

### Experimental Depth Stability Controls

VisionDepth3D now includes optional advanced depth-stability controls for testing different flicker-control behavior.

Added experimental toggles for:

* Depth Range Stabilization
* Motion-Aware Depth Smoothing

These controls are intended for testing and advanced tuning. They are kept separate from the normal depth normalization path so users can keep the recommended reliable depth output while still having access to experimental stabilization behavior when needed.

### Motion-Aware Depth Smoothing Option

Motion-Aware Depth Smoothing remains available as an optional experimental control.

The feature is designed to test depth smoothing based on frame-to-frame image motion, with different behavior for static areas and moving areas.

Because temporal depth blending can affect moving subjects, this option is best treated as an advanced test control rather than the default depth output mode.

### Current Depth Stability Direction

The current v5.0 depth strategy is:

* Keep regular depth normalization enabled by default.
* Use fixed percentile normalization as the recommended stable depth range path.
* Keep fast per-frame normalization available through the Disable Depth Normalizer option.
* Keep experimental stabilization and smoothing controls separate from the default path.
* Avoid forcing temporal smoothing into standard depth output.
* Treat ONNX single-frame flicker separately from normalization behavior.
* Keep VDA ONNX and standard ONNX models separated in the pipeline.

This gives VisionDepth3D a safer balance between stable depth range handling and clean depth maps for stereo rendering.

## Depth Resolution Updates

* Added `512x384 (ZoeDepth Native)` inference resolution.
* Preserved existing Depth Anything, MiDaS, Marigold, DA3, and widescreen inference sizes.
* Improved model-specific resolution testing.
* ZoeDepth should be tested at 512x384 first.
* DA-v2/Depth Anything models should continue using their preferred native or stable inference resolutions.

---

## Local Model and Backend Handling

### Local Backend Improvements

* Improved adapter isolation so model-specific patches do not affect the rest of VD3D.
* Improved release safety by keeping vendored model code cleaner.

### Hugging Face Depth Model Loading Fixes

* Fixed Hugging Face depth model loading for models that download directly from online repositories.
* Improved Distill-Any-Depth Large and Small Hugging Face model handling.
* Fixed an issue where Distill-Any-Depth models could fail to download or load correctly because the Hugging Face snapshot download path was being interrupted before the model files were fully cached.
* Added safer Hugging Face cache handling using a dedicated local model folder and cache directory.
* Improved local model folder handling for Hugging Face depth models so downloaded models can be reused more reliably after the first successful load.
* Added SSL environment repair for broken `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, or `CURL_CA_BUNDLE` paths that could prevent Hugging Face downloads from starting.
* Improved compatibility with `certifi` so packaged builds have a safer certificate fallback for Hugging Face / HTTPX downloads.
* Improved fallback behavior when a Hugging Face local model folder is incomplete or fails to load.
* Added cleaner re-download handling when a cached Hugging Face model is missing files or was only partially downloaded.
* Improved processor loading by trying both `AutoProcessor` and `AutoImageProcessor` paths where supported.
* Confirmed Distill-Any-Depth Large can now load and render through the generic Hugging Face depth path in the CUDA build.
* Added testing coverage for Distill-Any-Depth Large at higher inference resolutions such as `700x700`.

---

## UI and Settings Improvements

* Added 3D Assistant state persistence.
* Added close-event save behavior.
* Improved settings save/load consistency.
* Improved settings file path handling during testing.
* Improved Free/Pro label refresh after activation.
* Improved model dropdown refresh after tier changes.
* Improved Pro model visibility updates after activation.
* Improved app startup state handling when Pro license exists.

### Translation and Localized Status Support

VisionDepth3D v5.0 expands translation support across user-facing workflow status text.

New localization wiring includes:

* Depth Engine status text translation support.
* Translatable depth-processing completion, cancel, and error statuses.
* Translatable folder-processing status messages.
* Translator fallback behavior for core processing code.
* Service-level translator wiring so backend processing can safely display localized UI messages without directly depending on the main window.
* English fallback behavior when no translator is available.

This keeps developer logs in English while allowing user-facing status labels and workflow messages to follow the selected app language.

---

## Recommended Models After Testing

### Recommended Default Depth Models

* Depth Anything v2 Small
* Distill-Any-Depth models
* Video Depth Anything Small
* ZoeDepth NK Recommended
* DA3 models where licensing/build policy allows
* ONNX models after stability testing

### ZoeDepth Recommendation

ZoeDepth NK is now the recommended ZoeDepth option.

ZoeDepth N is better treated as an indoor-focused option and may produce artifacts on some cinematic, game, or unusual footage.

ZoeDepth K is better treated as an outdoor/driving-focused option.

---

## Known Notes

* Some ONNX depth models may still flicker structurally even when range normalization is enabled.
* Depth normalizers reduce depth breathing, but they cannot fully fix unstable per-frame model predictions.
* Experimental temporal smoothing can reduce flicker but may create artifacts or motion trailing depending on the clip.
* Motion-aware smoothing is available as an advanced testing option and is not the default depth output path.
* ZoeDepth N may produce white depth islands/blotches on some scenes.
* ZoeDepth NK produced better results in testing and is recommended over ZoeDepth N.
* Some model licenses differ between source code and model weights, so model availability may vary by official build policy.
* Windows may cache old app icons after rebuilds; unpin/reinstall/repin may be required.
* Linux support may vary depending on CUDA, drivers, FFmpeg, Qt, and system configuration.

---

## Summary

VisionDepth3D v5.0 is a major release-preparation update focused on turning VisionDepth3D into a cleaner Free/Pro desktop product.

This update improves the 3D pipeline, depth-estimation backend, model registry, Pro activation flow, offline Pro license cache behavior, watermark behavior, settings persistence, translation support, ONNX model support, ZoeDepth integration, and release packaging.

The result is a more stable and professional foundation for VisionDepth3D going forward, with a stronger core workflow for Free users and a fuller production pipeline for Pro users.
