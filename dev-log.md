# VisionDepth3D v4.2 Dev Log
Backend Stability, Performance, and Compatibility Update

This is the technical dev log for VisionDepth3D v4.1.2. This release focuses on backend stability, GPU/backend compatibility, FFmpeg reliability, render performance, depth-engine improvements, HDR10 handling, FPS/Upscale fixes, Depth Blender stability, and cleaner progress/debug reporting.

---

## Release Focus

- Backend stability across CUDA, DirectML, Metal, ROCm, and CPU paths.
- More reliable FFmpeg encoding, decoding, frame extraction, audio merge, and subprocess cleanup.
- Better failure reporting so failed or cancelled jobs are not shown as successful.
- Improved DirectML readiness for AMD / Intel builds.
- Safer depth model loading, inference, cancellation, and progress behavior.
- More consistent render-state reset between jobs.
- Better HDR10 preservation and FFmpeg color pipeline handling.
- Improved FPS / Upscale timing, frame validation, cancellation handling, and tiled ESRGAN behavior.
- Cleaner queue progress reporting and reduced debug/log spam.

---

## Preliminary Performance Notes

Performance depends heavily on GPU, model, resolution, codec, batch/window size, source FPS, output format, and enabled quality options. On the current test system:

- 3D Generator heavy Full-SBS cinematic render reports around **13-15 FPS** with real backend FPS reporting.
- Depth Engine standard video processing improved from roughly **3 FPS** to around **7 FPS** after UI/proxy and pipeline overhead reductions.
- Video Depth Anything Small at **518x518** tested around **7.5 FPS** with the updated VDA adapter and streaming path.
- The previous `0.01 FPS` style 3D render readout was a UI reporting bug caused by displaying progress-percent-per-second as FPS instead of true backend frame FPS.

These are development-test numbers and may vary depending on hardware, VRAM, model type, resolution, codec, edge repair, DOF, debug mode, batch size, and other settings.

---

## Short Release Notes

VisionDepth3D v4.2 is a backend stability, performance, and compatibility update focused on long-render reliability, GPU/backend clarity, DirectML readiness, FFmpeg/audio reliability, HDR10 preservation, depth-engine speed, adapter optimization, FPS/Upscale stability, and safer queue/debug reporting.

### Main User-Facing Changes

- Fixed incorrect 3D/depth FPS reporting in the Job Queue.
- Real backend FPS is now shown instead of progress-percent-per-second.
- Improved Depth Engine speed by removing per-frame timer-thread progress callback overhead.
- Optimized Depth Anything V2, Depth Anything 3, and Video Depth Anything adapters.
- Improved Video Depth Anything streaming performance and memory behavior.
- Fixed VDA streaming progress crashes caused by missing throttled UI state.
- Improved 3D render performance by reducing unnecessary CPU/GPU transfers and keeping more formatting work on GPU.
- Fixed edge-repair tensor shape mismatch crashes caused by even pooling kernels.
- Added compatibility handling for existing 3D KeyframeService implementations.
- Improved GPU/backend detection for CUDA, DirectML, Metal, ROCm, and CPU.
- Added support work for a separate AMD / Intel DirectML build.
- Improved ONNX Runtime provider selection, including DirectML-capable AMD / Intel systems.
- Improved HDR10 preservation and FFmpeg color handling.
- Improved FFmpeg stability, error reporting, cleanup, and long-render behavior.
- Improved audio merge handling after failed renders and for non-MP4 containers.
- Fixed cancelled or failed renders incorrectly appearing successful or 100% complete.
- Improved FPS / Upscale stability, timing accuracy, frame validation, cancellation handling, and tiled ESRGAN output scaling.
- Improved Depth Blender stability, preview safety, GPU efficiency, frame matching, and video output compatibility.
- Added safer no-model-loaded handling for image, folder, and video depth generation.
- Reduced debug/log spam when Debug mode is off.

---

## Highlights

- Improved CUDA, DirectML, Metal, ROCm, and CPU backend detection.
- Added support work for a separate AMD / Intel DirectML build.
- Improved ONNX Runtime provider selection for CUDA, ROCm, DirectML, CoreML, OpenVINO, and CPU.
- Fixed critical render failure reporting so failed or cancelled jobs are no longer shown as successful.
- Fixed incorrect FPS reporting in the shared Job Queue.
- Improved 3D render performance by reducing unnecessary shift-map, interpolation, CPU/GPU transfer, and formatting overhead.
- Fixed 3D keyframe compatibility with the existing KeyframeService implementation.
- Fixed edge-repair tensor shape mismatch crashes caused by even pooling kernels.
- Improved HDR10 preservation with corrected BT.2020 / PQ / RGB48 / P010 handling.
- Improved FFmpeg subprocess cleanup, stderr draining, error reporting, odd-dimension padding, and long-render stability.
- Improved audio merge handling after failed renders and for MKV/AVI outputs.
- Improved split-eye rendering, output naming, temporary cleanup, and audio merge behavior.
- Improved FPS / Upscale stability, frame validation, cancellation handling, and tiled ESRGAN output scaling.
- Improved Depth Engine performance by removing UI/proxy callback overhead and reducing per-frame progress update cost.
- Reworked Video Depth Anything processing to use streaming windows instead of holding full videos in memory.
- Optimized Depth Anything V2, Depth Anything 3, and Video Depth Anything adapters.
- Improved Depth Blender stability, GPU efficiency, preview thread safety, frame matching, and video output compatibility.
- Added safer no-model-loaded handling for image, folder, and video depth generation.
- Reduced debug/log spam during normal non-debug use.

---

## GPU and DirectML

- Fixed AMD / Intel systems incorrectly showing `CPU` in the top GPU label when CUDA/NVIDIA detection was unavailable.
- The top bar now separates the physical system GPU from the active compute backend.
- Added clearer backend detection for CUDA, DirectML, Metal, ROCm, and CPU.
- Added DirectML detection support for AMD / Intel GPU builds.
- Added DirectML-safe handling for Torch operations that are not fully supported on DirectML.
- Improved GPU Diagnostics messaging so users can better understand CUDA vs DirectML behavior.
- Improved ONNX Runtime provider selection:
  - CUDA / ROCm are preferred when available.
  - DirectML is selected on AMD / Intel DirectML builds when `DmlExecutionProvider` is available.
  - CoreML and OpenVINO are considered when available.
  - CPU remains the safe fallback.
- Fixed ONNX depth models incorrectly falling back to CPU on some DirectML-capable systems.

---

## DirectML Build Notes

- Added support work for a separate AMD / Intel DirectML `.exe` build.
- The DirectML build uses `torch-directml` instead of CUDA PyTorch.
- NVIDIA/CUDA and AMD/Intel DirectML builds are now treated as separate release targets.
- NVIDIA users should continue using the CUDA build for best performance.
- The DirectML build is experimental and depends on community testing for AMD / Intel hardware.
- ONNX models now have improved provider selection for DirectML systems.

---

## Live 3D

- Improved Live 3D backend detection for CUDA, DirectML, Metal, and CPU.
- Added DirectML device support work for Live 3D depth processing on AMD / Intel builds.
- Updated Live 3D device handling so it no longer depends only on CUDA detection.
- Added safer FP16 behavior so FP16 is only enabled when using CUDA.
- Added zero parallax and convergence controls to Live 3D for better real-time depth tuning.
- Zero parallax and convergence values can now use negative settings for workflows that need stronger depth placement control.
- Improved Live 3D browser/MJPEG preview workflow for lower-latency testing.
- Improved Live 3D startup/logging clarity so users can better see which backend is being used.
- Live 3D stereo acceleration remains best on CUDA while DirectML support continues expanding for AMD / Intel users.

---

## 3D Render Performance and Stability

- Fixed a major slowdown caused by returning the full shift map during normal video renders.
- Shift maps are now reserved for preview/heatmap use instead of full renders.
- Full renders still keep lightweight metadata needed for subject depth, zero parallax, convergence, and floating-window logic.
- Reduced heavy per-frame work in the 3D render path.
- Edge Repair `Off` now skips unnecessary repair-mask processing.
- Heavy RGB-guided depth refinement is no longer used during normal renders unless higher edge repair modes require it.
- Reduced redundant interpolation inside the render pipeline.
- DOF focal tracking now only runs when DOF is enabled.
- Main subject tracking, zero parallax, convergence, and floating-window logic remain active outside DOF.
- Improved SDR render formatting so more sharpen, resize, padding, floating-window masking, VR projection, and final 3D packing work stays on GPU.
- Reduced unnecessary CPU/GPU transfers during final SDR frame formatting.
- Improved tensor-to-frame conversion by doing RGB-to-BGR conversion and uint8 quantization on GPU before the final CPU copy.
- Added fast-path returns for tensor padding helpers when frames already match the target output size.
- Removed avoidable GPU synchronization from SDR tensor validation inside the frame-packing path.
- Added FFmpeg write timing to render profiling so encode/pipe overhead can be measured separately.
- Fixed still-image 3D rendering unnecessarily copying full shift-map data from GPU to CPU.
- Still-image renders now request only lightweight render metadata unless shift-map preview data is actually needed.
- Improved tensor conversion safety before writing SDR frames.
- Edge repair presets now return isolated preset copies to avoid accidental global preset mutation.
- Fixed tensor shape mismatch crashes such as `1921` vs `1920` caused by even-sized max-pool kernels.
- Edge repair mask dilation now forces odd kernel sizes to preserve tensor dimensions.
- Added final shape guards for repair/protect masks to prevent H+1/W+1 mask drift from crashing frame blending.
- Fixed a critical issue where `render_sbs_3d()` could return an output path from cleanup even after a render crash or FFmpeg failure.
- Render and FFmpeg failures now propagate correctly so the UI and queue can report real failures.
- Added safer default suspend/cancel event handling for non-GUI render calls.
- Reset subject-depth EMA state at the start of each render to prevent previous renders from influencing new subject tracking, convergence, and floating-window behavior.
- Fixed cancelled renders incorrectly forcing progress to 100%.
- Added lightweight progress/status logging when Debug mode is off so users can monitor long renders without enabling expensive profiling logs.
- Fixed a startup crash in `app_controller.py` caused by a missing `copy` import when freezing render state for background render threads.
- Improved render-state snapshot behavior so UI changes do not mutate active render settings mid-render.

---

## Split-Eye 3D Rendering

- Improved left-eye, right-eye, and both-eye video export handling.
- Split-eye video exports now render one temporary SBS file and crop final left/right eye outputs from it, reducing duplicate full-render work.
- Fixed VR180 split-eye output naming so left and right eye files receive unique filenames.
- Added early validation for unsupported split-eye output formats such as red-cyan anaglyph and passive interlaced.
- Added collision checks to prevent left/right split-eye outputs from resolving to the same path.
- Improved temporary SBS cleanup when split-eye rendering fails, is cancelled, or exits early.
- Improved FFmpeg crop cancellation handling for split-eye extraction.
- Added cleanup for partial or invalid split-eye crop outputs.
- Improved split-eye crop failure reporting by surfacing recent FFmpeg stderr output.
- Improved split-eye audio merge behavior so final outputs keep the expected user-selected filenames.
- Reduced temporary file clutter from split-eye audio merge operations.

---

## Experimental 3D Keyframes

- Added early backend/UI support for 3D parameter keyframes.
- Users can define different stereo/depth settings across different sections of a clip.
- Keyframes can help switch between close-up, wide-shot, and safer panning settings during one render.
- Added automatic keyframe JSON creation, loading, saving, updating, and deletion from the 3D Generator UI.
- Added section-based keyframe behavior where each keyframe controls settings from its frame until the next keyframe.
- Fixed keyframed render crashes caused by missing `KeyframeService.get_settings_for_frame()` in the active service implementation.
- Added a keyframe compatibility helper so existing keyframe data can be resolved per frame without requiring a full KeyframeService rewrite.
- Keyframed renders now use the absolute source frame index so keyframes remain aligned when rendering clipped segments.
- Improved keyframe-service import behavior so real keyframe-service errors are no longer hidden by overly broad exception handling.

---

## HDR10 Rendering

- Fixed HDR10 preservation issues where output could be tagged as HDR but not fully converted through the correct HDR color pipeline.
- Added explicit HDR10 input handling from BT.2020 / PQ / limited-range YUV into RGB48 for VD3D processing.
- Added explicit HDR10 output conversion from RGB48 back to BT.2020nc / PQ / limited-range P010 for encoding.
- Improved HDR10 metadata handling for HEVC output.
- Added source HDR10 metadata probing for mastering display and MaxCLL / MaxFALL when available.
- Added fallback HDR10 metadata when source metadata cannot be read.
- Disabled SDR-style color grading when Preserve HDR10 is enabled to avoid HDR color shifts.
- Added an early guard preventing HDR10 preservation from running through OpenCV VideoWriter, which is SDR-only in this pipeline.
- Improved HDR10 render cleanup by closing the HDR FFmpeg reader generator during render shutdown.
- Fixed YUV10 reader cleanup so FFmpeg subprocesses are not left running if iteration stops early.
- HDR10 render failures now propagate correctly instead of being hidden by final cleanup code.

---

## Queue Dock, Progress, and Debug Logging

- Fixed Depth Blender progress not updating correctly in the shared Job Queue.
- Restored the unified queue progress format used by the 3D Generator:

```txt
119/9557 | FPS: 3.10 | Elapsed: 00:00:39 | ETA: 00:50:46
```

- Fixed duplicate progress/status text appearing in the queue.
- Increased the queue dock/log area size for better readability.
- Added **Copy Log** and **Clear Log** buttons.
- Added log line capping so large debug logs do not overload Copy Log.
- Cancelled renders now leave the queue/status text in a cancelled state instead of showing a misleading completed render.
- Fixed 3D render FPS display so the queue no longer shows progress-percent-per-second as FPS.
- Fixed Depth Engine FPS parsing so real backend status FPS is preferred when available.
- Queue status now preserves real render/depth status text when it includes FPS, elapsed time, and ETA.
- Added safer handling for render/depth progress payloads with frame counts, totals, elapsed time, ETA, and real FPS.
- Improved normal non-debug progress visibility without requiring expensive Debug profiling output.
- Improved Debug toggle behavior.
- Moved non-essential render and depth profiling output behind the Debug toggle.
- Depth video stage profiling now only runs when Debug is enabled.
- Depth profiling now avoids expensive CUDA synchronization unless Debug profiling is enabled.
- Important warnings and errors still print normally.
- Improved profiling labels for easier render performance diagnosis.
- Reduced VDA adapter log spam by moving per-window runtime messages behind Debug logging.
- Added better separation between normal progress updates and debug-only performance profiling.
- Improved diagnostic output for depth model settings and active adapter type.

---

## Depth Engine Performance

- Improved DirectML compatibility for depth generation.
- Added clearer user-facing messages when attempting to process depth before loading a model.
- Image, image-folder, and video depth processing now stop cleanly with:

```txt
No depth model loaded. Please select a model first.
```

- Prevented unclear `NoneType is not callable` failures when users start processing without a loaded model.
- Improved model-call stability by snapshotting the active depth pipeline at inference time.
- Reduced the chance of a model switch affecting an in-progress inference call.
- Improved adapter fallback behavior when a depth pipeline does not accept `inference_size` or extra runtime kwargs.
- Improved Hugging Face FP16 handling so standard online HF model loads now respect the FP16 option on CUDA.
- Added safer image-folder prediction handling when a model returns a different number of predictions than input images.
- Fixed potential image-folder crashes from indexing past the available input files.
- Made 16-bit depth median cleanup optional instead of always applying it.
- Default 16-bit depth outputs now preserve more fine detail and avoid unnecessary CPU work.
- Fixed `DepthProgressLabelProxy.after()` and `DepthProgressProxy.after()` so they no longer spawn a new `threading.Timer` for every progress/status update.
- Removed thousands of tiny timer-thread creations during long depth jobs, significantly reducing CPU overhead.
- Throttled depth video UI/progress updates to avoid updating Qt/Tk-compatible proxies every frame.
- Depth video progress now updates at a controlled interval while preserving accurate FPS, ETA, and progress reporting.
- Added safer depth stage profiling with optional CUDA synchronization only when Debug mode is enabled.
- Added clearer depth settings logging so model type, inference size, batch size, codec, normalization mode, save-frames mode, and backend are easier to diagnose.
- Reduced unnecessary per-frame UI bridge overhead in both normal depth video processing and VDA streaming mode.
- Improved `DepthService` stability when wrapping legacy video depth processing.
- Depth processing now clears both service-local and legacy `core.render_depth` suspend/cancel flags before each new run, preventing stale cancellation state from immediately stopping later jobs.
- Added clearer validation for missing input videos, invalid output paths, output-directory creation failures, invalid batch sizes, and OpenCV frame-count probing.
- Depth processing now returns the backend-provided output path when available instead of always assuming a fixed `.mkv` output name.

---

## Video Depth Anything Processing

- Reworked Video Depth Anything processing to use a true streaming-window workflow instead of loading the entire video into memory.
- VDA now processes active frame windows, writes finalized depth frames as it goes, and keeps only overlap frames needed for temporal blending.
- Greatly reduced RAM pressure during VDA video renders, especially on longer clips.
- Improved stability for larger VDA window sizes, including successful 32-frame VDA Large processing on supported systems.
- Reduced unnecessary garbage collection and cache clearing during VDA processing to avoid slowing down active inference loops.
- Improved VDA memory behavior for laptop GPUs and lower-VRAM systems.
- Fixed a VDA streaming progress crash caused by `last_ui_update` not being available in the nested VDA window function.
- VDA and non-VDA depth paths now share safe throttled progress update state.
- Added optional lower-overlap VDA streaming configuration to reduce duplicate overlap inference when speed is preferred.

---

## Depth Model Adapter Optimizations

### Depth Anything V2 Adapter

- Reworked the Depth Anything V2 safetensors adapter to use true batched inference.
- Previous behavior processed each input frame one at a time even when the depth pipeline supplied a batch.
- New behavior stacks frames into a batch, performs one GPU transfer, runs one batched model call, and returns batched predictions.
- Added optional CUDA FP16 support from the Depth Engine FP16 setting.
- FP32 safetensors can now run in FP16 on CUDA when the user enables FP16, improving performance on RTX GPUs.
- Enabled CUDA TF32 / cuDNN benchmarking where available for better GPU throughput.
- Moved normalization constants to GPU to avoid repeated CPU-side normalization work.
- Added channels-last memory-format support where safe.
- Reduced CPU preprocessing overhead by using direct NumPy/PyTorch conversion instead of repeated torchvision transforms.
- Batched output resizing and normalization are now performed on GPU when possible.
- Mixed-size image-folder inputs still fall back safely to per-image processing.

### Depth Anything 3 Adapter

- Reworked DA3 inference-size handling so selected UI resolution is respected more directly.
- Fixed hidden slowdown where `518x518` could be rounded up to `640` process resolution.
- DA3 now maps selected inference size to `process_res` using the actual maximum selected dimension.
- Removed redundant adapter-side percentile normalization by default.
- Main VD3D depth normalization now handles normalization once, avoiding extra per-frame `torch.quantile()` work.
- Switched DA3 inference wrapper from `torch.no_grad()` to `torch.inference_mode()`.
- Added safer conversion of DA3 prediction outputs from NumPy, tensor, or list formats into consistent `[N,H,W]` tensors.
- Added optional fast adapter-side min/max normalization only when explicitly requested.
- Improved DA3 warmup to use the selected inference resolution instead of forcing a heavier fixed `process_res=756`.

### Video Depth Anything Adapter

- Reduced VDA adapter overhead by avoiding unnecessary NumPy → Torch → NumPy round-trips.
- VDA adapter now returns NumPy float32 depth arrays directly, which the main depth pipeline already supports.
- Replaced always-on VDA per-window prints with debug-only logging to reduce console/UI log overhead.
- Added one-time warning for very short VDA sequences instead of repeating warnings every window.
- Switched VDA inference wrapper to `torch.inference_mode()`.
- Enabled CUDA TF32 / cuDNN benchmarking where available.
- Improved VDA input-size handling so the selected UI inference resolution can control VDA `input_size`.
- Removed hardcoded `input_size=518` from the VDA streaming call path so presets such as `518x518` or widescreen settings can behave as expected.
- Reduced CPU overhead in the VDA streaming path by avoiding NumPy → PIL → NumPy conversion per frame.
- VDA streaming now passes RGB NumPy frames directly into the adapter.

---

## Depth Blender Stability and Performance

- Fixed a Depth Blender import/startup crash that could occur when PyTorch was not installed or failed to import.
- Added safer PyTorch helper defaults so CPU-only systems can still load the Depth Blender module correctly.
- Added Windows-safe cleanup for unsupported PyTorch CUDA allocator settings such as `expandable_segments`, preventing noisy allocator warnings inherited from other launch environments.
- Improved Depth Blender GPU processing by reducing unnecessary CPU/GPU round trips during the blend path.
- Cached the V2 white-threshold calculation more consistently to avoid redundant percentile work per frame.
- Improved CPU normalization performance by using OpenCV mean/std helpers.
- Improved Depth Blender preview stability by moving Tkinter image creation back to the main UI thread.
- Preview rendering now uses a queue-based handoff from the worker thread to the Tkinter UI thread.
- Preview errors are now logged instead of being silently ignored.
- Improved preview responsiveness by discarding stale queued preview frames and displaying only the latest result.
- Fixed Depth Blender frame-folder pairing so batch processing uses matching filenames from both V1 and V2 folders.
- Updated Depth Blender preview pairing to better match batch frame-pairing behavior.
- Added warnings when V1 and V2 frame folders contain different numbers of matching PNG files.
- Added safer output-size validation before starting Depth Blender batch processing.
- Added validation requiring both width and height to be entered together.
- Added video-size validation to prevent odd-dimension MP4 output issues.
- Improved Depth Blender video output compatibility by writing grayscale depth output as standard BGR video frames.
- Added Depth Blender video-writer open validation so failed output initialization is reported immediately.
- Improved Depth Blender video resource cleanup so `VideoCapture` and `VideoWriter` handles are released on success, failure, or early return.
- Reduced the chance of empty or invalid Depth Blender MP4 files from unsupported single-channel video writer configurations.
- Implemented Depth Blender output-control UI state handling so output folder controls are disabled when overwriting V2 frames.
- Added frame-write failure detection using `cv2.imwrite()` return values.
- Reduced unnecessary forced garbage collection during long Depth Blender renders to avoid periodic stalls.
- Clarified that decoder warnings such as `mmco: unref short failure` usually come from the source H.264 stream and are normally non-fatal.

---

## ONNX Depth Models

- Added improved ONNX Runtime provider selection.
- DirectML-capable AMD / Intel systems can now use `DmlExecutionProvider` when available.
- CUDA / ROCm remain preferred on compatible GPU builds.
- CPU remains the fallback provider when no GPU provider is available.
- Improved ONNX VDA fixed-size handling and warm-up behavior.
- Improved ONNX model stability by keeping the warm-up-proven inference size when available.
- Improved ONNX fallback behavior for fixed temporal-window video models.
- Added safer ONNX Runtime thread-count initialization when CPU count cannot be detected.
- Fixed ONNX models falling back to CPU on some DirectML-capable systems.

---

## FFmpeg, Video Writing, and Audio Merge

- Improved FFmpeg writer stability for long video exports.
- Fixed a possible FFmpeg pipe deadlock by draining FFmpeg `stderr` in a background thread.
- FFmpeg error output is now collected safely and reported after encode completion.
- Added padding for odd-width or odd-height videos when using `yuv420p` encoders.
- Fixed failures where H.264 / H.265 / AV1 encoders could reject odd video dimensions.
- Improved FFmpeg cleanup when encoding fails, is cancelled, or times out.
- Added safer handling around FFmpeg process shutdown and stderr collection.
- Reduced chances of stalled depth renders during video encoding.
- Fixed a critical render-path issue where exceptions raised during rendering or FFmpeg cleanup could be accidentally swallowed by a `finally` return.
- Render failures and FFmpeg encode failures now correctly propagate instead of being reported as successful output paths.
- Improved FFmpeg stderr handling so drained error output is reused for failure reporting instead of attempting to read from an already-drained pipe.
- Optimized FFmpeg stderr draining to avoid repeated full-buffer joins during long or noisy encodes.
- Fixed FFmpeg error reporting after pipe-write failures so recent stderr output is shown reliably.
- Fixed helper FFmpeg writer option placement so encoder options are inserted before the output pixel format instead of the input raw pixel format.
- Improved FFmpeg cleanup for HDR and YUV reader subprocesses when rendering stops early, fails, or is cancelled.
- Improved FFmpeg frame extraction memory behavior by avoiding full progress-output capture during extraction.
- FFmpeg frame extraction now logs only recent error output on failure instead of storing normal progress output in memory.
- Improved FFmpeg usage in the FPS / Upscale pipeline by validating raw BGR frames before writing to stdin.
- Improved FFmpeg rawvideo writing safety by ensuring frames are valid `uint8` BGR frames with the expected output resolution.
- Added safeguards so audio merge does not run on missing, empty, or invalid render outputs.
- Prevented failed or corrupt renders from producing confusing FFmpeg audio merge errors.
- Changed misleading “Render complete” messaging after failed renders to a neutral session-ended message.
- Fixed audio merge command generation so `-movflags +faststart` is only used for MP4/MOV-style containers.
- Improved audio merge compatibility with MKV/AVI outputs by avoiding MP4-specific muxer flags on non-MP4 containers.
- Split-eye renders now merge original audio back into final left/right outputs only after the cropped eye video is successfully created.
- Added safer temporary audio-merge output handling and cleanup for split-eye renders.

---

## FPS / Upscale Pipeline

- Fixed threaded FPS / Upscale processing dropping the first source frame.
- Threaded FPS / Upscale output now matches the non-threaded pipeline timeline more accurately.
- Improved RIFE/FPS interpolation output ordering in the threaded pipeline by ensuring the first source frame is written before interpolated/current frames.
- Added safer frame validation before sending processed frames to FFmpeg.
- Improved threaded pipeline cancellation and failure reporting so stopped or failed jobs are no longer reported as successfully completed.
- Fixed cancelled threaded FPS / Upscale jobs incorrectly reaching a completed state.
- Improved threaded pipeline audio merge behavior so audio merge is skipped when processing fails or is cancelled.
- Improved ONNX ESRGAN tiled upscaling.
- Fixed tiled ESRGAN output allocation so tiled models that return 2x or 4x output no longer collapse back to the original input size.
- Improved tile placement logic for padded ESRGAN tiles to better preserve scaled output dimensions.
- Improved image blending safety for upscaled frames.
- Blend modes now resize the original frame to match the upscaled frame before blending, preventing size mismatch failures.
- Improved no-upscaler fallback behavior so output frames are still resized to the requested render size when needed.
- Reduced the chance of FFmpeg hangs or corrupted output from invalid frame shapes, dtypes, or channel layouts.
- Improved FPS / Upscale threaded pipeline shutdown behavior after cancellation or worker failure.

---

## Folder Processing and UX

- Added output-folder validation before batch video-folder processing starts.
- Video-folder processing now stops early with a clear message if the output folder is missing or cannot be created.
- Prevented misleading “All videos processed successfully” messages when processing could not start due to a missing output folder.
- Added no-model-loaded checks for:
  - single image processing
  - image-folder processing
  - single video processing
- Cleaned up duplicate `if file_path:` logic in image loading.
- Improved user-facing status messages for missing setup steps.
- Improved Depth Blender folder processing by matching V1/V2 PNG frames by common filename.
- Added clearer warnings when frame folders contain missing or unmatched PNG files.
- Improved Depth Blender output-folder behavior and validation.
- Improved Depth Blender size-field validation so invalid width/height input is reported instead of silently ignored.
- Improved Depth Blender overwrite-mode UX by disabling unused output-folder controls while overwriting V2 frames.
- Improved video-folder 3D render progress so folder renders now report overall batch progress instead of resetting each video to 0–100%.
- Video-folder 3D renders now explicitly report completion progress when successful, even if later files are skipped due to missing depth matches.
- Improved image/video depth-map matching by only accepting exact filename matches or `_depth` suffix matches, reducing accidental false matches.
- Improved render-service media matching logic to avoid broad `_depth` string replacement false positives.
- Improved render-service batch progress isolation between single renders, video-folder renders, and image-folder renders.

---

## Code Quality and Stability

- Reduced unnecessary always-on profiling overhead.
- Improved global depth pipeline safety by using local snapshots during inference calls.
- Improved retry behavior for depth adapters with different call signatures.
- Added safer batch prediction bounds checking.
- Added clearer processing precondition checks before expensive work begins.
- Made optional depth cleanup behavior explicit instead of always-on.
- Added safer default suspend/cancel event handling for direct render calls outside the GUI path.
- Improved render subprocess cleanup for early exits and generator-based FFmpeg readers.
- Reduced risk of shared global state mutation from edge repair preset dictionaries.
- Improved safety of tensor-to-frame conversion by detaching tensors before NumPy conversion.
- Cleaned up misleading GPU blur helper naming to reflect that the fast GPU smoothing path uses average pooling.
- Improved Depth Blender numerical safety by clamping sigmoid-mask input ranges to avoid overflow warnings.
- Reduced redundant Depth Blender threshold computation during blending.
- Removed avoidable stale-preview rendering work in the standalone Depth Blender UI.
- Hardened `services/depth_service.py` with safer legacy backend integration, better exception chaining, safer progress-callback isolation, improved Tk-style proxy compatibility, and more reliable resource cleanup.
- Cleaned up `render_service.py` imports and safer optional keyframe-service import handling.
- Reduced redundant GPU synchronization points in render/depth performance-critical paths.
- Improved adapter output compatibility by avoiding unnecessary format conversions where the main pipeline already supports the native output type.
- Added stronger frame-shape validation helpers for rawvideo FFmpeg writes.
- Improved blend-mode safety by normalizing frame format and resolution before OpenCV blending.
- Fixed Depth Blender PyTorch helper default arguments so the module can load safely without PyTorch.
- Improved Depth Blender resource cleanup with safer `finally` handling around video capture and writer objects.
- Improved Depth Blender Tkinter thread safety by avoiding Tk object creation from preview worker threads.

---

## Build Note

The NVIDIA/CUDA build and AMD/Intel DirectML build are separate. NVIDIA users should use the CUDA build for best performance.

AMD / Intel users should use the DirectML build once available. The DirectML build is experimental and may vary by GPU, driver, and model type.

ONNX models should now make better use of DirectML where supported.
