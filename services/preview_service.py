from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.app_state import AppState
from core.render_3d import (
    frame_to_tensor,
    depth_to_tensor,
    pixel_shift_cuda,
    apply_sharpening,
    tensor_to_frame,
    format_3d_output,
    apply_color_grade,
)
from core.preview_utils import generate_preview_image


@dataclass
class PreviewResult:
    image_bgr: Optional[np.ndarray]
    input_frame_bgr: Optional[np.ndarray]
    depth_frame_bgr: Optional[np.ndarray]
    total_frames: int


class PreviewService:
    def __init__(self):
        self.preview_cap = None
        self.depth_cap = None
        self.input_total = 0
        self.depth_total = 0

    def open_sources(self, input_path: str, depth_path: str) -> int:
        self.close_sources()

        self.preview_cap = cv2.VideoCapture(input_path)
        self.depth_cap = cv2.VideoCapture(depth_path)

        if not self.preview_cap.isOpened():
            self.close_sources()
            raise RuntimeError(f"Failed to open input video:\n{input_path}")

        if not self.depth_cap.isOpened():
            self.close_sources()
            raise RuntimeError(f"Failed to open depth video:\n{depth_path}")

        self.input_total = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.depth_total = int(self.depth_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if self.input_total <= 0:
            self.input_total = 1

        if self.input_total > 0 and self.depth_total > 0:
            total = min(self.input_total, self.depth_total)
        else:
            total = self.input_total

        return max(1, total)

    def close_sources(self):
        if self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None

        if self.depth_cap is not None:
            self.depth_cap.release()
            self.depth_cap = None

        self.input_total = 0
        self.depth_total = 0

    def _get_frame(self, capture, frame_idx: int):
        if capture is None:
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        return frame if ret else None

    def _reset_preview_temporal_state(self):
        """
        Matches the old preview window behavior when scrubbing,
        so temporal smoothing does not carry across unrelated frames.
        """
        try:
            from core import render_3d

            render_3d.pixel_shift_cuda._shift_ema = None
            render_3d.subject_depth_ema.val = None
            render_3d.depth_ema_norm._lo = None
            render_3d.depth_ema_norm._hi = None
            render_3d.conv_ema.val = None
            render_3d.floating_window_tracker.prev_offset = 0.0
            render_3d.floating_window_tracker.frame_counter = 0
        except Exception:
            pass

    def _draw_convergence_guides(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None:
            return img_bgr

        out = img_bgr.copy()
        h, w = out.shape[:2]

        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        row_mean = gray.mean(axis=1)

        thresh = 8.0
        active_rows = np.where(row_mean > thresh)[0]

        if active_rows.size > 0:
            y0 = int(active_rows[0])
            y1 = int(active_rows[-1])
        else:
            y0, y1 = 0, h - 1

        pad = max(2, int(min(w, h) * 0.01))
        y0 = max(0, y0 + pad)
        y1 = min(h - 1, y1 - pad)

        ax0, ax1 = 0, w - 1
        ay0, ay1 = y0, y1
        aw = ax1 - ax0 + 1
        ah = ay1 - ay0 + 1

        cx = (ax0 + ax1) // 2
        cy = (ay0 + ay1) // 2

        base = min(aw, ah)
        arm = max(10, base // 55)
        gap = max(6, base // 110)
        thickness = 2
        col = (235, 235, 235)

        cv2.line(out, (cx - arm, cy), (cx - gap, cy), col, thickness, cv2.LINE_AA)
        cv2.line(out, (cx + gap, cy), (cx + arm, cy), col, thickness, cv2.LINE_AA)
        cv2.line(out, (cx, cy - arm), (cx, cy - gap), col, thickness, cv2.LINE_AA)
        cv2.line(out, (cx, cy + gap), (cx, cy + arm), col, thickness, cv2.LINE_AA)

        cv2.circle(out, (cx, cy), 3, col, -1, cv2.LINE_AA)

        marker_size = max(12, base // 35)
        for gx in (ax0 + aw // 4, cx, ax0 + (3 * aw) // 4):
            for gy in (ay0 + ah // 4, cy, ay0 + (3 * ah) // 4):
                cv2.drawMarker(
                    out,
                    (gx, gy),
                    (200, 200, 200),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

        return out

    def generate_preview(
        self,
        state: AppState,
        frame_idx: int,
        preview_mode: str = "Red-Blue Anaglyph",
        ipd_enabled: bool = True,
        ipd_scale: float = 1.0,
    ) -> PreviewResult:
        if self.preview_cap is None or self.depth_cap is None:
            return PreviewResult(None, None, None, 0)

        total_frames = max(1, min(self.input_total, self.depth_total or self.input_total))
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        self._reset_preview_temporal_state()

        frame = self._get_frame(self.preview_cap, frame_idx)
        if frame is None:
            return PreviewResult(None, None, None, total_frames)

        depth_idx = frame_idx
        if self.depth_total > 0 and depth_idx >= self.depth_total:
            depth_idx = self.depth_total - 1

        depth = self._get_frame(self.depth_cap, depth_idx)
        if depth is None:
            return PreviewResult(None, frame, None, total_frames)
        
        h, w = frame.shape[:2]
        dh, dw = depth.shape[:2]

        # 🔍 DIAGNOSTIC
        print(f"🔍 PREVIEW DIAGNOSTIC: frame.shape={frame.shape}, depth.shape={depth.shape}")
        print(f"🔍 PREVIEW DIAGNOSTIC: w={w}, h={h}, dw={dw}, dh={dh}")

        # Force match
        if (h != dh) or (w != dw):
            print(f"🔍 MISMATCH DETECTED: resizing depth from {dw}x{dh} to {w}x{h}")
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Force frame and depth to the same dimensions.
        # Use the frame dimensions as the master target.
        if (h != dh) or (w != dw):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Now both are guaranteed to be (h, w)
        frame_tensor = frame_to_tensor(frame)
        depth_tensor = depth_to_tensor(depth)

        # Only interpolate if we actually need to change resolution.
        # Since we already matched dimensions above, these are no-ops,
        # but keep them for any downstream size enforcement.
        frame_tensor = F.interpolate(
            frame_tensor.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        fg_val = float(state.fg_shift)
        mg_val = float(state.mg_shift)
        bg_val = float(state.bg_shift)

        if ipd_enabled:
            fg_val *= ipd_scale
            mg_val *= ipd_scale
            bg_val *= ipd_scale

        # Apply the same ShiftSmoother the render path uses so preview matches render output
        from core.render_3d import ShiftSmoother
        if not hasattr(self, '_preview_smoother'):
            self._preview_smoother = ShiftSmoother(alpha=0.5)
        fg_val, mg_val, bg_val = self._preview_smoother.smooth(fg_val, mg_val, bg_val)

        print(f"[PREVIEW] fg={fg_val:.2f} mg={mg_val:.2f} bg={bg_val:.2f} balance={state.parallax_balance:.3f} max_shift={state.max_pixel_shift:.4f}")
        
        left_tensor, right_tensor, shift_meta = pixel_shift_cuda(
            frame_tensor,
            depth_tensor,
            w,
            h,
            fg_val,
            mg_val,
            bg_val,
            return_shift_map=True,
            use_subject_tracking=state.use_subject_tracking,
            enable_floating_window=state.use_floating_window,
            max_pixel_shift_percent=state.max_pixel_shift,
            zero_parallax_strength=state.zero_parallax_strength,
            parallax_balance=state.parallax_balance,
            enable_edge_masking=state.enable_edge_masking,
            enable_feathering=state.enable_feathering,
            dof_strength=state.dof_strength,
            convergence_strength=state.convergence_strength,
            enable_dynamic_convergence=state.enable_dynamic_convergence,
            depth_pop_gamma=state.depth_pop_gamma,
            depth_pop_mid=state.depth_pop_mid,
            depth_stretch_lo=state.depth_stretch_lo,
            depth_stretch_hi=state.depth_stretch_hi,
            fg_pop_multiplier=state.fg_pop_multiplier,
            bg_push_multiplier=state.bg_push_multiplier,
            subject_lock_strength=state.subject_lock_strength,
            disable_shift_ema=getattr(state, "disable_shift_ema", False),
        )

        preview_shift_map = None
        if isinstance(shift_meta, dict):
            preview_shift_map = shift_meta.get("shift_map", None)
        else:
            preview_shift_map = shift_meta

        left_frame = tensor_to_frame(left_tensor) if isinstance(left_tensor, torch.Tensor) else left_tensor
        right_frame = tensor_to_frame(right_tensor) if isinstance(right_tensor, torch.Tensor) else right_tensor

        left_frame = apply_sharpening(left_frame, float(state.sharpness_factor))
        right_frame = apply_sharpening(right_frame, float(state.sharpness_factor))

        lt = apply_color_grade(
            frame_to_tensor(left_frame),
            saturation=state.saturation,
            contrast=state.contrast,
            brightness=state.brightness,
        )
        rt = apply_color_grade(
            frame_to_tensor(right_frame),
            saturation=state.saturation,
            contrast=state.contrast,
            brightness=state.brightness,
        )

        left_frame = tensor_to_frame(lt)
        right_frame = tensor_to_frame(rt)

        preview_shift_map = None
        if isinstance(shift_meta, dict):
            preview_shift_map = shift_meta.get("shift_map")
        else:
            preview_shift_map = shift_meta

        if preview_mode in ("HSBS", "Half-SBS"):
            preview_img = format_3d_output(left_frame, right_frame, "Half-SBS")
        else:
            preview_img = generate_preview_image(
                preview_mode,
                left_frame,
                right_frame,
                preview_shift_map,
                w,
                h,
            )

        if preview_img is not None and getattr(state, "show_convergence_guides", False):
            preview_img = self._draw_convergence_guides(preview_img)

        return PreviewResult(preview_img, frame, depth, total_frames)

    def save_preview(self, save_path: str, image_bgr: np.ndarray) -> None:
        if image_bgr is None:
            raise ValueError("No preview image to save.")

        ok = cv2.imwrite(save_path, image_bgr)
        if not ok:
            raise RuntimeError(f"Failed to save preview image:\n{save_path}")