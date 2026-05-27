# services/keyframe_service.py

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_KEYFRAME_KEYS = {
    # Main stereo strength controls
    "fg_shift",
    "mg_shift",
    "bg_shift",

    # Parallax / convergence
    "max_pixel_shift_percent",
    "parallax_balance",
    "zero_parallax_strength",
    "convergence_strength",
    "ipd_factor",

    # Depth shaping
    "depth_pop_gamma",
    "depth_pop_mid",
    "depth_stretch_lo",
    "depth_stretch_hi",
    "fg_pop_multiplier",
    "bg_push_multiplier",

    # Subject controls
    "subject_lock_strength",
    "subject_plane_lock_strength",
    "subject_plane_lock_width",
    "foreground_curvature_strength",

    # Optional render look controls
    "dof_strength",
    "color_saturation",
    "color_contrast",
    "color_brightness",
}


def get_default_keyframes_dir() -> Path:
    """
    Default local folder for VisionDepth3D 3D keyframe files.
    Creates:
        keyframes/
    beside the running app.
    """
    folder = Path.cwd() / "keyframes"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def create_default_keyframe_path(source_video_path: str | None = None) -> Path:
    """
    Creates a default keyframe JSON path.

    If a source video is loaded:
        keyframes/SourceVideoName_keyframes.json

    Otherwise:
        keyframes/untitled_keyframes.json
    """
    folder = get_default_keyframes_dir()

    if source_video_path:
        stem = Path(source_video_path).stem.strip()
        if stem:
            return folder / f"{stem}_keyframes.json"

    return folder / "untitled_keyframes.json"

@dataclass
class RenderKeyframe:
    frame: int
    label: str = ""
    transition_frames: int = 0
    transition_type: str = "smoothstep"
    settings: dict[str, float] = field(default_factory=dict)


class KeyframeService:
    """
    Section-based 3D render keyframes.

    Rule:
    A keyframe defines the active 3D settings from that frame forward.
    Those settings stay active until the next keyframe.

    If transition_frames > 0, the render eases from the previous keyframe
    into the new keyframe over that many frames.
    """

    def __init__(self, keyframes: list[RenderKeyframe] | None = None):
        self.keyframes = sorted(keyframes or [], key=lambda k: int(k.frame))

    def get_default_keyframes_dir() -> Path:
        """
        Default local folder for VisionDepth3D 3D keyframe files.
        """
        folder = Path.cwd() / "keyframes"
        folder.mkdir(parents=True, exist_ok=True)
        return folder


    def create_default_keyframe_path(source_video_path: str | None = None) -> Path:
        """
        Creates a default keyframe file path based on the loaded source video.
        """
        folder = get_default_keyframes_dir()

        if source_video_path:
            stem = Path(source_video_path).stem.strip()
            if stem:
                return folder / f"{stem}_keyframes.json"

        return folder / "untitled_keyframes.json"

    @classmethod
    def load(cls, path: str | Path) -> "KeyframeService":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeService":
        raw_items = data.get("keyframes", []) or []
        keyframes: list[RenderKeyframe] = []

        for item in raw_items:
            try:
                frame = int(item.get("frame", 0))
                label = str(item.get("label", "") or "")
                transition_frames = int(item.get("transition_frames", 0) or 0)
                transition_type = str(item.get("transition_type", "smoothstep") or "smoothstep").lower()

                raw_settings = item.get("settings", {}) or {}
                settings: dict[str, float] = {}

                for key, value in raw_settings.items():
                    key = str(key)
                    if key not in SUPPORTED_KEYFRAME_KEYS:
                        continue
                    if value is None:
                        continue
                    settings[key] = float(value)

                keyframes.append(
                    RenderKeyframe(
                        frame=max(0, frame),
                        label=label,
                        transition_frames=max(0, transition_frames),
                        transition_type=transition_type,
                        settings=settings,
                    )
                )
            except Exception as e:
                print(f"[Keyframes] Skipped invalid keyframe entry: {e}")

        return cls(keyframes=keyframes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "type": "visiondepth3d_3d_parameter_keyframes",
            "keyframes": [
                {
                    "frame": int(kf.frame),
                    "label": kf.label,
                    "transition_frames": int(kf.transition_frames),
                    "transition_type": kf.transition_type,
                    "settings": dict(kf.settings),
                }
                for kf in self.keyframes
            ],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def has_keyframes(self) -> bool:
        return bool(self.keyframes)

    def upsert_keyframe(
        self,
        frame: int,
        settings: dict[str, float],
        label: str = "",
        transition_frames: int = 24,
        transition_type: str = "smoothstep",
    ) -> None:
        """
        Add or replace a keyframe at the exact frame.
        """
        frame = max(0, int(frame))
        clean_settings = {}

        for key, value in (settings or {}).items():
            if key not in SUPPORTED_KEYFRAME_KEYS:
                continue
            if value is None:
                continue
            clean_settings[str(key)] = float(value)

        # Remove any old keyframe on this same frame.
        self.keyframes = [kf for kf in self.keyframes if int(kf.frame) != frame]

        self.keyframes.append(
            RenderKeyframe(
                frame=frame,
                label=str(label or ""),
                transition_frames=max(0, int(transition_frames or 0)),
                transition_type=str(transition_type or "smoothstep").lower(),
                settings=clean_settings,
            )
        )

        self.keyframes = sorted(self.keyframes, key=lambda k: int(k.frame))


    def remove_keyframe_at_frame(self, frame: int) -> bool:
        """
        Remove a keyframe by frame number.
        Returns True if something was removed.
        """
        frame = int(frame)
        before = len(self.keyframes)
        self.keyframes = [kf for kf in self.keyframes if int(kf.frame) != frame]
        return len(self.keyframes) != before


    def get_keyframe_at_frame(self, frame: int):
        frame = int(frame)
        for kf in self.keyframes:
            if int(kf.frame) == frame:
                return kf
        return None

def get_settings_for_frame(
    self,
    frame_idx: int,
    base_settings: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Section-based keyframe behavior.

    A keyframe defines the active settings from that frame forward.

    If a keyframe has transition_frames > 0, the renderer eases from the
    previous keyframe into this keyframe starting at this keyframe frame.
    After the transition ends, the new keyframe settings are held until
    the next keyframe.
    """
    base = dict(base_settings or {})

    if not self.keyframes:
        return base

    frame_idx = max(0, int(frame_idx))

    # Before first keyframe, use first keyframe settings.
    if frame_idx < self.keyframes[0].frame:
        out = dict(base)
        out.update(self.keyframes[0].settings)
        return out

    # Find the active keyframe at or before this frame.
    active_index = 0
    for i, kf in enumerate(self.keyframes):
        if kf.frame <= frame_idx:
            active_index = i
        else:
            break

    active_kf = self.keyframes[active_index]

    # First keyframe or no previous keyframe, just hold active settings.
    if active_index == 0:
        out = dict(base)
        out.update(active_kf.settings)
        return out

    prev_kf = self.keyframes[active_index - 1]

    transition_frames = int(active_kf.transition_frames or 0)

    # No transition, hard cut into active keyframe.
    if transition_frames <= 0:
        out = dict(base)
        out.update(active_kf.settings)
        return out

    transition_start = int(active_kf.frame)
    transition_end = transition_start + transition_frames

    # Inside transition window, blend previous keyframe into active keyframe.
    if transition_start <= frame_idx <= transition_end:
        t = (frame_idx - transition_start) / float(max(1, transition_frames))
        t = max(0.0, min(1.0, t))

        if active_kf.transition_type == "smoothstep":
            t = t * t * (3.0 - 2.0 * t)
        elif active_kf.transition_type == "linear":
            pass
        else:
            t = 1.0

        blended = self._blend_settings(prev_kf.settings, active_kf.settings, t)

        out = dict(base)
        out.update(blended)
        return out

    # After transition window, hold active keyframe settings.
    out = dict(base)
    out.update(active_kf.settings)
    return out

    @staticmethod
    def _blend_settings(
        a: dict[str, float],
        b: dict[str, float],
        t: float,
    ) -> dict[str, float]:
        """
        Blend only between keys that exist in either keyframe.

        If a key is missing from one side, it holds the value from the other side.
        """
        keys = set(a.keys()) | set(b.keys())
        out: dict[str, float] = {}

        for key in keys:
            av = float(a.get(key, b.get(key, 0.0)))
            bv = float(b.get(key, a.get(key, 0.0)))
            out[key] = av + (bv - av) * t

        return out