import json
from pathlib import Path
from dataclasses import asdict


PRESET_DIR = Path("presets")


# Presets should only store 3D tuning behavior.
# Do not save:
# - input/output paths
# - output format
# - encoding settings
# - aspect ratio
# - color grading
# - VR180 settings
# - preview frame/navigation
# - keyframe file/project state
# - language/theme
PRESET_ALLOWED_KEYS = {
    # Stereo shift
    "fg_shift",
    "mg_shift",
    "bg_shift",
    "sharpness_factor",

    # Depth / parallax
    "max_pixel_shift",
    "zero_parallax_strength",
    "parallax_balance",
    "dof_strength",
    "convergence_strength",
    "enable_dynamic_convergence",
    "edge_repair_quality",

    # Depth shaping / pop
    "depth_pop_gamma",
    "depth_pop_mid",
    "depth_stretch_lo",
    "depth_stretch_hi",
    "fg_pop_multiplier",
    "bg_push_multiplier",

    # Subject / structure controls
    "subject_lock_strength",
    "subject_plane_lock_strength",
    "subject_plane_lock_width",
    "foreground_curvature_strength",

    # Edge / feather behavior
    "feather_strength",
    "blur_ksize",
    "enable_edge_masking",
    "enable_feathering",

    # 3D stability behavior
    "use_subject_tracking",
    "use_floating_window",
    "disable_shift_ema",

    # Stereo scaling
    "ipd_enabled",
    "ipd_scale",
}


class PresetService:
    def __init__(self, preset_dir: str | Path = PRESET_DIR):
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[str]:
        return sorted(p.stem for p in self.preset_dir.glob("*.json"))

    def save_preset(self, state, filename: str) -> Path:
        if not filename.lower().endswith(".json"):
            filename += ".json"

        path = self.preset_dir / filename

        state_data = asdict(state)

        config = {
            key: state_data[key]
            for key in PRESET_ALLOWED_KEYS
            if key in state_data
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        return path

    def load_preset_file(self, path: str | Path) -> dict:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Preset must be a JSON object.")

        return data

    def apply_preset_to_state(self, state, config: dict) -> list[str]:
        if not isinstance(config, dict):
            raise ValueError("Preset must be a JSON object.")

        ignored = []

        for key, value in config.items():
            if key not in PRESET_ALLOWED_KEYS:
                ignored.append(key)
                continue

            if hasattr(state, key):
                try:
                    setattr(state, key, value)
                except Exception:
                    ignored.append(key)
            else:
                ignored.append(key)

        return ignored
