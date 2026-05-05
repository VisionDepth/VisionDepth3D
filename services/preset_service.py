import json
from pathlib import Path
from dataclasses import asdict


PRESET_DIR = Path("presets")


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

        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=4)

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
            if hasattr(state, key):
                try:
                    setattr(state, key, value)
                except Exception:
                    ignored.append(key)
            else:
                ignored.append(key)

        return ignored