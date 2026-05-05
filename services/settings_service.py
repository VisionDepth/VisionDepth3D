import json
from pathlib import Path


class SettingsService:
    def __init__(self, filename: str = "settings.json"):
        self.path = Path(filename)

    def load(self) -> dict:
        if not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load settings: {e}")
            return {}

    def save(self, data: dict):
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")