import json
import os
from PySide6.QtCore import QObject, Signal


class LanguageService(QObject):
    language_changed = Signal(str)

    def __init__(self, languages_dir: str = "languages"):
        super().__init__()
        self.languages_dir = languages_dir
        self.current_language = "en"
        self.translations = {}
        self._load_language("en")

    def available_languages(self) -> dict:
        """Return {code: display_name} for all found language files."""
        langs = {}

        if not os.path.isdir(self.languages_dir):
            print(f"[LanguageService] Missing languages folder: {self.languages_dir}")
            return {"en": "English"}

        for fname in os.listdir(self.languages_dir):
            if not fname.endswith(".json"):
                continue

            code = fname[:-5]
            path = os.path.join(self.languages_dir, fname)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                display = data.get("_language_name", code)
                langs[code] = display

            except Exception as e:
                print(f"[LanguageService] Failed to read language file {path}: {e}")
                langs[code] = code

        if "en" not in langs:
            langs["en"] = "English"

        return langs

    def set_language(self, code: str):
        # Always reload, even if the same language is selected again.
        # This helps while editing JSON files during development.
        ok = self._load_language(code)

        if not ok:
            print(f"[LanguageService] Language switch failed: {code}")
            return

        self.current_language = code
        self.language_changed.emit(code)

    def t(self, key: str, default: str = "") -> str:
        return self.translations.get(key, default or key)

    def _load_language(self, code: str) -> bool:
        path = os.path.join(self.languages_dir, f"{code}.json")

        if not os.path.exists(path):
            print(f"[LanguageService] Language file missing: {path}")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.translations = json.load(f)

            return True

        except Exception as e:
            print(f"[LanguageService] Failed to load {path}: {e}")
            self.translations = {}
            return False