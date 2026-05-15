import json
import os
import sys
from copy import deepcopy

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette


def resource_root():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.abspath(".")

def app_root():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")


DEFAULT_THEMES = {
    "dark": {
        "name": "Dark",
        "colors": {
            "bg": "#0b0f14",
            "topbar": "#0f141a",
            "panel": "#111821",
            "panel_2": "#0d131b",
            "panel_3": "#151d29",
            "border": "#263445",
            "border_soft": "#2d3b4f",
            "text": "#e6edf3",
            "text_bright": "#f0f6fc",
            "muted": "#8b949e",
            "accent": "system",
            "accent_text": "#ffffff",
            "danger": "#ff7b72",
            "danger_bg": "#2b1518",
            "warning": "#f0b429",
            "success": "#3fb950",
            "preview_bg": "#0d131b",
        },
    },
    "midnight": {
        "name": "Midnight",
        "colors": {
            "bg": "#070b12",
            "topbar": "#0a1020",
            "panel": "#0f172a",
            "panel_2": "#0b1220",
            "panel_3": "#162033",
            "border": "#25324a",
            "border_soft": "#32415f",
            "text": "#e5e7eb",
            "text_bright": "#f8fafc",
            "muted": "#94a3b8",
            "accent": "#60a5fa",
            "accent_text": "#ffffff",
            "danger": "#fb7185",
            "danger_bg": "#31131b",
            "warning": "#fbbf24",
            "success": "#34d399",
            "preview_bg": "#08111f",
        },
    },
    "forest": {
        "name": "Forest",
        "colors": {
            "bg": "#07110d",
            "topbar": "#0b1711",
            "panel": "#102018",
            "panel_2": "#0b1711",
            "panel_3": "#14291f",
            "border": "#244334",
            "border_soft": "#315744",
            "text": "#e8f5ee",
            "text_bright": "#f3fff8",
            "muted": "#9ab8a8",
            "accent": "#35d07f",
            "accent_text": "#04130b",
            "danger": "#ff7b72",
            "danger_bg": "#2b1518",
            "warning": "#e5c07b",
            "success": "#35d07f",
            "preview_bg": "#07130d",
        },
    },
    "light": {
        "name": "Light",
        "colors": {
            "bg": "#f4f7fb",
            "topbar": "#ffffff",
            "panel": "#ffffff",
            "panel_2": "#eef3f8",
            "panel_3": "#e7eef6",
            "border": "#c9d6e4",
            "border_soft": "#b8c7d8",
            "text": "#17202a",
            "text_bright": "#0b1220",
            "muted": "#5f6f82",
            "accent": "system",
            "accent_text": "#ffffff",
            "danger": "#b42318",
            "danger_bg": "#fff1f0",
            "warning": "#9a6700",
            "success": "#1a7f37",
            "preview_bg": "#eef3f8",
        },
    },
}


class ThemeService:
    def __init__(self, themes_dir=None):
        self.builtin_themes_dir = os.path.join(resource_root(), "resources", "themes")
        self.user_themes_dir = themes_dir or os.path.join(app_root(), "themes")

        os.makedirs(self.user_themes_dir, exist_ok=True)

        self._themes = self._load_themes()

    def _system_accent_hex(self):
        app = QApplication.instance()
        if app is not None:
            color = app.palette().color(QPalette.ColorRole.Highlight)
            if color.isValid():
                return color.name()
        return "#2f81f7"

    def _normalize_theme(self, theme_id, theme):
        result = deepcopy(DEFAULT_THEMES["dark"])
        result["id"] = theme_id
        result["_qss_template"] = ""

        if isinstance(theme, dict):
            result["name"] = theme.get("name", theme_id.title())
            result["_qss_template"] = theme.get("_qss_template", "")

            colors = theme.get("colors", {})
            if isinstance(colors, dict):
                result["colors"].update(colors)

        if result["colors"].get("accent") == "system":
            result["colors"]["accent"] = self._system_accent_hex()

        return result
        
    def _load_qss_for_theme(self, folder, theme_id, theme_data):
        qss_name = theme_data.get("qss")

        # If JSON does not specify a qss file, try matching name automatically:
        # neon_blue.json -> neon_blue.qss
        if not qss_name:
            qss_name = f"{theme_id}.qss"

        qss_path = os.path.join(folder, qss_name)

        if not os.path.exists(qss_path):
            return ""

        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            print(f"[ThemeService] Failed to load QSS {qss_path}: {exc}")
            return ""

    def _load_theme_folder(self, folder, themes):
        if not os.path.isdir(folder):
            return

        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):
                continue

            path = os.path.join(folder, name)
            theme_id = os.path.splitext(name)[0]

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    print(f"[ThemeService] Invalid theme JSON: {path}")
                    continue

                data["_qss_template"] = self._load_qss_for_theme(folder, theme_id, data)
                themes[theme_id] = self._normalize_theme(theme_id, data)

            except Exception as exc:
                print(f"[ThemeService] Failed to load {path}: {exc}")

    def _load_themes(self):
        themes = {}

        # Safe fallback themes from code
        for theme_id, theme in DEFAULT_THEMES.items():
            themes[theme_id] = self._normalize_theme(theme_id, theme)

        # Shipped themes
        self._load_theme_folder(self.builtin_themes_dir, themes)

        # User themes override shipped themes
        self._load_theme_folder(self.user_themes_dir, themes)

        return themes

    def reload(self):
        self._themes = self._load_themes()

    def available_themes(self):
        return self._themes

    def get_theme(self, theme_id):
        if theme_id not in self._themes:
            theme_id = "dark"
        return self._themes[theme_id]