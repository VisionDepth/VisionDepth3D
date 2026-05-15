import json
import os
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QColorDialog,
    QMessageBox,
    QWidget,
    QFrame,
)


DEFAULT_THEME_COLORS = {
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
    "accent": "#2f81f7",
    "accent_text": "#ffffff",
    "danger": "#ff7b72",
    "danger_bg": "#2b1518",
    "warning": "#f0b429",
    "success": "#3fb950",
    "preview_bg": "#0d131b",
}


COLOR_GROUPS = [
    (
        "Core",
        [
            ("bg", "Background"),
            ("topbar", "Top Bar"),
            ("panel", "Panel"),
            ("panel_2", "Panel Dark"),
            ("panel_3", "Panel Raised"),
            ("preview_bg", "Preview Area"),
        ],
    ),
    (
        "Borders",
        [
            ("border", "Border"),
            ("border_soft", "Soft Border"),
        ],
    ),
    (
        "Text",
        [
            ("text", "Text"),
            ("text_bright", "Bright Text"),
            ("muted", "Muted Text"),
            ("accent_text", "Accent Text"),
        ],
    ),
    (
        "Accents",
        [
            ("accent", "Accent"),
            ("danger", "Danger"),
            ("danger_bg", "Danger BG"),
            ("warning", "Warning"),
            ("success", "Success"),
        ],
    ),
]


class ColorTile(QPushButton):
    def __init__(self, key: str, label: str, color: str, accent: str = "#ffffff"):
        super().__init__()
        self.key = key
        self.label = label
        self.color = color
        self.accent = accent

        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(118, 74)
        self.setMaximumHeight(84)
        self.setText(f"{label}\n{color}")
        self._apply_style()

    def set_color(self, color: str):
        self.color = color
        self.setText(f"{self.label}\n{color}")
        self._apply_style()

    def _apply_style(self):
        text_color = "#ffffff"

        qcolor = QColor(self.color)
        if qcolor.isValid():
            brightness = (
                qcolor.red() * 0.299
                + qcolor.green() * 0.587
                + qcolor.blue() * 0.114
            )
            text_color = "#111111" if brightness > 160 else "#ffffff"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                color: {text_color};
                border: 1px solid rgba(255, 255, 255, 0.22);
                border-radius: 10px;
                padding: 8px;
                text-align: left;
                font-weight: 700;
            }}

            QPushButton:hover {{
                border: 2px solid {self.accent};
            }}
        """)

class ThemeCreatorDialog(QDialog):
    def __init__(self, themes_dir: str, parent=None, base_theme: dict | None = None):
        super().__init__(parent)

        self.themes_dir = themes_dir
        os.makedirs(self.themes_dir, exist_ok=True)

        self.saved_theme_id = None
        self.color_tiles = {}
        self.base_theme = base_theme or {}

        self.colors = DEFAULT_THEME_COLORS.copy()
        if isinstance(base_theme, dict):
            self.colors.update(base_theme.get("colors", {}) or {})

        self.setWindowTitle("Create Theme")
        self.setMinimumSize(980, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(12)

        title_box = QVBoxLayout()
        title = QLabel("Theme Studio")
        title.setObjectName("ThemeStudioTitle")

        subtitle = QLabel("Build a custom VisionDepth3D color theme using live preview swatches.")
        subtitle.setObjectName("ThemeStudioSubtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header.addLayout(title_box)
        header.addStretch()

        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(16)

        left_panel = QFrame()
        left_panel.setObjectName("ThemePanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(14)

        name_label = QLabel("Theme Name")
        name_label.setObjectName("SectionLabel")

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Example: Purple Night")

        left_layout.addWidget(name_label)
        left_layout.addWidget(self.name_edit)

        preview_label = QLabel("Live Preview")
        preview_label.setObjectName("SectionLabel")
        left_layout.addWidget(preview_label)

        self.preview_card = QFrame()
        self.preview_card.setObjectName("PreviewCard")
        preview_layout = QVBoxLayout(self.preview_card)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        preview_layout.setSpacing(12)

        self.fake_topbar = QFrame()
        self.fake_topbar.setObjectName("FakeTopBar")
        fake_topbar_layout = QHBoxLayout(self.fake_topbar)
        fake_topbar_layout.setContentsMargins(10, 8, 10, 8)

        self.fake_title = QLabel("VisionDepth3D")
        self.fake_nav = QLabel("3D Generator   Depth Engine   FPS/Upscale")
        fake_topbar_layout.addWidget(self.fake_title)
        fake_topbar_layout.addStretch()
        fake_topbar_layout.addWidget(self.fake_nav)

        preview_layout.addWidget(self.fake_topbar)

        self.fake_panel = QFrame()
        self.fake_panel.setObjectName("FakePanel")
        fake_panel_layout = QVBoxLayout(self.fake_panel)
        fake_panel_layout.setContentsMargins(12, 12, 12, 12)
        fake_panel_layout.setSpacing(10)

        self.fake_panel_title = QLabel("Preview Panel")
        self.fake_panel_title.setObjectName("FakePanelTitle")

        self.fake_muted = QLabel("This shows how panels, text, buttons, and accents will feel.")
        self.fake_muted.setObjectName("FakeMuted")

        self.fake_input = QLineEdit()
        self.fake_input.setText("Sample input field")

        self.fake_button = QPushButton("Primary Action")
        self.fake_button.setObjectName("PrimaryButton")

        self.fake_danger = QPushButton("Danger Action")
        self.fake_danger.setObjectName("DangerButton")

        fake_panel_layout.addWidget(self.fake_panel_title)
        fake_panel_layout.addWidget(self.fake_muted)
        fake_panel_layout.addWidget(self.fake_input)
        fake_panel_layout.addWidget(self.fake_button)
        fake_panel_layout.addWidget(self.fake_danger)

        preview_layout.addWidget(self.fake_panel)

        left_layout.addWidget(self.preview_card, 1)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save Theme")
        self.save_btn.setObjectName("PrimaryButton")

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self._save_theme)

        button_row.addWidget(self.cancel_btn)
        button_row.addWidget(self.save_btn)

        left_layout.addLayout(button_row)

        right_panel = QFrame()
        right_panel.setObjectName("ThemePanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        palette_title = QLabel("Theme Colors")
        palette_title.setObjectName("SectionLabel")
        right_layout.addWidget(palette_title)

        palette_hint = QLabel("Click any color block to adjust it.")
        palette_hint.setObjectName("ThemeStudioSubtitle")
        right_layout.addWidget(palette_hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        row = 0
        for group_name, items in COLOR_GROUPS:
            group_label = QLabel(group_name)
            group_label.setObjectName("MiniSectionLabel")
            grid.addWidget(group_label, row, 0, 1, 3)
            row += 1

            col = 0
            for key, label in items:
                theme_colors = self.base_theme.get("colors", {}) if isinstance(self.base_theme, dict) else {}
                tile_accent = theme_colors.get("accent", self.colors.get("accent", "#ffffff"))

                tile = ColorTile(key, label, self.colors.get(key, "#000000"), tile_accent)
                tile.clicked.connect(lambda checked=False, k=key: self._pick_color(k))

                self.color_tiles[key] = tile
                grid.addWidget(tile, row, col)

                col += 1
                if col >= 3:
                    col = 0
                    row += 1

            if col != 0:
                row += 1

        right_layout.addLayout(grid)
        right_layout.addStretch()

        body.addWidget(left_panel, 1)
        body.addWidget(right_panel, 1)

        root.addLayout(body, 1)

        self._apply_dialog_style()
        self._update_preview()

    def _apply_dialog_style(self):
        theme = getattr(self, "base_theme", {}) or {}
        colors = theme.get("colors", {}) if isinstance(theme, dict) else {}

        bg = colors.get("bg", "#080611")
        panel = colors.get("panel", "#10091d")
        panel_2 = colors.get("panel_2", "#080611")
        panel_3 = colors.get("panel_3", "#160d28")
        border = colors.get("border", "#35215f")
        border_soft = colors.get("border_soft", "#5b36a0")
        text = colors.get("text", "#f4edff")
        text_bright = colors.get("text_bright", "#ffffff")
        muted = colors.get("muted", "#a99bc4")
        accent = colors.get("accent", "#a855f7")
        accent_text = colors.get("accent_text", "#ffffff")

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg};
                color: {text};
                font-family: "Segoe UI";
                font-size: 13px;
            }}

            QLabel {{
                background: transparent;
                color: {text};
            }}

            QLabel#ThemeStudioTitle {{
                font-size: 24px;
                font-weight: 900;
                color: {text_bright};
            }}

            QLabel#ThemeStudioSubtitle {{
                color: {muted};
                font-size: 12px;
            }}

            QLabel#SectionLabel {{
                background-color: {panel_2};
                color: {text_bright};
                font-weight: 900;
                font-size: 14px;
                padding: 4px 6px;
                border-radius: 4px;
            }}

            QLabel#MiniSectionLabel {{
                background-color: {panel_2};
                color: {muted};
                font-weight: 800;
                font-size: 11px;
                text-transform: uppercase;
                padding: 5px 6px;
                border-radius: 4px;
            }}

            QFrame#ThemePanel {{
                background-color: {panel};
                border: 1px solid {border};
                border-radius: 14px;
            }}

            QLineEdit {{
                background-color: {panel_2};
                border: 1px solid {border_soft};
                border-radius: 8px;
                padding: 8px 10px;
                color: {text};
            }}

            QLineEdit:hover,
            QLineEdit:focus {{
                border: 1px solid {accent};
            }}

            QPushButton {{
                background-color: {panel_3};
                border: 1px solid {border_soft};
                border-radius: 8px;
                padding: 8px 14px;
                color: {text_bright};
                font-weight: 700;
            }}

            QPushButton:hover {{
                border: 1px solid {accent};
            }}

            QPushButton#PrimaryButton {{
                background-color: {accent};
                border: 1px solid {accent};
                color: {accent_text};
            }}

            QPushButton#PrimaryButton:hover {{
                background-color: {accent};
                border: 1px solid {text_bright};
            }}
        """)

    def _slugify(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = text.strip("_")
        return text or "custom_theme"

    def _pick_color(self, key: str):
        current = QColor(self.colors.get(key, "#000000"))

        dialog = QColorDialog(current, self)
        dialog.setWindowTitle(f"Pick {key}")
        dialog.setOption(QColorDialog.ShowAlphaChannel, False)

        if dialog.exec() != QColorDialog.Accepted:
            return

        color = dialog.selectedColor()
        if not color.isValid():
            return

        self.colors[key] = color.name()

        tile = self.color_tiles.get(key)
        if tile:
            tile.set_color(color.name())

        self._update_preview()

    def _update_preview(self):
        c = self.colors

        self.preview_card.setStyleSheet(f"""
            QFrame#PreviewCard {{
                background-color: {c.get("bg")};
                border: 1px solid {c.get("border")};
                border-radius: 12px;
            }}

            QFrame#FakeTopBar {{
                background-color: {c.get("topbar")};
                border: 1px solid {c.get("border")};
                border-radius: 8px;
            }}

            QLabel {{
                color: {c.get("text")};
                background: transparent;
            }}

            QLabel#FakePanelTitle {{
                color: {c.get("text_bright")};
                font-weight: 900;
                font-size: 15px;
            }}

            QLabel#FakeMuted {{
                color: {c.get("muted")};
            }}

            QFrame#FakePanel {{
                background-color: {c.get("panel")};
                border: 1px solid {c.get("border_soft")};
                border-radius: 10px;
            }}

            QLineEdit {{
                background-color: {c.get("panel_2")};
                border: 1px solid {c.get("border_soft")};
                border-radius: 7px;
                padding: 8px 10px;
                color: {c.get("text")};
            }}

            QPushButton {{
                background-color: {c.get("panel_3")};
                border: 1px solid {c.get("border_soft")};
                border-radius: 8px;
                padding: 8px 12px;
                color: {c.get("text_bright")};
                font-weight: 800;
            }}

            QPushButton#PrimaryButton {{
                background-color: {c.get("accent")};
                border: 1px solid {c.get("accent")};
                color: {c.get("accent_text")};
            }}

            QPushButton#DangerButton {{
                background-color: {c.get("danger_bg")};
                border: 1px solid {c.get("danger")};
                color: {c.get("text_bright")};
            }}
        """)

    def _save_theme(self):
        theme_name = self.name_edit.text().strip()

        if not theme_name:
            QMessageBox.warning(self, "Missing Theme Name", "Please enter a theme name.")
            return

        theme_id = self._slugify(theme_name)
        theme_path = os.path.join(self.themes_dir, f"{theme_id}.json")

        if os.path.exists(theme_path):
            overwrite = QMessageBox.question(
                self,
                "Overwrite Theme?",
                f"A theme named '{theme_id}' already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if overwrite != QMessageBox.Yes:
                return

        theme_data = {
            "name": theme_name,
            "colors": self.colors,
        }

        try:
            with open(theme_path, "w", encoding="utf-8") as f:
                json.dump(theme_data, f, indent=2)

        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return

        self.saved_theme_id = theme_id
        self.accept()
