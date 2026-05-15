def apply_unified_page_theme(widget, theme=None):
    theme = theme or {}
    colors = theme.get("colors", {}) if isinstance(theme, dict) else {}

    bg = colors.get("bg", "#0b0f14")
    panel = colors.get("panel", "#111821")
    panel_2 = colors.get("panel_2", "#0d131b")
    panel_3 = colors.get("panel_3", "#151d29")
    border = colors.get("border", "#263445")
    border_soft = colors.get("border_soft", "#2d3b4f")
    text = colors.get("text", "#e6edf3")
    text_bright = colors.get("text_bright", "#f0f6fc")
    muted = colors.get("muted", "#8b949e")
    accent = colors.get("accent", "#2f81f7")
    accent_text = colors.get("accent_text", "#ffffff")
    danger = colors.get("danger", "#ff7b72")
    danger_bg = colors.get("danger_bg", "#2b1518")
    preview_bg = colors.get("preview_bg", panel_2)

    stylesheet = theme.get("_qss_template", "")

    if not stylesheet:
        stylesheet = """
        QWidget {
            background-color: __BG__;
            color: __TEXT__;
            font-family: "Segoe UI";
            font-size: 13px;
        }

        QLabel#PageTitle {
            font-size: 22px;
            font-weight: 800;
            color: __TEXT_BRIGHT__;
        }

        QLabel#PageSubtitle {
            color: __MUTED__;
        }

        QGroupBox {
            background-color: __PANEL__;
            border: 1px solid __BORDER__;
            border-radius: 4px;
            margin-top: 10px;
            padding: 10px;
            font-weight: 800;
            color: __TEXT_BRIGHT__;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: __TEXT_BRIGHT__;
        }

        QLineEdit,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox {
            background-color: __PANEL_2__;
            border: 1px solid __BORDER__;
            border-radius: 3px;
            padding: 5px 8px;
            color: __TEXT__;
        }

        QPushButton {
            background-color: __PANEL_3__;
            border: 1px solid __BORDER_SOFT__;
            border-radius: 4px;
            padding: 6px 10px;
            color: __TEXT_BRIGHT__;
            font-weight: 600;
        }

        QPushButton:hover {
            border: 1px solid __ACCENT__;
        }

        QSlider::groove:horizontal {
            height: 5px;
            background: __PANEL_2__;
            border: 1px solid __BORDER__;
            border-radius: 2px;
        }

        QSlider::handle:horizontal {
            background: __ACCENT__;
            border: 1px solid __ACCENT__;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }

        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            border: 1px solid __BORDER_SOFT__;
            background-color: __PANEL_2__;
        }

        QCheckBox::indicator:checked {
            background-color: __ACCENT__;
            border: 1px solid __ACCENT__;
        }

        QLabel#PreviewPanel,
        QLabel#PreviewPlaceholder,
        QFrame#PreviewFrame {
            background-color: __PREVIEW_BG__;
            border: 1px solid __BORDER__;
            border-radius: 4px;
            color: __MUTED__;
        }
        """

    replacements = {
        "__BG__": bg,
        "__PANEL__": panel,
        "__PANEL_2__": panel_2,
        "__PANEL_3__": panel_3,
        "__BORDER__": border,
        "__BORDER_SOFT__": border_soft,
        "__TEXT__": text,
        "__TEXT_BRIGHT__": text_bright,
        "__MUTED__": muted,
        "__ACCENT__": accent,
        "__ACCENT_TEXT__": accent_text,
        "__DANGER__": danger,
        "__DANGER_BG__": danger_bg,
        "__PREVIEW_BG__": preview_bg,
    }

    for key, value in replacements.items():
        stylesheet = stylesheet.replace(key, value)

    widget.setStyleSheet(stylesheet)