import sys
import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QSplashScreen

from services.language_service import LanguageService
from services.live_service import LiveService



def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VisionDepth3D")
    app.setOrganizationName("VisionDepth")
    app.setStyle("Fusion")

    # ── Show splash BEFORE any heavy imports ──
    base_dir = os.path.dirname(__file__)
    splash_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(base_dir, "resources", "icons", f"splash{ext}")
        if os.path.exists(candidate):
            splash_path = candidate
            break

    if splash_path and os.path.exists(splash_path):
        splash_pixmap = QPixmap(splash_path).scaled(800, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    else:
        splash_pixmap = QPixmap(800, 500)
        splash_pixmap.fill(QColor("#0f141a"))

    splash = QSplashScreen(splash_pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

    # Bigger, bolder font for splash messages
    splash_font = QFont("Segoe UI", 13)
    splash_font.setBold(True)
    splash.setFont(splash_font)

    splash.show()

    def splash_msg(msg):
        splash.showMessage(
            msg,
            Qt.AlignHCenter | Qt.AlignBottom,
            QColor("#e8ecf1"),   # matches the app's main text color
        )
        app.processEvents()

    # ── Loading sequence ──
    splash_msg("Starting VisionDepth3D...")

    splash_msg("Loading 3D pipeline...")
    from models.app_state import AppState
    from models.depth_state import DepthState
    from controllers.app_controller import AppController
    from services.settings_service import SettingsService
    from services.render_service import RenderService
    from services.preview_service import PreviewService
    from services.preset_service import PresetService
    from services.depth_service import DepthService
    from ui.main_window import MainWindow

    splash_msg("Initializing services...")
    state = AppState()
    depth_state = DepthState()
    settings_service = SettingsService()
    render_service = RenderService()
    depth_service = DepthService()
    preview_service = PreviewService()
    preset_service = PresetService()
    live_service = LiveService()

    splash_msg("Loading settings...")
    
    languages_dir = os.path.join(base_dir, "resources", "languages")
    language_service = LanguageService(languages_dir)
        
    controller = AppController(
        state=state,
        depth_state=depth_state,
        settings_service=settings_service,
        render_service=render_service,
        depth_service=depth_service,
        preview_service=preview_service,
        preset_service=preset_service,
        language_service=language_service,
        live_service=live_service,
    )
    controller.load_settings()

    splash_msg("Building interface...")
    window = MainWindow(controller)

    splash_msg("Starting...")
    splash.finish(window)
    window.show()
    app.processEvents()
    window.showMaximized()

    exit_code = app.exec()
    controller.save_settings()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
