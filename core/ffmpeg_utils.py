import os
import sys
import shutil


def app_base_dir() -> str:
    """
    Returns the install folder beside VisionDepth3D.exe when frozen,
    or the project root in dev mode.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def bundle_base_dir() -> str:
    """
    Returns PyInstaller _MEIPASS when frozen, otherwise project root.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS

    return app_base_dir()


def find_tool(tool_name: str) -> str | None:
    exe_name = tool_name

    if os.name == "nt" and not exe_name.lower().endswith(".exe"):
        exe_name += ".exe"

    app_base = app_base_dir()
    bundle_base = bundle_base_dir()

    candidates = [
        os.path.join(app_base, exe_name),
        os.path.join(app_base, "ffmpeg", exe_name),
        os.path.join(app_base, "bin", exe_name),
        os.path.join(app_base, "_internal", exe_name),
        os.path.join(app_base, "_internal", "ffmpeg", exe_name),
        os.path.join(app_base, "_internal", "bin", exe_name),

        os.path.join(bundle_base, exe_name),
        os.path.join(bundle_base, "ffmpeg", exe_name),
        os.path.join(bundle_base, "bin", exe_name),
        os.path.join(bundle_base, "resources", "ffmpeg", exe_name),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    path_hit = shutil.which(tool_name)
    if path_hit:
        return path_hit

    if exe_name != tool_name:
        path_hit = shutil.which(exe_name)
        if path_hit:
            return path_hit

    return None


def require_tool(tool_name: str) -> str:
    path = find_tool(tool_name)

    if path:
        return path

    exe_name = tool_name
    if os.name == "nt" and not exe_name.lower().endswith(".exe"):
        exe_name += ".exe"

    app_base = app_base_dir()
    bundle_base = bundle_base_dir()

    checked = [
        os.path.join(app_base, exe_name),
        os.path.join(app_base, "ffmpeg", exe_name),
        os.path.join(app_base, "bin", exe_name),
        os.path.join(app_base, "_internal", exe_name),
        os.path.join(app_base, "_internal", "ffmpeg", exe_name),
        os.path.join(app_base, "_internal", "bin", exe_name),
        os.path.join(bundle_base, exe_name),
        os.path.join(bundle_base, "ffmpeg", exe_name),
        os.path.join(bundle_base, "bin", exe_name),
        os.path.join(bundle_base, "resources", "ffmpeg", exe_name),
        "System PATH",
    ]

    raise FileNotFoundError(
        f"{exe_name} was not found.\n\n"
        f"VisionDepth3D needs {exe_name} for this operation.\n\n"
        f"Checked:\n- " + "\n- ".join(checked) + "\n\n"
        f"Please reinstall VisionDepth3D with FFmpeg included, or install FFmpeg and add it to PATH."
    )