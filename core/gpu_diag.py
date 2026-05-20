def gpu_diagnostics(return_text: bool = False):
    import platform
    import subprocess
    import shutil

    def hidden_subprocess_kwargs():
        if platform.system().lower() != "windows":
            return {}

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0

        return {
            "startupinfo": startupinfo,
            "creationflags": subprocess.CREATE_NO_WINDOW,
        }

    lines = []
    lines += ["GPU Diagnostics version: v4.1.1"]

    try:
        import torch
        lines += [
            f"PyTorch: {torch.__version__}",
            f"CUDA available: {torch.cuda.is_available()}",
            f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}",
        ]

        if torch.cuda.is_available():
            lines += [
                f"Device count: {torch.cuda.device_count()}",
                f"Device 0: {torch.cuda.get_device_name(0)}",
                f"cuDNN enabled: {torch.backends.cudnn.enabled}",
            ]

        try:
            if torch.cuda.is_available():
                import time
                t0 = time.time()
                x = torch.randn(1024, 1024, device="cuda")
                y = torch.randn(1024, 1024, device="cuda")
                z = (x @ y).sum().item()
                dt = time.time() - t0
                lines += [f"CUDA matmul OK in {dt:.3f}s (checksum {z:.3g})"]
        except Exception as e:
            lines += [f"CUDA op test failed: {e}"]

    except Exception as e:
        lines += [f"PyTorch import failed: {e}"]

    # FFmpeg / FFprobe / NVENC presence
    try:
        import os
        import sys

        def app_base_dir():
            """
            Installed app folder beside VisionDepth3D.exe when frozen,
            or project root in dev mode.
            """
            if getattr(sys, "frozen", False):
                return os.path.dirname(sys.executable)

            return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def bundle_base_dir():
            """
            PyInstaller _MEIPASS folder when frozen, otherwise app base.
            """
            if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                return sys._MEIPASS

            return app_base_dir()

        def find_tool_local(tool_name):
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
                    return path, "bundled/app"

            path_hit = shutil.which(tool_name)
            if path_hit:
                return path_hit, "system PATH"

            if exe_name != tool_name:
                path_hit = shutil.which(exe_name)
                if path_hit:
                    return path_hit, "system PATH"

            return None, "missing"

        ffmpeg_path, ffmpeg_source = find_tool_local("ffmpeg")
        ffprobe_path, ffprobe_source = find_tool_local("ffprobe")

        lines += [
            f"App base: {app_base_dir()}",
            f"Bundle base: {bundle_base_dir()}",
            f"FFmpeg found: {'YES' if ffmpeg_path else 'NO'}",
            f"FFmpeg source: {ffmpeg_source}",
            f"FFmpeg path: {ffmpeg_path if ffmpeg_path else '(not found)'}",
            f"FFprobe found: {'YES' if ffprobe_path else 'NO'}",
            f"FFprobe source: {ffprobe_source}",
            f"FFprobe path: {ffprobe_path if ffprobe_path else '(not found)'}",
        ]

        if ffmpeg_path:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                **hidden_subprocess_kwargs(),
            )

            out = result.stdout or ""
            has_nvenc = any("nvenc" in ln.lower() for ln in out.splitlines())

            lines += [
                "FFmpeg executable test: YES" if result.returncode == 0 else "FFmpeg executable test: MAYBE",
                f"NVENC encoders listed: {'YES' if has_nvenc else 'NO'}",
            ]
        else:
            lines += [
                "FFmpeg executable test: NO",
                "NVENC encoders listed: NO",
            ]

        if ffprobe_path:
            result = subprocess.run(
                [ffprobe_path, "-hide_banner", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                **hidden_subprocess_kwargs(),
            )

            lines += [
                "FFprobe executable test: YES" if result.returncode == 0 else "FFprobe executable test: MAYBE",
            ]
        else:
            lines += [
                "FFprobe executable test: NO",
            ]

    except Exception as e:
        lines += [f"FFmpeg/FFprobe check failed: {e}"]
        
    # NVIDIA driver version
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                **hidden_subprocess_kwargs(),
            )

            if result.returncode == 0 and result.stdout.strip():
                lines += [f"NVIDIA driver: {result.stdout.strip()}"]
            else:
                lines += ["NVIDIA driver: (nvidia-smi not found)"]

        except Exception:
            lines += ["NVIDIA driver: (nvidia-smi not found)"]

    report = "\n".join(lines)
    print("\n=== GPU DIAGNOSTICS ===\n" + report + "\n=======================\n")

    if return_text:
        return report

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("GPU Diagnostics", report)
        root.destroy()
    except Exception:
        pass
