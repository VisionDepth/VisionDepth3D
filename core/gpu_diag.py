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

    # FFmpeg / NVENC presence
    try:
        ff = shutil.which("ffmpeg") or "ffmpeg"

        result = subprocess.run(
            [ff, "-hide_banner", "-encoders"],
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
            "FFmpeg found: YES" if result.returncode == 0 else "FFmpeg found: MAYBE",
            f"NVENC encoders listed: {'YES' if has_nvenc else 'NO'}",
        ]

    except Exception as e:
        lines += [f"FFmpeg check failed: {e}"]

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
