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
    lines += ["GPU Diagnostics version: v4.2"]

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

        # DirectML check for AMD / Intel / non-CUDA Windows GPU users.
        dml_ok = False
        dml_device = None
        dml_error = None

        try:
            import torch_directml

            dml_device = torch_directml.device()
            test_tensor = torch.zeros(1, device=dml_device)
            dml_ok = str(test_tensor.device).startswith("privateuseone")

            lines += [
                f"DirectML available: {'YES' if dml_ok else 'NO'}",
                f"DirectML device: {dml_device if dml_ok else '(not available)'}",
            ]

        except Exception as e:
            dml_error = str(e)
            lines += [
                "DirectML available: NO",
                f"DirectML error: {dml_error}",
            ]

        if torch.cuda.is_available():
            lines += ["Active VD3D backend: CUDA"]
        elif dml_ok:
            lines += ["Active VD3D backend: DirectML"]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            lines += ["Active VD3D backend: MPS"]
        else:
            lines += ["Active VD3D backend: CPU"]

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
        
        try:
            if dml_ok and dml_device is not None:
                import time
                t0 = time.time()
                x = torch.randn(512, 512, device=dml_device)
                y = torch.randn(512, 512, device=dml_device)
                z = (x @ y).sum().cpu().item()
                dt = time.time() - t0
                lines += [f"DirectML matmul OK in {dt:.3f}s (checksum {z:.3g})"]
        except Exception as e:
            lines += [f"DirectML op test failed: {e}"]

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
        _show_modern_gpu_diagnostics(report)
    except Exception as e:
        print(f"GPU diagnostics window failed: {e}")

def _show_modern_gpu_diagnostics(report: str):
    import tkinter as tk
    from tkinter import ttk

    def parse_report(text):
        data = {}
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        return data

    data = parse_report(report)

    def status_value(value):
        v = str(value).strip().lower()

        if v in ("yes", "true", "ok"):
            return "OK"

        if v in ("no", "false", "none", "(not found)", "missing"):
            return "BAD"

        if "failed" in v or "not found" in v or "missing" in v:
            return "BAD"

        if "maybe" in v or "unavailable" in v:
            return "WARN"

        return "INFO"

    def add_card(parent, title, rows):
        card = tk.Frame(parent, bg="#1f2937", bd=0, highlightthickness=1, highlightbackground="#374151")
        card.pack(fill="x", padx=14, pady=8)

        header = tk.Label(
            card,
            text=title,
            bg="#111827",
            fg="#f9fafb",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            padx=12,
            pady=8,
        )
        header.pack(fill="x")

        body = tk.Frame(card, bg="#1f2937")
        body.pack(fill="x", padx=10, pady=10)

        for r, key in enumerate(rows):
            value = data.get(key, "Not checked")
            state = status_value(value)

            if state == "OK":
                badge_bg = "#065f46"
                badge_fg = "#d1fae5"
            elif state == "BAD":
                badge_bg = "#7f1d1d"
                badge_fg = "#fee2e2"
            elif state == "WARN":
                badge_bg = "#78350f"
                badge_fg = "#fef3c7"
            else:
                badge_bg = "#374151"
                badge_fg = "#e5e7eb"

            name_lbl = tk.Label(
                body,
                text=key,
                bg="#1f2937",
                fg="#d1d5db",
                font=("Segoe UI", 10),
                anchor="w",
            )
            name_lbl.grid(row=r, column=0, sticky="w", padx=(0, 10), pady=4)

            badge = tk.Label(
                body,
                text=value,
                bg=badge_bg,
                fg=badge_fg,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                padx=8,
                pady=3,
            )
            badge.grid(row=r, column=1, sticky="ew", pady=4)

        body.grid_columnconfigure(1, weight=1)

    root = tk.Tk()
    root.title("VisionDepth3D GPU Diagnostics")
    root.geometry("780x720")
    root.minsize(720, 560)
    root.configure(bg="#0f172a")

    title_bar = tk.Frame(root, bg="#020617")
    title_bar.pack(fill="x")

    tk.Label(
        title_bar,
        text="VisionDepth3D GPU Diagnostics",
        bg="#020617",
        fg="#f8fafc",
        font=("Segoe UI", 18, "bold"),
        anchor="w",
        padx=18,
        pady=14,
    ).pack(fill="x")

    active_backend = data.get("Active VD3D backend", "Unknown")

    tk.Label(
        title_bar,
        text=f"Active Backend: {active_backend}",
        bg="#020617",
        fg="#93c5fd",
        font=("Segoe UI", 11, "bold"),
        anchor="w",
        padx=18,
        pady=(0, 14),
    ).pack(fill="x")

    container = tk.Frame(root, bg="#0f172a")
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, bg="#0f172a", highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

    scroll_frame = tk.Frame(canvas, bg="#0f172a")

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def resize_canvas(event):
        canvas.itemconfig(canvas_window, width=event.width)

    canvas.bind("<Configure>", resize_canvas)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    add_card(scroll_frame, "Backend Summary", [
        "GPU Diagnostics version",
        "Active VD3D backend",
        "PyTorch",
    ])

    add_card(scroll_frame, "CUDA / NVIDIA", [
        "CUDA available",
        "torch.version.cuda",
        "Device count",
        "Device 0",
        "cuDNN enabled",
        "CUDA matmul OK in",
        "NVIDIA driver",
        "NVENC encoders listed",
    ])

    add_card(scroll_frame, "DirectML", [
        "DirectML available",
        "DirectML device",
        "DirectML matmul OK in",
        "DirectML error",
    ])

    add_card(scroll_frame, "FFmpeg Tools", [
        "FFmpeg found",
        "FFmpeg source",
        "FFmpeg executable test",
        "FFprobe found",
        "FFprobe source",
        "FFprobe executable test",
    ])

    add_card(scroll_frame, "Install Paths", [
        "App base",
        "Bundle base",
        "FFmpeg path",
        "FFprobe path",
    ])

    raw_card = tk.Frame(scroll_frame, bg="#1f2937", highlightthickness=1, highlightbackground="#374151")
    raw_card.pack(fill="both", expand=True, padx=14, pady=8)

    tk.Label(
        raw_card,
        text="Raw Diagnostic Log",
        bg="#111827",
        fg="#f9fafb",
        font=("Segoe UI", 12, "bold"),
        anchor="w",
        padx=12,
        pady=8,
    ).pack(fill="x")

    text_box = tk.Text(
        raw_card,
        bg="#020617",
        fg="#d1d5db",
        insertbackground="#ffffff",
        font=("Consolas", 9),
        wrap="word",
        height=12,
        relief="flat",
    )
    text_box.pack(fill="both", expand=True, padx=10, pady=10)
    text_box.insert("1.0", report)
    text_box.configure(state="disabled")

    button_bar = tk.Frame(root, bg="#020617")
    button_bar.pack(fill="x")

    def copy_report():
        root.clipboard_clear()
        root.clipboard_append(report)

    tk.Button(
        button_bar,
        text="Copy Report",
        command=copy_report,
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=14,
        pady=8,
    ).pack(side="left", padx=18, pady=12)

    tk.Button(
        button_bar,
        text="Close",
        command=root.destroy,
        bg="#374151",
        fg="#ffffff",
        activebackground="#4b5563",
        activeforeground="#ffffff",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=14,
        pady=8,
    ).pack(side="right", padx=18, pady=12)

    root.mainloop()
