# DB.py — Depth Blender with path pickers + frames OR videos + Live Preview & Frame Scrubber
import os, gc, cv2, math, time, numpy as np, threading, queue, tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# --- Universal PyTorch device selector ---
try:
    import torch
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA or AMD ROCm if compiled with CUDA runtime
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    else:
        device = torch.device("cpu")

    print(f"Depth Blender Compute device: {device.type.upper()}")

except Exception as e:
    print(f"Depth Blender: PyTorch not available: {e}")
    torch = None
    device = None

# ------------ Core blending ------------
def detect_white_threshold(image, percentile=95):
    return np.percentile(image, percentile)

def create_soft_white_mask(image, threshold, softness=0.1):
    normalized = (image - threshold) / (255 * softness)
    mask = 1 / (1 + np.exp(-normalized))
    return (mask * 255).astype(np.uint8)

def boost_whites(image, threshold, boost_percent=30):
    wmask = create_soft_white_mask(image, threshold)
    boosted = image * (1 + (boost_percent / 100.0) * (wmask / 255.0))
    return np.clip(boosted, 0, 255).astype(np.uint8)

def blend_whites_seamlessly(v1_map, v2_map, blur_kernel_size=35, white_strength=1.0):
    # ensure odd kernel for GaussianBlur
    k = int(blur_kernel_size) | 1
    thr = detect_white_threshold(v2_map)
    v1_w = create_soft_white_mask(v1_map, thr)
    v2_w = create_soft_white_mask(v2_map, thr)
    v1_unique_mask = cv2.subtract(v1_w, v2_w)
    v1_unique_white = cv2.bitwise_and(v1_map, v1_map, mask=v1_unique_mask)
    v1_unique_white = np.clip(v1_unique_white, 0, thr)
    trans = cv2.GaussianBlur(v1_unique_mask.astype(np.float32), (k, k), 0) / 255.0
    blended = (v2_map * (1 - trans) + v1_unique_white * trans * white_strength).astype(np.uint8)
    blended = cv2.medianBlur(blended, 5)
    return blended

def normalize_to_v2(blended_map, v2_map):
    v2_mean, v2_std = float(np.mean(v2_map)), float(np.std(v2_map))
    b_mean, b_std   = float(np.mean(blended_map)), float(np.std(blended_map)) or 1.0
    out = (blended_map - b_mean) * (v2_std / b_std) + v2_mean
    return np.clip(out, 0, 255).astype(np.uint8)

def lighten_beta(v1_map, v2_map,
                 clip_limit=2.0, tile_grid=(8,8),
                 d=12, sC=75, sS=75, blur_k=35, white_strength=1.0,
                 use_gpu=False):
    if v1_map.ndim != 2 or v2_map.ndim != 2:
        raise ValueError("Inputs must be grayscale.")
    if isinstance(tile_grid, int):
        tile_grid = (tile_grid, tile_grid)

    h, w = v2_map.shape
    tg = (min(tile_grid[0], w), min(tile_grid[1], h))

    # --- GPU path (torch) for mask/feathering + norm ---
    if use_gpu and (device is not None and device.type != "cpu"):
        sigma = max(1.0, (int(blur_k) - 1) / 6.0)  # approx from kernel size
        blended = _blend_whites_torch(v1_map, v2_map, blur_sigma=sigma,
                                      white_strength=float(white_strength), device=device)
        # CLAHE & bilateral on CPU (OpenCV)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tg)
        blended = clahe.apply(blended)
        blended = cv2.bilateralFilter(blended, int(d), float(sC), float(sS))
        blended = _normalize_to_v2_torch(blended, v2_map, device=device)
        thr = detect_white_threshold(v2_map)
        blended = boost_whites(blended, thr, boost_percent=30)
        return blended

    # --- CPU fallback ---
    blended = blend_whites_seamlessly(v1_map, v2_map, blur_kernel_size=int(blur_k),
                                      white_strength=float(white_strength))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tg)
    blended = clahe.apply(blended)
    blended = cv2.bilateralFilter(blended, int(d), float(sC), float(sS))
    blended = normalize_to_v2(blended, v2_map)
    thr = detect_white_threshold(v2_map)
    blended = boost_whites(blended, thr, boost_percent=30)
    return blended

def _draw_preview_placeholder(self):
    self.preview_canvas.delete("all")
    self.preview_canvas.create_text(
        self.preview_canvas.winfo_width() // 2,
        self.preview_canvas.winfo_height() // 2,
        text="Preview will appear here",
        fill="#666",
        font=("Segoe UI", 14, "italic")
    )

def _redraw_preview(self, imgtk=None):
    if imgtk:
        self._preview_imgtk = imgtk  # keep reference
        self.preview_canvas.delete("all")
        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        w = imgtk.width()
        h = imgtk.height()
        x = (cw - w) // 2
        y = (ch - h) // 2
        self._preview_canvas_img = self.preview_canvas.create_image(x, y, anchor="nw", image=imgtk)

# ------------ Torch helpers ------------
def _to_torch_u8_gray(np_u8):
    t = torch.from_numpy(np_u8).to(torch.float32) / 255.0
    return t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

def _from_torch_u8_gray(t):
    t = t.clamp(0, 1).squeeze().detach().cpu().numpy()
    return (t * 255.0 + 0.5).astype(np.uint8)

def _gauss_kernel_1d(sig, radius=None, device="cpu", dtype=torch.float32):
    if radius is None:
        radius = int(max(3, round(3.0 * sig)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sig**2))
    k = (k / k.sum()).view(1, 1, -1)
    return k, radius

def _gaussian_blur_torch(img_01, sigma=5.0, device="cpu"):
    k, r = _gauss_kernel_1d(sigma, device=device, dtype=img_01.dtype)
    pad = (r, r)
    w_h = k.view(1, 1, 1, -1)
    tmp = torch.nn.functional.pad(img_01, (pad[0], pad[1], 0, 0), mode="reflect")
    tmp = torch.nn.functional.conv2d(tmp, w_h)
    w_v = k.view(1, 1, -1, 1)
    tmp = torch.nn.functional.pad(tmp, (0, 0, pad[0], pad[1]), mode="reflect")
    out = torch.nn.functional.conv2d(tmp, w_v)
    return out

def _sigmoid_soft_mask_torch(img_u8, threshold_u8, softness=0.1, device="cpu"):
    t = _to_torch_u8_gray(img_u8).to(device)
    thr = float(threshold_u8) / 255.0
    s = max(float(softness), 1e-4)
    m = torch.sigmoid((t - thr) / s)
    return m

def _blend_whites_torch(v1_u8, v2_u8, blur_sigma=7.0, white_strength=1.0, device="cpu"):
    thr = np.percentile(v2_u8, 95)
    m1 = _sigmoid_soft_mask_torch(v1_u8, thr, 0.10, device=device)
    m2 = _sigmoid_soft_mask_torch(v2_u8, thr, 0.10, device=device)
    m_unique = (m1 - m2).clamp(0, 1)

    v1 = _to_torch_u8_gray(v1_u8).to(device)
    v2 = _to_torch_u8_gray(v2_u8).to(device)

    cap = float(thr) / 255.0
    v1_cap = torch.minimum(v1, torch.tensor(cap, device=device, dtype=v1.dtype))

    trans = _gaussian_blur_torch(m_unique, sigma=float(blur_sigma), device=device).clamp(0, 1)
    out = v2 * (1.0 - trans) + v1_cap * (trans * float(white_strength))
    return _from_torch_u8_gray(out)

def _normalize_to_v2_torch(blended_u8, v2_u8, device="cpu"):
    b = _to_torch_u8_gray(blended_u8).to(device)
    v = _to_torch_u8_gray(v2_u8).to(device)
    bm, bs = b.mean(), b.std().clamp_min(1e-6)
    vm, vs = v.mean(), v.std().clamp_min(1e-6)
    out = (b - bm) * (vs / bs) + vm
    return _from_torch_u8_gray(out)


# ------------ Workers ------------
class FramesWorker(threading.Thread):
    def __init__(self, v1_dir, v2_dir, out_mode, out_path, out_w, out_h,
                 qlog, qprog, stop_evt, use_gpu=False, params=None):
        super().__init__(daemon=True)
        self.v1_dir, self.v2_dir = v1_dir, v2_dir
        self.out_mode, self.out_path = out_mode, out_path
        self.out_w, self.out_h = out_w, out_h
        self.qlog, self.qprog, self.stop_evt = qlog, qprog, stop_evt
        self.use_gpu = bool(use_gpu)
        self.params = params or {}

    def log(self, msg): self.qlog.put(msg)
    def prog(self, done, total): self.qprog.put((done, total))

    def run(self):
        try:
            v1_files = sorted([f for f in os.listdir(self.v1_dir) if f.lower().endswith(".png")])
            v2_files = sorted([f for f in os.listdir(self.v2_dir) if f.lower().endswith(".png")])
            if not v1_files or not v2_files:
                self.log("No PNG frames found in one of the folders.")
                return
            n = min(len(v1_files), len(v2_files))
            if n == 0:
                self.log("No matching frames to process.")
                return

            if self.out_mode == "output_folder":
                os.makedirs(self.out_path, exist_ok=True)

            self.prog(0, n)
            done = 0
            for i in range(n):
                if self.stop_evt.is_set():
                    self.log("Stopped by user.")
                    break
                v1p = os.path.join(self.v1_dir, v1_files[i])
                v2p = os.path.join(self.v2_dir, v2_files[i])
                v1 = cv2.imread(v1p, cv2.IMREAD_GRAYSCALE)
                v2 = cv2.imread(v2p, cv2.IMREAD_GRAYSCALE)
                if v1 is None or v2 is None:
                    self.log(f"Skip unreadable frame: {v1p} / {v2p}")
                    continue
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)

                blended = lighten_beta(
                    v1, v2,
                    clip_limit=self.params.get("clip_limit", 2.0),
                    tile_grid=(self.params.get("tile_grid", 8), self.params.get("tile_grid", 8)),
                    d=self.params.get("bf_d", 12),
                    sC=self.params.get("bf_sigmaColor", 75),
                    sS=self.params.get("bf_sigmaSpace", 75),
                    blur_k=self.params.get("blur_k", 35),
                    white_strength=self.params.get("white_strength", 1.0),
                    use_gpu=self.use_gpu
                )
                if self.out_w and self.out_h:
                    blended = cv2.resize(blended, (self.out_w, self.out_h), interpolation=cv2.INTER_LANCZOS4)

                if self.out_mode == "overwrite_v2":
                    cv2.imwrite(v2p, blended)
                else:
                    cv2.imwrite(os.path.join(self.out_path, v2_files[i]), blended)

                done += 1
                self.prog(done, n)
                del v1, v2, blended
                if (i+1) % 500 == 0:
                    gc.collect()
            self.log("Done.")
        except Exception as e:
            self.log(f"Error: {e}")

class VideosWorker(threading.Thread):
    def __init__(self, v1_file, v2_file, out_file, out_w, out_h,
                 qlog, qprog, stop_evt, use_gpu=False, params=None):
        super().__init__(daemon=True)
        self.v1_file, self.v2_file, self.out_file = v1_file, v2_file, out_file
        self.out_w, self.out_h = out_w, out_h
        self.qlog, self.qprog, self.stop_evt = qlog, qprog, stop_evt
        self.use_gpu = bool(use_gpu)
        self.params = params or {}

    def log(self, msg): self.qlog.put(msg)
    def prog(self, done, total): self.qprog.put((done, total))

    def run(self):
        try:
            cap1, cap2 = cv2.VideoCapture(self.v1_file), cv2.VideoCapture(self.v2_file)
            if not cap1.isOpened() or not cap2.isOpened():
                self.log("Could not open one of the videos.")
                return

            fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            ok2, fr2 = cap2.read()
            ok1, fr1 = cap1.read()
            if not ok1 or not ok2:
                self.log("Could not read first frames.")
                cap1.release(); cap2.release(); return

            v1g = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
            v2g = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
            if v1g.shape != v2g.shape:
                v2g = cv2.resize(v2g, (v1g.shape[1], v1g.shape[0]), interpolation=cv2.INTER_AREA)

            base_w, base_h = v2g.shape[1], v2g.shape[0]
            out_w = self.out_w or base_w
            out_h = self.out_h or base_h

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)
            writer = cv2.VideoWriter(self.out_file, fourcc, fps, (out_w, out_h), isColor=False)

            done = 0
            self.prog(0, total if total > 0 else 1)

            # first pair
            blended = lighten_beta(
                v1g, v2g,
                clip_limit=self.params.get("clip_limit", 2.0),
                tile_grid=(self.params.get("tile_grid", 8), self.params.get("tile_grid", 8)),
                d=self.params.get("bf_d", 12),
                sC=self.params.get("bf_sigmaColor", 75),
                sS=self.params.get("bf_sigmaSpace", 75),
                blur_k=self.params.get("blur_k", 35),
                white_strength=self.params.get("white_strength", 1.0),
                use_gpu=self.use_gpu
            )
            if (out_w, out_h) != (blended.shape[1], blended.shape[0]):
                blended = cv2.resize(blended, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            writer.write(blended)
            done += 1
            self.prog(done, total if total > 0 else done)

            while True:
                if self.stop_evt.is_set():
                    self.log("Stopped by user.")
                    break
                ok1, fr1 = cap1.read()
                ok2, fr2 = cap2.read()
                if not ok1 or not ok2:
                    break

                v1g = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
                v2g = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
                if v1g.shape != v2g.shape:
                    v2g = cv2.resize(v2g, (v1g.shape[1], v1g.shape[0]), interpolation=cv2.INTER_AREA)

                blended = lighten_beta(
                    v1g, v2g,
                    clip_limit=self.params.get("clip_limit", 2.0),
                    tile_grid=(self.params.get("tile_grid", 8), self.params.get("tile_grid", 8)),
                    d=self.params.get("bf_d", 12),
                    sC=self.params.get("bf_sigmaColor", 75),
                    sS=self.params.get("bf_sigmaSpace", 75),
                    blur_k=self.params.get("blur_k", 35),
                    white_strength=self.params.get("white_strength", 1.0),
                    use_gpu=self.use_gpu
                )
                if (out_w, out_h) != (blended.shape[1], blended.shape[0]):
                    blended = cv2.resize(blended, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                writer.write(blended)

                done += 1
                if done % 100 == 0: gc.collect()
                self.prog(done, total if total > 0 else done)

            writer.release(); cap1.release(); cap2.release()
            self.log(f"Done. Saved: {self.out_file}")
        except Exception as e:
            self.log(f"Error: {e}")


# ------------ GUI (with Live Preview + Frame Scrubber) ------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Depth Blender (Frames or Videos)")
        self.geometry("980x740")
        self.minsize(920, 680)

        # state
        self.mode = tk.StringVar(value="frames")
        self.overwrite_v2 = tk.BooleanVar(value=True)
        self.v1_path = tk.StringVar()
        self.v2_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.w_var = tk.StringVar()
        self.h_var = tk.StringVar()
        self.use_gpu = tk.BooleanVar(value=(device is not None and device.type != "cpu"))

        # params
        self.white_strength = tk.DoubleVar(value=1.0)
        self.blur_k = tk.IntVar(value=35)              # feather kernel size
        self.clip_limit = tk.DoubleVar(value=2.0)      # CLAHE
        self.tile_grid = tk.IntVar(value=8)            # CLAHE tile size
        self.bf_d = tk.IntVar(value=12)                # bilateral d
        self.bf_sigmaColor = tk.IntVar(value=75)
        self.bf_sigmaSpace = tk.IntVar(value=75)

        # preview infra
        self._preview_lock = threading.Lock()
        self._preview_thread = None
        self._preview_after = None
        self._preview_imgtk = None  # to keep reference

        # NEW: scrubber state
        self.preview_index = tk.IntVar(value=0)   # 0-based index
        self.preview_max   = tk.IntVar(value=0)   # max available index
        self._idx_scale = None

        self._build_ui()
        # Key bindings for scrubbing
        self.bind("<Left>",  lambda e: self._nudge_preview(-1))
        self.bind("<Right>", lambda e: self._nudge_preview(+1))

        # background poller
        self.qlog = queue.Queue()
        self.qprog = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker = None
        self.after(100, self._poll)

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        title = ttk.Label(self, text="Depth Blender", font=("Segoe UI", 18, "bold"))
        title.pack(fill="x", **pad)

        top = ttk.Frame(self); top.pack(fill="x", **pad)

        # Left: controls
        left = ttk.Frame(top); left.pack(side="left", fill="y", padx=6)
        self._build_controls(left)

        # Right: preview
        right = ttk.Frame(top); right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Preview (scrubbable):", style="VD3D.TLabel").pack(anchor="w")

        # subtle border frame
        border = tk.Frame(right, bg="#2a2a2a", highlightthickness=0)
        border.pack(fill="both", expand=True, padx=6, pady=6)

        # the actual drawing area (dark bg, no white highlight)
        self.preview_canvas = tk.Canvas(
            border,
            bg="#1c1c1c",
            highlightthickness=0,  # no white border
            bd=0,
            width=640, height=360   # sensible minimum so it’s visible when empty
        )
        self.preview_canvas.pack(fill="both", expand=True, padx=1, pady=1)

        # keep last image id so we can re-center on resize
        self._preview_canvas_img = None
        self.preview_canvas.bind("<Configure>", lambda e: self._redraw_preview())
        self._draw_preview_placeholder()

        # Bottom: progress/log
        bottom = ttk.Frame(self); bottom.pack(fill="both", expand=True, **pad)
        self._build_progress_and_log(bottom)

        self._toggle_mode()
        self._toggle_out_controls()
        self._update_preview_bounds()
        self._schedule_preview(0)

    def _build_controls(self, parent):
        # Mode
        mode_frame = ttk.LabelFrame(parent, text="Mode")
        mode_frame.pack(fill="x", padx=6, pady=6)
        ttk.Radiobutton(mode_frame, text="Folders (frames)", variable=self.mode, value="frames",
                        command=self._toggle_mode).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(mode_frame, text="Videos", variable=self.mode, value="videos",
                        command=self._toggle_mode).grid(row=0, column=1, sticky="w", padx=12, pady=4)

        gpu_type = device.type if device else "cpu"
        ttk.Checkbutton(gpu_row, text=f"Use GPU ({gpu_type})",
                        variable=self.use_gpu,
                        command=lambda: self._schedule_preview(120)).pack(anchor="w")

                        
        if device is None or device.type == "cpu":
            ttk.Label(gpu_row, text="GPU not available. Using CPU.", foreground="#c77").pack(anchor="w")
        else:
            ttk.Label(gpu_row, text=f"GPU Mode: {device.type}", foreground="#7c7").pack(anchor="w")

        # Paths
        paths = ttk.LabelFrame(parent, text="Inputs")
        paths.pack(fill="x", padx=6, pady=6)

        ttk.Label(paths, text="V1 path:").grid(row=0, column=0, sticky="e")
        ttk.Entry(paths, textvariable=self.v1_path, width=40).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(paths, text="Browse…", command=self._browse_v1).grid(row=0, column=2, padx=4)

        ttk.Label(paths, text="V2 path:").grid(row=1, column=0, sticky="e")
        ttk.Entry(paths, textvariable=self.v2_path, width=40).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(paths, text="Browse…", command=self._browse_v2).grid(row=1, column=2, padx=4)

        # Output
        outf = ttk.LabelFrame(parent, text="Output")
        outf.pack(fill="x", padx=6, pady=6)
        self.chk_over = ttk.Checkbutton(outf, text="Overwrite V2 (frames mode only)",
                                        variable=self.overwrite_v2, command=self._toggle_out_controls)
        self.chk_over.grid(row=0, column=0, sticky="w", padx=6)

        ttk.Label(outf, text="Output path/file:").grid(row=1, column=0, sticky="e")
        ttk.Entry(outf, textvariable=self.out_path, width=40).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(outf, text="Browse…", command=self._browse_out).grid(row=1, column=2, padx=4)

        # Size
        sizef = ttk.LabelFrame(parent, text="Final Size (optional)")
        sizef.pack(fill="x", padx=6, pady=6)
        ttk.Label(sizef, text="Width:").grid(row=0, column=0, sticky="e")
        ttk.Entry(sizef, textvariable=self.w_var, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(sizef, text="Height:").grid(row=0, column=2, sticky="e")
        ttk.Entry(sizef, textvariable=self.h_var, width=8).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(sizef, text="(Leave blank to keep source)").grid(row=0, column=4, sticky="w", padx=12)

        # Tunable parameters (with preview)
        parms = ttk.LabelFrame(parent, text="Blend Parameters (preview live)")
        parms.pack(fill="x", padx=6, pady=6)

        self._add_slider(parms, "White Strength", 0.0, 2.0, self.white_strength, 0)
        self._add_slider(parms, "Feather Blur (kernel)", 1, 99, self.blur_k, 1)
        self._add_slider(parms, "CLAHE Clip Limit", 0.5, 4.0, self.clip_limit, 2)
        self._add_slider(parms, "CLAHE Tile Grid", 2, 32, self.tile_grid, 3)
        self._add_slider(parms, "Bilateral d", 1, 25, self.bf_d, 4)
        self._add_slider(parms, "Bilateral sigmaColor", 1, 200, self.bf_sigmaColor, 5)
        self._add_slider(parms, "Bilateral sigmaSpace", 1, 200, self.bf_sigmaSpace, 6)

        # NEW: Preview Frame scrubber controls
        scrub = ttk.LabelFrame(parent, text="Preview Frame")
        scrub.pack(fill="x", padx=6, pady=6)

        self._idx_scale = ttk.Scale(
            scrub, from_=0, to=0, orient="horizontal",
            command=lambda _=None: self._schedule_preview(50),
            variable=self.preview_index
        )
        self._idx_scale.grid(row=0, column=0, sticky="we", padx=6, pady=4)
        scrub.grid_columnconfigure(0, weight=1)

        ttk.Label(scrub, textvariable=self.preview_index, width=6).grid(row=0, column=1, sticky="e", padx=6)

        btns = ttk.Frame(scrub); btns.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        ttk.Button(btns, text="⟨ Prev", command=lambda: self._nudge_preview(-1)).pack(side="left", padx=2)
        ttk.Button(btns, text="Next ⟩", command=lambda: self._nudge_preview(+1)).pack(side="left", padx=2)

        # Buttons
        btns2 = ttk.Frame(parent); btns2.pack(fill="x", padx=6, pady=6)
        self.btn_preview = ttk.Button(btns2, text="Preview Now", command=self._preview_now)
        self.btn_start   = ttk.Button(btns2, text="Start Batch", command=self._start)
        self.btn_stop    = ttk.Button(btns2, text="Stop",  command=self._stop, state="disabled")
        self.btn_preview.grid(row=0, column=0, padx=4)
        self.btn_start.grid(row=0, column=1, padx=4)
        self.btn_stop.grid(row=0, column=2, padx=4)

    def _add_slider(self, parent, label, mn, mx, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6)
        s = ttk.Scale(parent, from_=mn, to=mx, orient="horizontal",
                      command=lambda _=None: self._schedule_preview(120), variable=var)
        s.grid(row=row, column=1, sticky="we", padx=6)
        parent.grid_columnconfigure(1, weight=1)
        ttk.Label(parent, textvariable=var).grid(row=row, column=2, sticky="e", padx=6)

    def _build_progress_and_log(self, parent):
        pf = ttk.Frame(parent); pf.pack(fill="x")
        self.prog = ttk.Progressbar(pf, mode="determinate"); self.prog.pack(fill="x")
        self.prog_lbl = ttk.Label(pf, text="Progress: 0/0"); self.prog_lbl.pack(anchor="w")

        lf = ttk.LabelFrame(parent, text="Log"); lf.pack(fill="both", expand=True, padx=6, pady=6)
        self.log = tk.Text(lf, height=10, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    # ---------- Browsers ----------
    def _browse_v1(self):
        if self.mode.get() == "frames":
            p = filedialog.askdirectory(title="Select V1 frames folder")
        else:
            p = filedialog.askopenfilename(title="Select V1 video",
                                           filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi"), ("All", "*.*")])
        if p:
            self.v1_path.set(p)
            self._update_preview_bounds()
            self._schedule_preview(0)

    def _browse_v2(self):
        if self.mode.get() == "frames":
            p = filedialog.askdirectory(title="Select V2 frames folder (base)")
        else:
            p = filedialog.askopenfilename(title="Select V2 video (base)",
                                           filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi"), ("All", "*.*")])
        if p:
            self.v2_path.set(p)
            self._update_preview_bounds()
            self._schedule_preview(0)

    def _browse_out(self):
        if self.mode.get() == "frames":
            if not self.overwrite_v2.get():
                p = filedialog.askdirectory(title="Select output frames folder")
                if p: self.out_path.set(p)
            else:
                messagebox.showinfo("Output", "Overwrite V2 is on. No separate output folder needed.")
        else:
            p = filedialog.asksaveasfilename(title="Save output video as", defaultextension=".mp4",
                                             filetypes=[("MP4", "*.mp4"), ("All", "*.*")])
            if p: self.out_path.set(p)

    # ---------- Toggles ----------
    def _toggle_mode(self):
        if self.mode.get() == "videos":
            self.chk_over.state(["disabled"])
        else:
            self.chk_over.state(["!disabled"])
        self._toggle_out_controls()
        self._update_preview_bounds()
        self._schedule_preview(0)

    def _toggle_out_controls(self):
        pass  # keep simple

    # ---------- Scrubber helpers ----------
    def _nudge_preview(self, delta):
        cur = int(self.preview_index.get())
        mx  = int(self.preview_max.get())
        new = max(0, min(mx, cur + int(delta)))
        if new != cur:
            self.preview_index.set(new)
            self._schedule_preview(50)

    def _update_preview_bounds(self):
        """Recompute preview_max and slider range when inputs or mode change."""
        mx = 0
        if self.mode.get() == "frames":
            v1p, v2p = self.v1_path.get().strip(), self.v2_path.get().strip()
            try:
                n1 = len([f for f in os.listdir(v1p) if f.lower().endswith(".png")])
                n2 = len([f for f in os.listdir(v2p) if f.lower().endswith(".png")])
                mx = max(0, min(n1, n2) - 1)
            except Exception:
                mx = 0
        else:
            v2p = self.v2_path.get().strip()
            if v2p:
                cap = cv2.VideoCapture(v2p)
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
                    mx = max(0, total - 1)
                cap.release()
        self.preview_max.set(mx)
        if self._idx_scale is not None:
            self._idx_scale.configure(to=mx)
        self.preview_index.set(min(int(self.preview_index.get()), mx))

    # ---------- Start/Stop ----------
    def _start(self):
        mode = self.mode.get()
        v1, v2 = self.v1_path.get().strip(), self.v2_path.get().strip()
        if not v1 or not v2:
            messagebox.showerror("Missing paths", "Please select both V1 and V2 paths.")
            return

        out_w = int(self.w_var.get()) if self.w_var.get().strip().isdigit() else None
        out_h = int(self.h_var.get()) if self.h_var.get().strip().isdigit() else None

        params = {
            "white_strength": float(self.white_strength.get()),
            "blur_k": int(self.blur_k.get()),
            "clip_limit": float(self.clip_limit.get()),
            "tile_grid": int(self.tile_grid.get()),
            "bf_d": int(self.bf_d.get()),
            "bf_sigmaColor": int(self.bf_sigmaColor.get()),
            "bf_sigmaSpace": int(self.bf_sigmaSpace.get()),
        }

        self.stop_evt.clear()
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        self._set_prog(0, 0); self._log("Starting...")

        if mode == "frames":
            ow = self.overwrite_v2.get()
            out_mode = "overwrite_v2" if ow else "output_folder"
            out_path = self.out_path.get().strip()
            if not ow and not out_path:
                messagebox.showerror("Missing output folder", "Pick an output folder or enable Overwrite V2.")
                self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
                return
            self.worker = FramesWorker(
                v1, v2, out_mode, out_path, out_w, out_h,
                self.qlog, self.qprog, self.stop_evt,
                use_gpu=self.use_gpu.get(), params=params
            )
        else:
            out_file = self.out_path.get().strip()
            if not out_file:
                messagebox.showerror("Missing output file", "Choose where to save the output video.")
                self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
                return
            self.worker = VideosWorker(
                v1, v2, out_file, out_w, out_h,
                self.qlog, self.qprog, self.stop_evt,
                use_gpu=self.use_gpu.get(), params=params
            )

        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_evt.set()
            self._log("Stopping requested...")

    # ---------- Preview ----------
    def _schedule_preview(self, delay_ms=200):
        # debounce: cancel previous .after if any
        if self._preview_after is not None:
            try:
                self.after_cancel(self._preview_after)
            except Exception:
                pass
        self._preview_after = self.after(int(max(0, delay_ms)), self._preview_now)

    def _preview_now(self):
        # spawn a short worker to compute one blended frame
        with self._preview_lock:
            if self._preview_thread and self._preview_thread.is_alive():
                return
            self._preview_thread = threading.Thread(target=self._compute_preview, daemon=True)
            self._preview_thread.start()

    def _compute_preview(self):
        try:
            mode = self.mode.get()
            v1p, v2p = self.v1_path.get().strip(), self.v2_path.get().strip()
            if not v1p or not v2p:
                return

            idx = int(self.preview_index.get())

            if mode == "frames":
                # load selected pair by index
                v1_files = sorted([f for f in os.listdir(v1p) if f.lower().endswith(".png")])
                v2_files = sorted([f for f in os.listdir(v2p) if f.lower().endswith(".png")])
                if not v1_files or not v2_files:
                    return
                idx = max(0, min(idx, min(len(v1_files), len(v2_files)) - 1))
                v1 = cv2.imread(os.path.join(v1p, v1_files[idx]), cv2.IMREAD_GRAYSCALE)
                v2 = cv2.imread(os.path.join(v2p, v2_files[idx]), cv2.IMREAD_GRAYSCALE)
                if v1 is None or v2 is None:
                    return
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                cap1, cap2 = cv2.VideoCapture(v1p), cv2.VideoCapture(v2p)
                if not cap1.isOpened() or not cap2.isOpened():
                    if cap1: cap1.release()
                    if cap2: cap2.release()
                    return
                cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok1, fr1 = cap1.read(); ok2, fr2 = cap2.read()
                cap1.release(); cap2.release()
                if not ok1 or not ok2:
                    return
                v1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
                v2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
                if v1.shape != v2.shape:
                    v2 = cv2.resize(v2, (v1.shape[1], v1.shape[0]), interpolation=cv2.INTER_AREA)

            params = {
                "white_strength": float(self.white_strength.get()),
                "blur_k": int(self.blur_k.get()),
                "clip_limit": float(self.clip_limit.get()),
                "tile_grid": int(self.tile_grid.get()),
                "bf_d": int(self.bf_d.get()),
                "bf_sigmaColor": int(self.bf_sigmaColor.get()),
                "bf_sigmaSpace": int(self.bf_sigmaSpace.get()),
            }
            out = lighten_beta(
                v1, v2,
                clip_limit=params["clip_limit"],
                tile_grid=(params["tile_grid"], params["tile_grid"]),
                d=params["bf_d"],
                sC=params["bf_sigmaColor"],
                sS=params["bf_sigmaSpace"],
                blur_k=params["blur_k"],
                white_strength=params["white_strength"],
                use_gpu=self.use_gpu.get()
            )

            # compose a small preview: [V2 | OUT]
            vis_v2 = cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR)
            vis_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            label = f"Blended Preview (idx {idx})"
            panel = np.hstack([_put_label(vis_v2, "V2 Base"), _put_label(vis_out, label)])

            # fit into preview area (max width ~ 840)
            panel = _resize_max(panel, max_w=840, max_h=520)
            im = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(im)
            self._preview_imgtk = imgtk  # keep ref
            self.preview_canvas.after(0, lambda: self._redraw_preview(imgtk))
        except Exception:
            # best-effort: show nothing
            pass

    # ---------- Utility ----------
    def _set_prog(self, done, total):
        self.prog["maximum"] = max(total, 1)
        self.prog["value"] = done
        self.prog_lbl.config(text=f"Progress: {done}/{total}")

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll(self):
        try:
            while True:
                self._log(self.qlog.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                d, t = self.qprog.get_nowait()
                self._set_prog(d, t)
        except queue.Empty:
            pass
        if self.worker and not self.worker.is_alive():
            self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.after(100, self._poll)


# ---- small helpers for preview visuals ----
def _put_label(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out

def _resize_max(img, max_w=840, max_h=520):
    h, w = img.shape[:2]
    sc = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if sc < 1.0:
        img = cv2.resize(img, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    return img


if __name__ == "__main__":
    App().mainloop()
