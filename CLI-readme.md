## VisionDepth3D CLI (v3.1)

Command-line interface for advanced 2D-to-3D rendering with full depth control.

---

### Basic Usage

```bash
python render_cli.py --input video.mp4 --depth depthmap.mp4 --output output_3d.mp4
```

---

### Key Arguments

| Argument | Description |
|---------|-------------|
| `--input` | Input video file |
| `--depth` | Depth map video file |
| `--output` | Output 3D video file (auto-names if omitted) |
| `--format` | 3D output format: `Half-SBS`, `Full-SBS`, `VR`, `Passive Interlaced`, `Red-Cyan Anaglyph` |
| `--aspect` | Aspect ratio (e.g., "Default (16:9)") |
| `--codec` | OpenCV codec (e.g., `mp4v`, `XVID`) |
| `--ffmpeg` | Use FFmpeg instead of OpenCV (add `--ffmpeg_codec`, `--crf`) |

---

### Depth & Stereo Controls

| Flag | Default | Description |
|------|---------|-------------|
| `--fg_shift` | `10.0` | Foreground pop strength |
| `--mg_shift` | `-2.5` | Midground shaping |
| `--bg_shift` | `-5.0` | Push background depth |
| `--convergence_offset` | `0.0` | Zero-parallax depth shift (e.g., `-0.01` to `0.01`) |
| `--max_pixel_shift` | `0.02` | Caps extreme stereo warping |
| `--parallax_balance` | `0.8` | Scales total 3D effect (0 = flat, 1 = max) |

---

### Rendering Quality

| Flag | Default | Description |
|------|---------|-------------|
| `--feather` | `10.0` | Edge feather strength |
| `--blur` | `9` | Blur size for feathering |
| `--sharpness` | `0.15` | Output sharpening |

---

### Output Tweaks

| Flag | Description |
|------|-------------|
| `--width` / `--height` | Override output size |
| `--fps` | Override FPS |
| `--preserve_content` | Keep original aspect ratio |
| `--no_track` | Disable subject tracking |
| `--no_floating` | Disable floating window (black bar masking) |

---

### Dev Tools

| Flag | Description |
|------|-------------|
| `--dry_run` | Show config and exit without rendering |
| `--verbose` | Print full pipeline details |

---

### Example

```bash
python render_cli.py --input mymovie.mp4 --depth mymovie_depth.mp4 --format Full-SBS \
--fg_shift 8.0 --bg_shift -6.0 --convergence_offset -0.01 \
--parallax_balance 0.75 --max_pixel_shift 0.015 \
--ffmpeg --ffmpeg_codec h264_nvenc --crf 21 --output stereo_output.mp4
```

> Tip: Adjust `--convergence_offset` and `--parallax_balance` together to shift the 3D "focus" toward or away from the screen.
