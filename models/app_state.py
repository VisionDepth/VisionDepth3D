from dataclasses import dataclass, asdict


@dataclass
class AppState:
    input_video_path: str = ""
    depth_map_path: str = ""
    output_path: str = ""

    output_format: str = "Full-SBS"
    stereo_mode: str = "sbs"
    render_mode: str = "video"
    selected_aspect_ratio: str = "Default (16:9 / 1.78:1)"

    selected_codec: str = "mp4v"

    # RTX/NVIDIA speed-friendly default.
    # If you want safer cross-vendor default, use:
    # "H.264 / AVC (libx264 - CPU)"
    selected_ffmpeg_codec: str = "H.264 / AVC (NVENC - NVIDIA GPU)"

    # Prefer FFmpeg writer. It is more reliable and allows GPU encoders.
    use_ffmpeg: bool = True

    keep_original_audio: bool = True
    preserve_hdr10: bool = False
    crf_value: int = 20
    nvenc_cq_value: int = 20
    
    encoding_container: str = ""
    encoding_extension: str = ""
    encoding_pixel_format: str = "yuv420p"
    encoding_encoder_preset: str = "p5"
    encoding_audio_mode: str = "copy"
    encoding_warning: str = ""

    fg_shift: float = -8.0
    mg_shift: float = -1.5
    bg_shift: float = 2.70

    # CPU sharpening in video path costs time. Default off.
    sharpness_factor: float = 0.0

    # IMPORTANT:
    # This value is a fraction, not 45%.
    # 0.045 = 4.5% max pixel shift.
    max_pixel_shift: float = 0.045

    zero_parallax_strength: float = -0.015
    parallax_balance: float = 0.76

    # DOF is expensive. Default off unless user enables it.
    dof_strength: float = 0.0

    convergence_strength: float = 0.0
    enable_dynamic_convergence: bool = True

    # Faster default. User can switch to Balanced/High/Showcase manually.
    edge_repair_quality: str = "Fast"

    depth_pop_gamma: float = 0.85
    depth_pop_mid: float = 0.50
    depth_stretch_lo: float = 0.05
    depth_stretch_hi: float = 0.95
    fg_pop_multiplier: float = 1.08
    bg_push_multiplier: float = 1.04
    subject_lock_strength: float = 0.34
    subject_plane_lock_strength: float = 0.0
    subject_plane_lock_width: float = 0.08

    # Subject Zero Lock strength.
    # 0.0 = off
    # 1.0 = strongly cancel tracked subject disparity toward screen plane
    # Kept as subject_screen_plane internally for backward compatibility.
    subject_screen_plane: float = 0.0

    foreground_curvature_strength: float = 0.06

    feather_strength: float = 0.0
    blur_ksize: int = 1

    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0

    use_subject_tracking: bool = False
    use_floating_window: bool = False
    enable_edge_masking: bool = True
    enable_feathering: bool = True
    preserve_original_aspect: bool = False
    auto_crop_black_bars: bool = False
    skip_blank_frames: bool = False

    preview_mode: str = "Red-Blue Anaglyph"
    preview_frame_index: int = 0
    preview_width: int = 960
    preview_height: int = 540
    keyframes_enabled: bool = False
    keyframes_path: str = ""
    ipd_enabled: bool = True
    ipd_scale: float = 1.0
    show_convergence_guides: bool = False
    disable_shift_ema: bool = False

    clip_start: str = ""
    clip_end: str = ""

    vr180_equi_w: int = 3840
    vr180_equi_h: int = 1920
    vr180_flat_w: int = 1920
    vr180_flat_h: int = 1080
    vr180_hfov_deg: float = 110.0

    language: str = "en"
    selected_theme: str = "dark"

    def to_dict(self) -> dict:
        return asdict(self)
