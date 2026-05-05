from dataclasses import dataclass, asdict


@dataclass
class AppState:
    input_video_path: str = ""
    depth_map_path: str = ""
    output_path: str = ""

    output_format: str = "Full-SBS"
    stereo_mode: str = "sbs"
    selected_aspect_ratio: str = "Default (16:9)"

    selected_codec: str = "XVID"
    selected_ffmpeg_codec: str = "H.264 / AVC (libx264 - CPU)"
    use_ffmpeg: bool = False
    keep_original_audio: bool = True
    preserve_hdr10: bool = False
    crf_value: int = 23
    nvenc_cq_value: int = 23

    fg_shift: float = -8.10
    mg_shift: float = -1.30
    bg_shift: float = 3.10
    sharpness_factor: float = 0.2

    max_pixel_shift: float = 0.100
    zero_parallax_strength: float = 0.7
    parallax_balance: float = 0.80
    dof_strength: float = 0.6
    convergence_strength: float = 0.0
    enable_dynamic_convergence: bool = True

    depth_pop_gamma: float = 0.80
    depth_pop_mid: float = 0.50
    depth_stretch_lo: float = 0.05
    depth_stretch_hi: float = 0.95
    fg_pop_multiplier: float = 1.08
    bg_push_multiplier: float = 1.04
    subject_lock_strength: float = 1.00

    feather_strength: float = 0.0
    blur_ksize: int = 1

    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0

    use_subject_tracking: bool = True
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
    ipd_enabled: bool = False
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

    def to_dict(self) -> dict:
        return asdict(self)
