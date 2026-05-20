from dataclasses import dataclass, asdict


@dataclass
class DepthState:
    input_video_path: str = ""
    output_dir: str = ""
    selected_model: str = "  -- Select Model -- "
    inference_resolution: str = "Original"
    batch_size: int = 8
    inference_steps: int = 5
    codec: str = "H.265 / HEVC (NVENC - NVIDIA GPU)"
    colormap: str = "Default"
    invert_depth: bool = False
    save_frames: bool = False
    use_fp16: bool = False
    disable_depth_normalizer: bool = False
    offload_mode: str = "none"
    target_fps: int = 8
    ignore_letterbox_bars: bool = True
    prefer_opencv_writer: bool = False
    is_processing: bool = False
    is_paused: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
