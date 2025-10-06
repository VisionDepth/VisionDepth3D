# core/adapters/lbm_adapter.py
import os, numpy as np, torch
from pathlib import Path
from PIL import Image

def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)  # Ampere/Ada+ => 8.x
    return major >= 8

def load_lbm_adapter(spec: str, cache_dir: str):
    """
    spec: 'jasperai/LBM_depth' or a local path to a folder with config.yaml + model.safetensors
    cache_dir: where to store HF cache (we bind it to your weights folder)
    """
    cache_dir = str(Path(cache_dir).expanduser())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Make HF download into your weights dir (Windows-safe, no symlinks needed)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)

    # Import LBM (must be installed from GitHub)
    try:
        from lbm.inference import evaluate, get_model
    except Exception as e:
        raise RuntimeError(
            "LBM library not installed. Run:\n"
            "  pip install git+https://github.com/gojasper/LBM.git"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (
        torch.bfloat16 if _bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    # Let LBM download/use the HF weights (they’ll land under cache_dir)
    model = get_model(spec, torch_dtype=torch_dtype, device=device)

    def lbm_fn(images, inference_size=None, steps=1):
        if not isinstance(images, list):
            images = [images]
        outs = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            if inference_size:
                img = img.resize(inference_size, Image.BICUBIC)
            out_img = evaluate(model, img, num_sampling_steps=steps)  # returns PIL
            arr = np.asarray(out_img.convert("L"), dtype=np.float32) / 255.0
            outs.append({"predicted_depth": torch.from_numpy(arr)})
        return outs

    return lbm_fn, {"is_diffusion": True, "is_lbm": True}
