
# VisionDepth3D – Weights Folder Setup

This folder stores all model weights required by VisionDepth3D.

## Expected Folder Structure

Place your models inside the `weights/` folder using the following structure:

---

### 1. Upscale Weights  
Download: [Google Drive Link](https://drive.google.com/file/d/1eEMcKItBn8MqH6fTCJX890A9HD054Ei4/view?usp=sharing)

```
weights/
└── [Upscale Files Here]
```

> No subfolder needed — drop files directly into `weights/`.

---

### 2. Distill Any Depth (ONNX Models)  
Download: [Distill Any Depth Models (Hugging Face)](https://huggingface.co/collections/FuryTMP/distill-any-depth-onnx-models-681cad0ff43990f5dc2ff670)

```
weights/
└── Distill Any Depth X/
    └── model.onnx
```

> `X` can be `Large`, `Small`, etc.  
> Auto-detected by dropdown in GUI.

---

### 3. DepthCrafter (Diffusion Model)  
Download: [DepthCrafter Model Zoo](https://huggingface.co/tencent/DepthCrafter/tree/main)

```
weights/
└── DepthCrafter/
    ├── diffusion_pytorch_model.safetensors
    ├── unet_config.json
    └── scheduler_config.json
```

>  Folder must include config + safetensors for DepthCrafter to load properly.

---

Place your models as shown and launch the app — VisionDepth3D will handle the rest!  
You can name this file `README.md` and drop it into the `weights/` folder as a guide.
