<div align="center">

# ComfyUI Upscaler TensorRT ⚡

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.8-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

<p align="center">
  <img src="assets/node_v3.png" />

</p>

This project provides a [Tensorrt](https://github.com/NVIDIA/TensorRT) implementation for fast image upscaling inside ComfyUI (2-4x faster)

This project is licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), everyone is FREE to access, use, modify and redistribute with the same license.

For commercial purposes, please contact me directly at yuvraj108c@gmail.com

If you like the project, please give me a star! ⭐

---

## ⏱️ Performance

_Note: The following results were benchmarked on FP16 engines inside ComfyUI, using 100 identical frames_

| Device |     Model     | Input Resolution (WxH) | Output Resolution (WxH) | FPS |
| :----: | :-----------: | :--------------------: | :---------------------: | :-: |
|  H100  | 4x-UltraSharp |       512 x 512        |       2048 x 2048       |  10  |
|  H100  | 4x-UltraSharp |       960 x 540        |       3840 x 2160       |  5  |
|  H100  | 4x-UltraSharp |       1280 x 1280      |       5120 x 5120       |  1.7  |
|  RTX5090  | 4x-UltraSharp |       512 x 512        |       2048 x 2048       |  11  |
|  RTX5090  | 4x-UltraSharp |       960 x 540        |       3840 x 2160       |  5  |
|  RTX5090  | 4x-UltraSharp |       1280 x 1280      |       5120 x 5120       |  1.7  |

## 🚀 Installation
- Install via the manager
- Or, navigate to the `/ComfyUI/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt.git
cd ./ComfyUI-Upscaler-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Supported Models

- These upscaler models have been tested to work with Tensorrt. Onnx are available [here](https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/tree/main)
- The exported tensorrt models support dynamic image resolutions from 256x256 to 1280x1280 px (e.g 960x540, 512x512, 1280x720 etc).

   - [4x-AnimeSharp](https://openmodeldb.info/models/4x-AnimeSharp)
   - [4x-UltraSharp](https://openmodeldb.info/models/4x-UltraSharp)
   - [4x-WTP-UDS-Esrgan](https://openmodeldb.info/models/4x-WTP-UDS-Esrgan)
   - [4x_NMKD-Siax_200k](https://openmodeldb.info/models/4x-NMKD-Siax-CX)
   - [4x_RealisticRescaler_100000_G](https://openmodeldb.info/models/4x-RealisticRescaler)
   - [4x_foolhardy_Remacri](https://openmodeldb.info/models/4x-Remacri)
   - [RealESRGAN_x4](https://openmodeldb.info/models/4x-realesrgan-x4plus)
   - [4xNomos2_otf_esrgan](https://openmodeldb.info/models/4x-Nomos2-otf-esrgan)

## ☀️ Usage

- Load [example workflow](assets/tensorrt_upscaling_workflow.json) 
- Choose the appropriate model from the dropdown
- The tensorrt engine will be built automatically
- Load an image of resolution between 256-1280px
- Set `resize_to` to resize the upscaled images to fixed resolutions

## 🔧 Custom Models
- To export other ESRGAN models, you'll have to build the onnx model first, using [export_onnx.py](scripts/export_onnx.py) 
- Place the onnx model in `/ComfyUI/models/onnx/YOUR_MODEL.onnx`
- Then, add your model to this list as shown: https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt/blob/8f7ef5d1f713af3b4a74a64fa13a65ee5c404cd4/__init__.py#L77
- Finally, run the same workflow and choose your model
- If you've tested another working tensorrt model, let me know to add it officially to this node

## 🚨 Updates
### 4 March 2025 (breaking)
- Automatic tensorrt engines are built from the workflow itself, to simplify the process for non-technical people
- Separate model loading and tensorrt processing into different nodes
- Optimise post processing
- Update onnx export script

## ⚠️ Known issues

- Only models with ESRGAN architecture are currently working
- High ram usage when exporting `.pth` to `.onnx`

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.4, Tensorrt 10.8, Python 3.10, H100 GPU
- Windows 11

## 👏 Credits

- [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
