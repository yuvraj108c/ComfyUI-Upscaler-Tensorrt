<div align="center">

# ComfyUI Upscaler TensorRT

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.3-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.0-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

This project is licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), everyone is FREE to access, use, modify and redistribute with the same license.  

For commercial purposes, please contact me directly at yuvraj108c@gmail.com

If you like the project, please give me a star! ⭐

****

This project provides a Tensorrt implementation for fast image upscaling inside ComfyUI (3-4x faster)

## ⏱️ Performance

_Note: The following results were benchmarked on FP16 engines inside ComfyUI, using 100 frames_

| Device |     Model     | Input Resolution (WxH) | Output Resolution (WxH) | FPS |
| :----: | :-----------: | :--------------------: | :---------------------: | :-: |
|  L40s  | RealESRGAN_x4 |       512 x 512        |       2048 x 2048       |  5  |
|  L40s  | RealESRGAN_x4 |       960 x 540        |       3840 x 2160       |  2  |
|  L40s  | RealESRGAN_x4 |      1280 x 1280       |       5120 x 5120       | 0.7 |

## 🚀 Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt.git
cd ./ComfyUI-Upscaler-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Building Tensorrt Engine

1. Download one of the [available onnx models](https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/tree/main). These onnx models support dynamic image resolutions from 256x256 to 1280x1280 px (e.g 960x540, 512x512, 1280x720 etc). Here are the original models:

   - [4x-AnimeSharp](https://openmodeldb.info/models/4x-AnimeSharp)
   - [4x-UltraSharp](https://openmodeldb.info/models/4x-UltraSharp)
   - [4x-WTP-UDS-Esrgan](https://openmodeldb.info/models/4x-WTP-UDS-Esrgan)
   - [4x_NMKD-Siax_200k](https://openmodeldb.info/models/4x-NMKD-Siax-CX)
   - [4x_RealisticRescaler_100000_G](https://openmodeldb.info/models/4x-RealisticRescaler)
   - [4x_foolhardy_Remacri](https://openmodeldb.info/models/4x-Remacri)
   - [RealESRGAN_x4](https://openmodeldb.info/models/4x-realesrgan-x4plus)

2. Run `python export_trt.py` and set onnx/engine paths accordingly
3. Place the exported engine inside ComfyUI `/models/tensorrt/upscaler` directory

## ☀️ Usage

- Insert node by `Right Click -> tensorrt -> Upscaler Tensorrt`
- Choose the appropriate engine from the dropdown

## ⚠️ Known issues

- Only models with ESRGAN architecture are currently working
- High ram usage when exporting `.pth` to `.onnx`

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.3, Tensorrt 10.0.1, Python 3.10, L40s GPU
- Windows 11

## 👏 Credits

- [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
