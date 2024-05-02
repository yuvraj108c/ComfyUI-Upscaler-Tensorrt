<div align="center">

# ComfyUI Upscaler TensorRT

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.3-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.0-green)](https://developer.nvidia.com/tensorrt)
[![mit](https://img.shields.io/badge/license-MIT-blue)](https://github.com/spacewalk01/depth-anything-tensorrt/blob/main/LICENSE)

</div>

- This project provides a Tensorrt implementation for fast image upscaling inside ComfyUI (3-4x faster)

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

1. Download one of the available [onnx models](https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/tree/main). These models support dynamic image resolutions from 256x256 to 1280x1280 px (e.g 960x540, 360x640, 1280x720 etc). You can also convert other upscaler models supported by ComfyUI to onnx using [export_onnx.py](export_onnx.py)
2. Edit model paths inside [export_trt.py](export_trt.py) accordingly and run `python export_trt.py`
3. Place the exported engine inside ComfyUI `/models/tensorrt/upscaler` directory

## ☀️ Usage

- Insert node by `Right Click -> tensorrt -> Upscaler Tensorrt`
- Choose the appropriate engine from the dropdown

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.3, Tensorrt 10.0.1, Python 3.10, L40s GPU
- Windows (Not tested)

## 👏 Credits

- [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
