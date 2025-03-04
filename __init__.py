import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
import comfy.model_management as mm
import time

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

class UpscalerTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to be upscaled. Resolution must be between 256 and 1280 px"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "resize_to": (["none", "HD", "FHD", "2k", "4k"],{"tooltip": "Resize the upscaled image to fixed resolutions, optional"}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscaler_tensorrt"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Upscale images with tensorrt"

    def upscaler_tensorrt(self, images, upscaler_trt_model, resize_to):
        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape
        final_width, final_height = get_final_resolutions(W, H, resize_to)
        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*4}, W:{W*4} | Final resolution: H:{final_height}, W:{final_width} | resize_to: {resize_to}")

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*4, W*4)},
        }
        # setup engine
        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = []

        for img in images_list:
            result = upscaler_trt_model.infer({"input": img}, cudaStream)
            upscaled_frames.append(result['output'])
            pbar.update(1)

        outputs = torch.cat(upscaled_frames, dim=0)
        
        # Batch resize if needed
        upscaled_W, upscaled_H = outputs.shape[2], outputs.shape[3]
        if upscaled_W != final_width or upscaled_H != final_height:
            logger.info(f"Resizing to H:{final_height}, W:{final_width}")
            outputs = torch.nn.functional.interpolate(
                outputs, 
                size=(final_height, final_width),
                mode='bicubic',
                antialias=True
            )

        upscaler_trt_model.reset() # frees engine vram
        mm.soft_empty_cache()
        return (outputs.permute(0, 2, 3, 1).to(mm.intermediate_device()),)

class LoadUpscalerTensorrtModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["4x-AnimeSharp", "4x-UltraSharp", "4x-WTP-UDS-Esrgan", "4x_NMKD-Siax_200k", "4x_RealisticRescaler_100000_G", "4x_foolhardy_Remacri", "RealESRGAN_x4", "4xNomos2_otf_esrgan"], {"default": "4x-UltraSharp", "tooltip": "These models have been tested with tensorrt"}),
                "precision": (["fp16", "fp32"], {"default": "fp16", "tooltip": "Precision to build the tensorrt engines"}),
            }
        }
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_upscaler_tensorrt_model"
    
    def load_upscaler_tensorrt_model(self, model, precision):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}.trt")

        # Download onnx & build tensorrt engine
        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                logger.info(f"Onnx model found at: {onnx_model_path}")

            # Build tensorrt engine
            logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" else False, # mixed precision not working TODO: investigate
                input_profile=[
                    {"input": [(1,3,256,256), (1,3,512,512), (1,3,1280,1280)]}, # any sizes from 256x256 to 1280x1280
                    # should this power be given to people to decide?
                ],
            )
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

        # Load tensorrt model
        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()

        return (engine,)

NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrt": UpscalerTensorrt,
    "LoadUpscalerTensorrtModel": LoadUpscalerTensorrtModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "Upscaler Tensorrt ⚡",
    "LoadUpscalerTensorrtModel": "Load Upscale Tensorrt Model",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
