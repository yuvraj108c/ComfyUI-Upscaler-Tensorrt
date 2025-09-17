import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
import comfy.model_management as mm
import time
import json # <--- Import json module

# Support TensorRT-RTX
TENSORRT_RTX_AVAILABLE = False
import importlib
if importlib.util.find_spec('tensorrt_rtx') is not None:
    import tensorrt_rtx as tensorrt
    TENSORRT_RTX_AVAILABLE = True
else:
    import tensorrt

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 512
IMAGE_DIM_MAX = 1280

# --- Function to load configuration ---
def load_node_config(config_filename="load_upscaler_config.json"):
    """Loads node configuration from a JSON file."""
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)
    
    default_config = { # Fallback in case file is missing or corrupt
        "model": {
            "options": ["4x-UltraSharp"],
            "default": "4x-UltraSharp",
            "tooltip": "Default model (fallback from code)"
        },
        "precision": {
            "options": ["fp16", "fp32"],
            "default": "fp16",
            "tooltip": "Default precision (fallback from code)"
        }
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_filename}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config

# --- Load the configuration once when the module is imported ---
LOAD_UPSCALER_NODE_CONFIG = load_node_config()


class UpscalerTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": f"Images to be upscaled. Resolution must be between {IMAGE_DIM_MIN} and {IMAGE_DIM_MAX} px"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "resize_to": (["none", "HD", "FHD", "2k", "4k", "2x", "3x"],{"tooltip": "Resize the upscaled image to fixed resolutions, optional"}),
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

        for dim in (H, W):
            if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
                raise ValueError(f"Input image dimensions fall outside of the supported range: {IMAGE_DIM_MIN} to {IMAGE_DIM_MAX} px!\nImage dimensions: {W}px by {H}px")

        final_width, final_height = get_final_resolutions(W, H, resize_to)
        logger.info(f"Upscaling {B} images from H:{H}, W:{W} to H:{H*4}, W:{W*4} | Final resolution: H:{final_height}, W:{final_width} | resize_to: {resize_to}")

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*4, W*4)},
        }
        upscaler_trt_model.activate()
        upscaler_trt_model.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device())
        must_resize = W*4 != final_width or H*4 != final_height

        for i, img in enumerate(images_list):
            result = upscaler_trt_model.infer({"input": img}, cudaStream)
            result = result["output"]

            if must_resize:
                result = torch.nn.functional.interpolate(
                    result, 
                    size=(final_height, final_width),
                    mode='bicubic',
                    antialias=True
                )
            upscaled_frames[i] = result.to(mm.intermediate_device())
            pbar.update(1)

        output = upscaled_frames.permute(0, 2, 3, 1)
        upscaler_trt_model.reset()
        mm.soft_empty_cache()

        logger.info(f"Output shape: {output.shape}")
        return (output,)

class LoadUpscalerTensorrtModel:
    @classmethod
    def INPUT_TYPES(cls): # Changed 's' to 'cls' for convention
        # Use the pre-loaded configuration
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})
        
        # Provide sensible defaults if keys are missing in the config (though load_node_config handles this broadly)
        model_options = model_config.get("options", ["4x-UltraSharp"])
        model_default = model_config.get("default", "4x-UltraSharp")
        model_tooltip = model_config.get("tooltip", "Select a model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        return {
            "required": {
                "model": (model_options, {"default": model_default, "tooltip": model_tooltip}),
                "precision": (precision_options, {"default": precision_default, "tooltip": precision_tooltip}),
            }
        }
    
    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    # FUNCTION = "main" # This was duplicated, removing
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_upscaler_tensorrt_model" # This is the correct one
    
    def load_upscaler_tensorrt_model(self, model, precision):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")
        
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = 1, 1, 1
        engine_min_h, engine_opt_h, engine_max_h = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        engine_min_w, engine_opt_w, engine_max_w = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision if not TENSORRT_RTX_AVAILABLE else 'rtx'}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                logger.info(f"Onnx model found at: {onnx_model_path}")

            logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16= True if precision == "fp16" and not TENSORRT_RTX_AVAILABLE else False,
                input_profile=[
                    {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_min_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]},
                ],
            )
            e = time.time()
            logger.info(f"Time taken to build: {(e-s)} seconds")

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
