import os
import folder_paths
from ..trt_utilities import Engine
from ..utilities import download_file, logger, LOAD_UPSCALER_NODE_CONFIG
import comfy.model_management as mm
import time

# Support TensorRT-RTX
TENSORRT_RTX_AVAILABLE = False
import importlib
if importlib.util.find_spec('tensorrt_rtx') is not None:
    import tensorrt_rtx as trt
    TENSORRT_RTX_AVAILABLE = True
    logger.info("Using TensorRT RTX")
else:
    import tensorrt as trt

IMAGE_DIM_MIN = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MIN")
IMAGE_DIM_OPT = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_OPT")
IMAGE_DIM_MAX = LOAD_UPSCALER_NODE_CONFIG.get("IMAGE_DIM_MAX")

class LoadUpscalerTensorrtModel:
    @classmethod
    def INPUT_TYPES(cls):
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("models", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})

        model_options = [m["name"] for m in model_config]
        model_default = model_options[0] if model_options else "4x-UltraSharp"

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")

        return {
            "required": {
                "model": (model_options, {"default": model_default}),
                "precision": (precision_options, {"default": precision_default}),
            }
        }

    RETURN_NAMES = ("upscaler_trt_model",)
    RETURN_TYPES = ("UPSCALER_TRT_MODEL",)
    FUNCTION = "load_upscaler_tensorrt_model"
    CATEGORY = "tensorrt"

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
            tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision if not TENSORRT_RTX_AVAILABLE else 'rtx'}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{trt.__version__}.trt")

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
            engine.model_name = model

            return (engine,)
