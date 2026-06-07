from .nodes.load_tensorrt_model import LoadUpscalerTensorrtModel
from .nodes.upscaler_tensorrt import UpscalerTensorrt


NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrt": UpscalerTensorrt,
    "LoadUpscalerTensorrtModel": LoadUpscalerTensorrtModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "Upscaler Tensorrt ⚡",
    "LoadUpscalerTensorrtModel": "Load Upscale Tensorrt Model",
}

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]