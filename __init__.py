import os
import folder_paths
import numpy as np
import torch
import cv2
from comfy.utils import ProgressBar
from .utilities import Engine

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")


class UpscalerTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "engine": (os.listdir(ENGINE_DIR),),
                "resize_to": (["none", "HD", "FHD", "2k", "4k"],),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def get_final_resolutions(self, width, height, resize_to):
        final_width = None
        final_height = None
        aspect_ratio = float(width/height)

        match resize_to:
            case "HD":
                final_width = 1280
                final_height = 720
            case "FHD":
                final_width = 1920
                final_height = 1080
            case "2k":
                final_width = 2560
                final_height = 1440
            case "4k":
                final_width = 3840
                final_height = 2160
            case "none":
                final_width = width*4
                final_height = height*4

        if aspect_ratio == 1.0:
            final_width = final_height

        if aspect_ratio < 1.0 and resize_to != "none":
            temp = final_width
            final_width = final_height
            final_height = temp

        return (final_width, final_height)

    def main(self, images, engine, resize_to):
        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape

        shape_dict = {
            "input": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H*4, W*4)},
        }

        # setup tensorrt engine
        engine_path = os.path.join(ENGINE_DIR, engine)
        if (not hasattr(self, 'engine') or self.engine_label != engine):
            self.engine = Engine(engine_path)
            self.engine.load()
            self.engine.activate()
            self.engine_label = engine
        else:
            print(f"Using cached TensorRT engine: {engine_path}")

        # dynamic memory allocation, can't be cached
        self.engine.allocate_buffers(shape_dict=shape_dict)

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))

        upscaled_frames = []
        final_width, final_height = self.get_final_resolutions(W, H, resize_to)

        for img in images_list:
            result = self.engine.infer({"input": img}, cudaStream)

            output = result['output'].cpu().numpy().squeeze(0)
            output = np.transpose(output, (1, 2, 0))
            output = np.clip(255.0 * output, 0, 255).astype(np.uint8)
            output = cv2.resize(output, (final_width, final_height))

            upscaled_frames.append(output)
            pbar.update(1)

        upscaled_frames_np = np.array(
            upscaled_frames).astype(np.float32) / 255.0
        return (torch.from_numpy(upscaled_frames_np),)


NODE_CLASS_MAPPINGS = {
    "UpscalerTensorrt": UpscalerTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerTensorrt": "Upscaler Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
