import torch
import time
from utilities import Engine

def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    if trt_path is None:
        trt_path = input("Enter the path to save the TensorRT engine (e.g ./realesrgan.engine): ")
    if onnx_path is None:
        onnx_path = input("Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")

    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
        input_profile=[
            {"input": [(1,3,256,256), (1,3,512,512), (1,3,1280,1280)]}, # any sizes from 256x256 to 1280x1280
        ],
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret

export_trt()
