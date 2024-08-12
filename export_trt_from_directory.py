import os
import torch
import time
from utilities import Engine

def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    option = input("Choose an option:\n1. Convert a single ONNX file\n2. Convert all ONNX files in a directory\nEnter your choice (1 or 2): ")

    if option == '1':
        onnx_path = input("Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")
        onnx_files = [onnx_path]
        trt_dir = input("Enter the path to save the TensorRT engine (e.g ./trt_engine/): ")
    elif option == '2':
        onnx_dir = input("Enter the directory path containing ONNX models (e.g ./onnx_models/): ")
        onnx_files = [os.path.join(onnx_dir, file) for file in os.listdir(onnx_dir) if file.endswith('.onnx')]
        if not onnx_files:
            raise ValueError(f"No .onnx files found in directory: {onnx_dir}")
        trt_dir = input("Enter the directory path to save the TensorRT engines (e.g ./trt_engine/): ")
    else:
        raise ValueError("Invalid option. Please choose either 1 or 2.")

    # Check if trt_dir already exists as a directory
    if not os.path.exists(trt_dir):
        os.makedirs(trt_dir)
        
    #os.makedirs(trt_dir, exist_ok=True)
    total_files = len(onnx_files)
    for index, onnx_path in enumerate(onnx_files):
        engine = Engine(trt_path)

        torch.cuda.empty_cache()
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        trt_path = os.path.join(trt_dir, f"{base_name}.engine")

        print(f"Converting {onnx_path} to {trt_path}")

        s = time.time()

        # Initialize Engine with trt_path and clear CUDA cache
        engine = Engine(trt_path)
        torch.cuda.empty_cache()

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
        if index < total_files - 1:
            # Delay for 10 seconds
            print("Delaying for 10 seconds...")
            time.sleep(10)
            print("Resuming operations after delay...")

    return

export_trt()
