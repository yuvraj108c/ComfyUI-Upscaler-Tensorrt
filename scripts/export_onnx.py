# place this file inside comfyui root directory
# download the upscale models & place inside models/upscaler_models
# pip install onnx onnxscript
# edit model paths accordingly 
# set model scale accordingly

import torch
import folder_paths
from spandrel import ModelLoader, ImageModelDescriptor

model_name = "2x-ESRGAN.pth"
onnx_save_path = "./2x-ESRGAN.onnx"
model_scale = 2 # change to 1/2/4 accordingly

model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
model = ModelLoader().load_from_file(model_path).model.eval().cuda()

# Check dynamic shapes for esrgan 4x model
def supports_dynamic_shapes_esrgan(model, scale=model_scale):

    input_shapes = [
    (1, 3, 64, 64),
    (1, 3, 128, 128),
    (1, 3, 256, 192),
    (1, 3, 512, 256),
    (1, 3, 512, 512)
    ]

    all_passed = True

    with torch.no_grad():
        for shape in input_shapes:
            try:
                dummy_input = torch.randn(*shape).cuda()
                output = model(dummy_input)

                expected_h = shape[2] * scale
                expected_w = shape[3] * scale

                assert output.shape[0] == shape[0], "Batch size mismatch"
                assert output.shape[1] == shape[1], "Channel mismatch"
                assert output.shape[2] == expected_h, f"Height mismatch: expected {expected_h}, got {output.shape[2]}"
                assert output.shape[3] == expected_w, f"Width mismatch: expected {expected_w}, got {output.shape[3]}"

                print(f"Success: input {shape} → output {output.shape}")
            except Exception as e:
                all_passed = False
                print(f"Failure: input {shape} → error: {e}")
                torch.cuda.empty_cache()

    if all_passed: print(f"Success: Dynamic shapes supported.")
    if not all_passed: print(f"Failure: Dynamic shapes NOT supported.")
    return all_passed

# Use smaller dummy input if model supports
if supports_dynamic_shapes_esrgan(model):
    shape = (1, 3, 64, 64)
    print(f"Using {shape} input (less VRAM usage)")
else:
    shape = (1, 3, 512, 512)
    print(f"Using {shape} input (large VRAM usage)")

x = torch.rand(*shape).cuda()

dynamic_axes = {
    "input": {0: "batch_size", 2: "width", 3: "height"},
    "output": {0: "batch_size", 2: "width", 3: "height"},
}

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        onnx_save_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        export_params=True,
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

print("Saved onnx to:", onnx_save_path)