# download the upscale models & place inside models/upscaler_models
# edit model paths accordingly 

import torch
import folder_paths
from spandrel import ModelLoader, ImageModelDescriptor

model_name = "4xNomos2_otf_esrgan.pth"
onnx_save_path = "./4xNomos2_otf_esrgan.onnx"

model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
model = ModelLoader().load_from_file(model_path).model.eval().cuda()

x = torch.rand(1, 3, 512, 512).cuda()

dynamic_axes = {
    "input": {0: "batch_size", 2: "width", 3: "height"},
    "output": {0: "batch_size", 2: "width", 3: "height"},
}
  
torch.onnx.export(model,
                    x,
                    onnx_save_path,
                    verbose=True,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=17,
                    export_params=True,
                    dynamic_axes=dynamic_axes,
                )
print("Saved onnx to:", onnx_save_path)