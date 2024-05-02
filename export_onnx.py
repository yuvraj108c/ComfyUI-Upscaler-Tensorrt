# run this file inside comfyui root directory
# download the upscale models & place inside models/upscaler_models
# edit the paths accordingly 

import os
from comfy_extras.chainner_models import model_loading
from comfy import model_management
import torch
import comfy.utils
import folder_paths
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device=output_device)
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                x = max(0, min(s.shape[-1] - overlap, x))
                y = max(0, min(s.shape[-2] - overlap, y))
                s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                print(s_in.shape)
                ps = function(s_in).to(output_device)
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                        mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b:b+1] = out/out_div
    return output

def load_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
    out = model_loading.load_state_dict(sd).eval()
    return out

def upscale(upscale_model, image):
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)
    free_memory = model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s

def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]



# img = cv2.imread("/ComfyUI/1.png", cv2.IMREAD_COLOR)

# transform = transforms.Compose([transforms.ToTensor()])
# img_t = transform(img).unsqueeze(0).permute(0, 2, 3, 1)

upscale_model = load_model("RealESRGAN_x4.pth")
# upscaled_image_t = upscale(upscale_model, img_t)

# tensor2pil(upscaled_image_t)[0].save("upscaled.jpg")

x = torch.rand(1, 3, 512, 512)
# x = x.cuda()

dynamic_axes = {
    "input": {0: "batch_size", 2: "width", 3: "height"},
    "output": {0: "batch_size", 2: "width", 3: "height"},
}
    
torch.onnx.export(upscale_model,
                    x,
                    "/workspace/ComfyUI/RealESRGAN_x4.onnx",
                    verbose=True,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=17,
                    export_params=True,
                    dynamic_axes=dynamic_axes,
                    )

# trtexec --fp16 --onnx=4x_ultrasharp.onnx --minShapes=input:1x3x1x1 --optShapes=input:1x3x256x256 --maxShapes=input:1x3x512x512 --saveEngine=model.engine