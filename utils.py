import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, UnidentifiedImageError

from u2net import U2NET

# 🎯 模型配置
MODEL_URL = "https://huggingface.co/lilpotat/pytorch3d/resolve/346374a95673795896e94398d65700cb19199e31/u2net.pth"
MODEL_PATH = "model/u2net.pth"


def download_model(model_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"📥 下載模型：{MODEL_URL}")
        r = requests.get(MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(r.content)


def run_u2net_inference(image: Image.Image, model_path: str) -> Image.Image:
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    transform = transforms.Compose(
        [transforms.Resize((320, 320)), transforms.ToTensor()]
    )
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        d1, *_ = net(input_tensor)
        mask = d1[0][0].numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)

    return mask_image


def feather_alpha(image: Image.Image, radius: int = 2) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    r, g, b, a = image.split()
    a = a.filter(ImageFilter.GaussianBlur(radius))
    return Image.merge("RGBA", (r, g, b, a))


def save_optimized_image(
    image: Image.Image,
    output_path: str,
    format: Literal["PNG", "WEBP"] = "PNG",
    crop: bool = True,
    quantize: bool = True,
    quality: int = 90,
):
    if crop:
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

    if quantize and format.upper() == "PNG":
        image = image.quantize(colors=256, method=2).convert("RGBA")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_params = {"optimize": True}
    if format.upper() == "WEBP":
        save_params["quality"] = quality

    image.save(output_path, format=format.upper(), **save_params)
    print(f"💾 儲存圖檔：{output_path}")


def remove_background(
    input_path: str,
    output_path: str,
    mask_path: str = "mask.png",
    enhance_edge: bool = True,
    format: Literal["PNG", "WEBP"] = "PNG",
    crop: bool = True,
    quantize: bool = True,
    quality: int = 90,
):
    download_model(MODEL_PATH)

    image = Image.open(input_path).convert("RGB")
    mask_image = run_u2net_inference(image, MODEL_PATH)
    mask_image.save(mask_path)

    result = image.copy()
    result.putalpha(mask_image)

    if enhance_edge:
        result = feather_alpha(result, radius=2)

    save_optimized_image(
        image=result,
        output_path=output_path,
        format=format,
        crop=crop,
        quantize=quantize,
        quality=quality,
    )

    print(f"✅ 完成輸出：{output_path}")
    print(f"🌀 遮罩儲存：{mask_path}")
