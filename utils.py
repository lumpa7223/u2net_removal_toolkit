import os

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image

from u2net import U2NET  # 請確保你已加入官方 u2net.py


def download_model(model_path):
    if not os.path.exists(model_path):
        url = "https://huggingface.co/reidn3r/u2net-image-rembg/resolve/main/u2net.pth"
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)


def remove_background(input_path, output_path, mask_path="mask.png"):
    model_path = "model/u2net.pth"
    os.makedirs("model", exist_ok=True)
    download_model(model_path)

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    image = Image.open(input_path).convert("RGB")
    transform = transforms.Compose(
        [transforms.Resize((320, 320)), transforms.ToTensor()]
    )
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        d1, *_ = net(input_tensor)
        mask = d1[0][0].numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)

    # 輸出遮罩圖
    # mask_image.save(mask_path)

    # 建立透明圖像
    image.putalpha(mask_image)
    image.save(output_path)
