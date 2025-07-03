import logging
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from u2net import U2NET

# ─── 設定日誌 ─────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

# ─── 模型參數與全域緩存 ────────────────────────────────────
MODEL_URL = (
    "https://huggingface.co/lilpotat/pytorch3d/"
    "resolve/346374a95673795896e94398d65700cb19199e31/u2net.pth"
)
MODEL_PATH = "model/u2net.pth"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_u2net_model: U2NET | None = None


def download_model(model_path: str):
    """下載 U²-Net 權重到本地（若不存在）。"""
    try:
        os.makedirs(Path(model_path).parent, exist_ok=True)
        if not Path(model_path).exists():
            logging.info(f"下載模型到 {model_path}")
            resp = requests.get(MODEL_URL, timeout=30)
            resp.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(resp.content)
            logging.info("模型下載完成")
    except Exception:
        logging.exception("下載模型時發生錯誤")
        raise


def get_u2net() -> U2NET:
    """單例模式載入並回傳 U²-Net 模型（放到 CPU/GPU）。"""
    global _u2net_model
    if _u2net_model is None:
        try:
            net = U2NET(3, 1).to(_device)
            state = torch.load(MODEL_PATH, map_location=_device)
            net.load_state_dict(state)
            net.eval()
            _u2net_model = net
            logging.info(f"U2NET 模型載入完成 (device={_device})")
        except Exception:
            logging.exception("載入模型時發生錯誤")
            raise
    return _u2net_model


def run_u2net_inference(
    image: Image.Image, thresh: float = 0.2, kernel_size: int = 3
) -> Image.Image:
    """對 PIL 圖做 U²-Net 推理，回傳二值化 & 清理後的遮罩 PIL.Image."""
    net = get_u2net()
    try:
        transform = transforms.Compose(
            [
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            d1, *_ = net(tensor)
            mask = d1[0][0].cpu().numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min())

            # 閾值化
            bin_mask = (mask > thresh).astype(np.uint8) * 255
            # 開運算清雜點
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            clean = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
            mask_img = Image.fromarray(clean).resize(image.size)
            return mask_img
    except Exception:
        logging.exception("推理或遮罩處理時發生錯誤")
        raise


def feather_alpha(image: Image.Image, radius: int = 2) -> Image.Image:
    """羽化透明通道邊緣。"""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    r, g, b, a = image.split()
    a = a.filter(ImageFilter.GaussianBlur(radius))
    return Image.merge("RGBA", (r, g, b, a))


def save_optimized_image(
    image: Image.Image,
    output_path: str,
    format: Literal["PNG", "WEBP"] = "PNG",  # 改用 format
    crop: bool = True,
    quantize: bool = True,
    quality: int = 90,
):
    """裁切多餘透明區、量化 & 優化後存檔。"""
    try:
        if crop:
            bbox = image.getbbox()
            if bbox:
                image = image.crop(bbox)

        if quantize and format.upper() == "PNG":
            image = image.quantize(colors=256, method=2).convert("RGBA")

        os.makedirs(Path(output_path).parent, exist_ok=True)
        params = {"optimize": True}
        if format.upper() == "WEBP":
            params["quality"] = quality

        image.save(output_path, format.upper(), **params)
        logging.info(f"圖檔儲存：{output_path}")
    except Exception:
        logging.exception("儲存圖檔時發生錯誤")
        raise


def remove_background(
    input_path: str,
    output_path: str,
    mask_path: str = "mask.png",
    thresh: float = 0.2,
    kernel_size: int = 3,
    feather_radius: int = 2,
    enable_autocontrast: bool = True,
    enhance_edge: bool = True,
    format: Literal["PNG", "WEBP"] = "PNG",  # 改用 format
    crop: bool = True,
    quantize: bool = True,
    quality: int = 90,
):
    """主流程：下載模型 → 讀圖 → 前處理 → 推理 → 後處理 → 存檔。"""
    download_model(MODEL_PATH)

    try:
        img = Image.open(input_path).convert("RGB")
    except UnidentifiedImageError:
        logging.error(f"無法讀取檔案：{input_path}")
        return
    except Exception:
        logging.exception("讀入圖檔時發生未知錯誤")
        return

    if enable_autocontrast:
        img = ImageOps.autocontrast(img, cutoff=1)
        logging.info("已自動對比增強")

    # 推理並儲存遮罩
    mask_img = run_u2net_inference(img, thresh=thresh, kernel_size=kernel_size)
    try:
        mask_img.save(mask_path)
        logging.info(f"遮罩儲存：{mask_path}")
    except Exception:
        logging.exception("儲存遮罩時發生錯誤")

    # 合成結果
    result = img.copy()
    result.putalpha(mask_img)

    if enhance_edge:
        result = feather_alpha(result, radius=feather_radius)
        logging.info(f"已羽化邊緣 (radius={feather_radius})")

    # 存檔時傳入 format
    save_optimized_image(
        image=result,
        output_path=output_path,
        format=format,
        crop=crop,
        quantize=quantize,
        quality=quality,
    )
    logging.info("背景移除完成")
