import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from utils import remove_background

app = FastAPI(title="U^2-Net Background Removal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output"
ALLOWED_FORMATS = {"png", "webp"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_input_image(file: UploadFile) -> str:
    if not file.content_type.startswith("image/"):
        raise ValueError("Only image files are supported.")

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    try:
        Image.open(path).verify()
    except UnidentifiedImageError:
        os.remove(path)
        raise ValueError("Invalid or corrupted image.")

    return path


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        # 圖片準備
        input_path = prepare_input_image(file)

        # 輸出設定
        output_format = "png"  # or "png"
        if output_format not in ALLOWED_FORMATS:
            raise ValueError("Unsupported output format.")

        output_name = f"{uuid4().hex}.{output_format}"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # 去背處理
        remove_background(
            input_path=input_path,
            output_path=output_path,
            enhance_edge=True,
            format=output_format,
            crop=True,
            quantize=False,
            quality=90,
        )

        return FileResponse(
            output_path,
            media_type=f"image/{output_format}",
            filename=output_name,
        )

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)

    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)
