import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from utils import remove_background

app = FastAPI()


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    input_path = f"input_{file.filename}"
    output_path = f"output/{file.filename.split('.')[0]}.png"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    remove_background(input_path, output_path)
    os.remove(input_path)
    return FileResponse(output_path, media_type="image/png")
