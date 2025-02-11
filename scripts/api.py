import os
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Optional, List
import gradio as gr
from PIL import Image
import numpy as np

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from scripts.anydoor import run_api


def decode_to_pil(image):
    if type(image) is str:
        return decode_base64_to_image(image)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        Exception("Not an image")


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image).decode()
    elif type(image) is np.ndarray:
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    else:
        Exception("Invalid type")


def anydoor_api(_: gr.Blocks, app: FastAPI):
    @app.get("/anydoor/heartbeat")
    async def heartbeat():
        return {
            "msg": "Success!",
            "status_code": 200,
        }

    class AnydoorPredictRequest(BaseModel):
        input_image: str
        # input_mask: str
        ref_image: str
        ref_mask: str
        bg_mask_x1: float
        bg_mask_y1: float
        bg_mask_x2: float
        bg_mask_y2: float
        num_samples: Optional[int] = 1
        strength: Optional[float] = 1.0
        ddim_steps: Optional[int] = 30
        scale: Optional[float] = 4.5
        enable_shape_control: Optional[bool] = False
        reference_mask_refine: Optional[bool] = False

    @app.post("/anydoor/predict")
    async def api_anydoor_predict(payload: AnydoorPredictRequest = Body(...)) -> Any:
        print(f"ANYDOOR API /anydoor/predict received request")
        payload.input_image = decode_to_pil(payload.input_image).convert("RGB")
        # payload.input_mask = decode_to_pil(payload.input_mask).convert("L")
        payload.ref_image = decode_to_pil(payload.ref_image).convert("RGB")
        payload.ref_mask = decode_to_pil(payload.ref_mask).convert("L")

        try:
            outputs = run_api(
                payload.input_image,
                # payload.input_mask,
                payload.ref_image,
                payload.ref_mask,
                payload.bg_mask_x1,
                payload.bg_mask_y1,
                payload.bg_mask_x2,
                payload.bg_mask_y2,
                payload.num_samples,
                payload.reference_mask_refine,
                payload.strength,
                payload.ddim_steps,
                payload.scale,
                payload.enable_shape_control,
            )
            message = "Success!"
            status_code = 200
        except Exception as e:
            outputs = None
            message = f"Failed: {e}"
            status_code = 400

        print(f"ANYDOOR API /anydoor/predict finished with message: {message}")
        result = {
            "msg": message,
            "status_code": status_code,
        }
        if outputs:
            result["images"] = list(map(encode_to_base64, outputs))

        return result

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(anydoor_api)
except:
    print("ANYDOOR Web UI API failed to initialize")
