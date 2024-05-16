from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from io import BytesIO
import time
import logging
import asyncio
from pydantic import BaseModel
import uuid
import os
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def log_duration(request: Request, call_next):
    start_time = time.time()

    # process request
    response = await call_next(request)

    # calculate duration
    duration = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.2f} seconds")

    return response


# Ensure model is on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = os.environ.get("MODEL_NAME", "stabilityai/sdxl-turbo")

# Load model and assign it to available GPUs
num_gpus = torch.cuda.device_count()
txt2img_pipes = [
    AutoPipelineForText2Image.from_pretrained(
        model_name, torch_dtype=torch.float16, variant="fp16"
    ).to(f"cuda:{i}")
    for i in range(num_gpus)
]
img2img_pipes = [
    AutoPipelineForImage2Image.from_pretrained(
        model_name,
        vae=txt2img_pipes[i].vae,
        text_encoder=txt2img_pipes[i].text_encoder,
        tokenizer=txt2img_pipes[i].tokenizer,
        unet=txt2img_pipes[i].unet,
        scheduler=txt2img_pipes[i].scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(f"cuda:{i}")
    for i in range(num_gpus)
]

# Locks for managing concurrent access to GPUs
gpu_locks = [asyncio.Lock() for _ in range(num_gpus)]

# Counter for round-robin GPU selection
current_gpu = 0

logger.info(f"{num_gpus} GPU(s) initialized")


class Txt2ImgInput(BaseModel):
    prompt: str


@app.post("/txt2img")
async def txt2img(input_data: Txt2ImgInput):
    global current_gpu

    try:
        # Select a GPU using round-robin
        gpu_id = current_gpu % num_gpus

        # Update the GPU counter
        current_gpu += 1

        # Lock the selected GPU to prevent other requests from using it simultaneously
        async with gpu_locks[gpu_id]:
            # Ensure no gradients are calculated for faster inference
            with torch.no_grad():
                # Ensure to use the selected GPU for computations
                pipe = txt2img_pipes[gpu_id]

                image = pipe(
                    prompt=input_data.prompt, num_inference_steps=4, guidance_scale=0.0
                ).images[0]

            # Convert image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Generate a random UUID
            file_uuid = uuid.uuid4()

            # Define a filename
            filename = f"{file_uuid}.png"

            # Return image as response with Content-Disposition header
            headers = {"Content-Disposition": f"attachment; filename={filename}"}
            return StreamingResponse(
                img_byte_arr, media_type="image/png", headers=headers
            )

    except Exception as e:
        # Log the error message
        logger.error(f"An error occurred: {str(e)}")

        # Log the full stack trace
        logger.error("Exception traceback:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class Img2ImgInput(BaseModel):
    prompt: str = Form()
    image: UploadFile = File()


@app.post("/img2img")
async def img2img(input_data: Img2ImgInput):
    global current_gpu
    try:
        # Select a GPU using round-robin
        gpu_id = current_gpu % num_gpus

        # Update the GPU counter
        current_gpu += 1

        # Lock the selected GPU to prevent other requests from using it simultaneously
        async with gpu_locks[gpu_id]:
            # Ensure no gradients are calculated for faster inference
            with torch.no_grad():
                # Ensure to use the selected GPU for computations
                pipe = img2img_pipes[gpu_id]

                # Read image bytes and convert to PIL Image
                image_bytes = await input_data.image.read()
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

                # Convert PIL Image to numpy array and scale to [0, 1]
                np_image = np.array(pil_image) / 255.0
                np_image = np_image.astype(np.float32)

                # Convert numpy array to PyTorch tensor and adjust to FP16
                tensor_image = (
                    torch.from_numpy(np_image)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(torch.float16)
                )

                image = pipe(
                    prompt=input_data.prompt,
                    image=tensor_image,
                    num_inference_steps=4,
                    strength=0.5,
                    guidance_scale=0.0,
                ).images[0]

            # Convert image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Generate a random UUID
            file_uuid = uuid.uuid4()

            # Define a filename
            filename = f"{file_uuid}.png"

            # Return image as response with Content-Disposition header
            headers = {"Content-Disposition": f"attachment; filename={filename}"}
            return StreamingResponse(
                img_byte_arr, media_type="image/png", headers=headers
            )
    except Exception as e:
        # Log the error message
        logger.error(f"An error occurred: {str(e)}")

        # Log the full stack trace
        logger.error("Exception traceback:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
