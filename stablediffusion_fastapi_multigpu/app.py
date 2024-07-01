from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from io import BytesIO
import time
import logging
import asyncio
from pydantic import BaseModel
from uuid import uuid4
import os
from PIL import Image
import numpy as np
from redis import Redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

redis_client = Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=os.environ.get("REDIS_PORT", 6379),
    db=0,
)


@app.middleware("http")
async def log_duration(request: Request, call_next):
    start_time = time.time()

    # process request
    response = await call_next(request)

    # calculate duration
    duration = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.2f} seconds")

    return response


# Ensure model is on GPU
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model_name = os.environ.get("MODEL_NAME", "stabilityai/sdxl-turbo")

# Load models and assign them to the GPU
txt2img_pipeline = AutoPipelineForText2Image.from_pretrained(
    model_name, torch_dtype=torch.float16, variant="fp16"
).to(device)
img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    variant="fp16",
    vae=txt2img_pipeline.vae,
    text_encoder=txt2img_pipeline.text_encoder,
    tokenizer=txt2img_pipeline.tokenizer,
    unet=txt2img_pipeline.unet,
    scheduler=txt2img_pipeline.scheduler,
    safety_checker=None,
    feature_extractor=None,
).to(device)

# Locks for managing concurrent access in case not behind router
gpu_lock = asyncio.Lock()
logger.info("GPU initialized")

# Directory where uploaded images will be stored
IMAGE_DIR = "uploaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


class Txt2ImgInput(BaseModel):
    prompt: str
    num_inference_steps: int = 4


@app.post("/txt2img")
async def txt2img(input_data: Txt2ImgInput):
    try:
        # Lock the selected GPU to prevent other requests from using it simultaneously
        async with gpu_lock:
            # Ensure no gradients are calculated for faster inference
            with torch.no_grad():
                pipe = txt2img_pipeline

                image = pipe(
                    prompt=input_data.prompt,
                    num_inference_steps=input_data.num_inference_steps,
                    guidance_scale=0.0,
                ).images[0]

            # Convert image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Generate a random UUID
            file_uuid = uuid4()

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


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Generate a unique file ID
    file_id = str(uuid4())

    # Save the uploaded file to memory
    file_content = await file.read()

    # Store file content in Redis
    redis_client.set(file_id, file_content)

    # Load the image and convert to PyTorch tensor
    pil_image = Image.open(BytesIO(file_content)).convert("RGB")
    np_image = np.array(pil_image) / 255.0
    np_image = np_image.astype(np.float16)
    tensor_image = (
        torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).to(torch.float16)
    )

    # Save the tensor to disk
    tensor_bytes = BytesIO()
    torch.save(tensor_image, tensor_bytes)
    tensor_bytes.seek(0)

    # Store tensor in Redis
    redis_client.set(f"{file_id}_tensor", tensor_bytes.getvalue())

    return {"file_id": file_id}


class Img2ImgInput(BaseModel):
    prompt: str
    file_id: str
    num_inference_steps: int = 4
    strength: float = 0.5


@app.post("/img2img")
async def img2img(input_data: Img2ImgInput):
    # Retrieve tensor data from Redis
    tensor_data = redis_client.get(f"{input_data.file_id}_tensor")
    if not tensor_data:
        raise HTTPException(status_code=404, detail="File not found")

    tensor_bytes = BytesIO(tensor_data)
    tensor_bytes.seek(0)
    tensor_image = torch.load(tensor_bytes)

    try:
        # Lock the selected GPU to prevent other requests from using it simultaneously
        async with gpu_lock:
            # Ensure no gradients are calculated for faster inference
            with torch.no_grad():
                pipe = img2img_pipeline

                image = pipe(
                    prompt=input_data.prompt,
                    image=tensor_image,
                    num_inference_steps=input_data.num_inference_steps,
                    strength=input_data.strength,
                    guidance_scale=0.0,
                ).images[0]

            # Convert image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Generate a random UUID
            file_uuid = uuid4()

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
