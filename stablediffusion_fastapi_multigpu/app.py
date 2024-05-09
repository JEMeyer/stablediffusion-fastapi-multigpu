from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from diffusers import (
    AutoPipelineForText2Image,
)
import torch
from io import BytesIO
import time
import logging
import asyncio
from pydantic import BaseModel
import uuid

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

# Load model and assign it to available GPUs
num_gpus = 1  # torch.cuda.device_count()
models = [
    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(f"cuda:{i}")
    for i in range(num_gpus)
]

# Locks for managing concurrent access to GPUs
gpu_locks = [asyncio.Lock() for _ in range(num_gpus)]

# Counter for round-robin GPU selection
current_gpu = 0

logger.info(f"{num_gpus} GPU(s) initialized")


# Define a Pydantic model for the request body
class GenerateImageInput(BaseModel):
    prompt: str


@app.post("/txt2img")
async def txt2img(input_data: GenerateImageInput):
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
                pipe = models[gpu_id]

                image = pipe(
                    prompt=input_data.prompt, num_inference_steps=1, guidance_scale=0.0
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


@app.post("/img2img")
async def img2img(input_data: GenerateImageInput, image: UploadFile):
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
                pipe = models[gpu_id]

                init_image = await image.read()

                image = pipe(
                    prompt=input_data.prompt,
                    image=init_image,
                    num_inference_steps=2,
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
