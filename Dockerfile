FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3

# Set the working directory
WORKDIR /app

# Copy the current project folder to /app
COPY . .

# Install the stablediffusion_fastapi_multigpu package
RUN pip install .

# Export the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "stablediffusion_fastapi_multigpu.app:app", "--host", "0.0.0.0", "--port", "8000"]
