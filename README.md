# Stable Diffusion FastAPI Multi-GPU Server

This project is a web server that provides a FastAPI-based interface for generating images using Stable Diffusion models. It supports both text-to-image (`/txt2img`) and image-to-image (`/img2img`) generation endpoints. The server is designed to distribute the workload across multiple GPUs in a round-robin fashion, allowing for efficient utilization of available hardware resources.

## Prerequisites

- Docker
- NVIDIA GPU with driver installed
- NVIDIA Container Toolkit

## Usage

### Docker Run

The container accepts an environment variable `MODEL_NAME` which will default to `stabilityai/sdxl-turbo`. You can swap out this with any other model that works with the Diffusion pipelines AutoPipelineForText2Image and AutoPipelineForImage2Image.

Ports and gpus are configured with standard docker flags.

To use the most recent stable image, pull the `latest` tag:

```bash
docker run -e MODEL_NAME=stabilityai/sdxl-turbo  -p 8000:8000 ghcr.io/jemeyer/stablediffusion-fastapi-multigpu:latest
```

This will start the server and make it accessible at <http://localhost:8000>.

#### GPU Configuration

If you have an NVIDIA GPU and want to use it with the UI, you can pass the --gpus flag to docker run:

- To use all available GPUs:

```bash
docker run --gpus all -e MODEL_NAME=stabilityai/sdxl-turbo -p 8000:8000 ghcr.io/jemeyer/stablediffusion-fastapi-multigpu:latest
```

- To use a specific number of GPUs:

```bash
docker run --gpus 2 -e MODEL_NAME=stabilityai/sdxl-turbo -p 8000:8000 ghcr.io/jemeyer/stablediffusion-fastapi-multigpu:latest
```

- To use a specific GPU by its device ID (e.g., GPU 2):

```bash
docker run --gpus -e MODEL_NAME=stabilityai/sdxl-turbo device=2 -p 8000:8000 ghcr.io/jemeyer/stablediffusion-fastapi-multigpu:latest
```

Note that you need to have the NVIDIA Container Toolkit installed on your host for GPU passthrough to work.

### Docker Compose

You can also use Stable Diffusion WebUI with Docker Compose. Here's an example docker-compose.yml file:

```yaml
services:
  stablediffusion-fastapi-multigpu:
    image: ghcr.io/jemeyer/stablediffusion-fastapi-multigpu:latest
    ports:
      - 8000:8000
    environment:
      - MODEL_NAME=stabilityai/sdxl-turbo
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

This configuration will start a container using the latest image and make it accessible at <http://localhost:8000>. It also configures the container to use 1 GPU.

To use a specific GPU, you can use the device_ids property instead of count:

```yaml
reservations:
  devices:
    - driver: nvidia
      device_ids: ["2"]
      capabilities: [gpu]
```

To use all available GPUs, set count to `all`.

Start the container with:

```bash
docker-compose up -d
```

### Endpoints

- `/txt2img`: Generate an image from a text prompt.
- `/img2img`: Generate an image based on an input image and a text prompt.

This server uses `-turbo` SD models, and as such no negPrompt is used.

## Development

1. Clone the repository:

   ```bash
   git clone https://github.com/jemeyer/stable-diffusion-fastapi-multigpu.git
   cd stable-diffusion-fastapi-multigpu
   ```

2. Set up a virtual environment using `pyenv`:

   ```bash
   pyenv virtualenv 3.10.13 stable-diffusion-fastapi-multigpu
   pyenv local stable-diffusion-fastapi-multigpu
   ```

3. Install the required dependencies:

   ```bash
   pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/
   pip install .
   ```

   Note: The `torch` package is explicitly installed from the PyTorch download URL to ensure compatibility with the CUDA version.

## Dockerization

The server can be dockerized for easy deployment and scalability. A Dockerfile is provided to build the Docker image.

To build the Docker image, run the following command:

```bash
docker build -t jemeyer/stablediffusion-fastapi-multigpu .
```

To run the Docker container, use the following command:

```bash
docker run -p 8000:8000 jemeyer/stablediffusion-fastapi-multigpu
```

The server will be accessible at `http://localhost:8000`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Stable Diffusion](https://stability.ai/stable-image)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
