from setuptools import setup, find_packages

setup(
    name="stablediffusion-fastapi-multigpu",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "diffusers",
        "transformers",
        "accelerate",
        "aiohttp",
        "python-multipart",
        "Pillow",
        "xformers",
        "torch",
        "numpy",
    ],
)
