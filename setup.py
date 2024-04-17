from setuptools import setup, find_packages

setup(
    name='stablediffusion-fastapi-multigpu',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'diffusers',
        'aiohttp',
        'transformers',
        'python-multipart',
        'Pillow',
        'xformers',
        'torch', # pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    ],
)
