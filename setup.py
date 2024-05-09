import platform
import subprocess
from setuptools import setup, find_packages

# Helper function to get the CUDA version
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version'])
        version_str = output.decode('utf-8').split(',')[2].strip()
        version_parts = version_str.split('.')
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1])
        return f'cu{major_version}{minor_version}'
    except (subprocess.CalledProcessError, IndexError):
        return None

# Get the CUDA version
cuda_version = get_cuda_version()

# Set the PyTorch version and URL based on the CUDA version
if cuda_version:
    torch_version = '2.1.0'
    torch_url = f'https://download.pytorch.org/whl/{cuda_version}'
else:
    torch_version = '2.1.0+cpu'
    torch_url = 'https://download.pytorch.org/whl/cpu'

# Set the PyTorch package name
if platform.system() == 'Linux':
    torch_package = f'torch=={torch_version}+{cuda_version}' if cuda_version else f'torch=={torch_version}'
else:
    torch_package = f'torch=={torch_version}'

setup(
    name='stablediffusion-fastapi-multigpu',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'diffusers',
        'transformers',
        'accelerate',
        'aiohttp',
        'python-multipart',
        'Pillow',
        'xformers',
        torch_package
    ],
)
