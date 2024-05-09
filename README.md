use pyenv

create .venv

install the requirements (will be missing torch explicitly)

run pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
old requirements:
fastapi[all]==0.103.2
uvicorn==0.23.2
accelerate==0.23.0
diffusers==0.21.4
aiohttp==3.8.6
transformers==4.34.0
python-multipart==0.0.6
Pillow==10.0.1

xformers==0.0.23.dev639

uvicorn app:app --host 0.0.0.0 --port 8000
