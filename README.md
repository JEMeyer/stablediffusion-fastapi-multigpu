use pyenv
create .venv
install the requirements (will be missing torch explicitly)
run pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
uvicorn app:app --host 0.0.0.0 --port 8000
