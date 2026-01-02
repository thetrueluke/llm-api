FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
       && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Up-to-date packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1) Install PyTorch CPU wheel from the official index (avoids source builds)
#    Known-good wheel line for Python 3.12.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# 2) Install tokenizers from wheel only (avoid Rust/PyO3 compiles)
#    0.21.4 ships manylinux abi3 wheels and satisfies common transformers constraints.
RUN pip install --no-cache-dir --only-binary=:all: tokenizers==0.21.4

# 3) Install the rest
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy app
COPY app.py /app/app.py

# Run settings
EXPOSE 8000
ENV MODEL_ID=Qwen/Qwen2.5-7B

# Start FastAPI on all interfaces, port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
