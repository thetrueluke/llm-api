
# Qwen2.5-7B WebAPI (Docker-ready)

Two options:

1. **Local model (transformers)** – Runs Qwen2.5-7B inside the container.
2. **Hugging Face Inference API** – Lightweight client calling HF-hosted model.

## Prerequisites
- Docker 24+
- For local model: adequate CPU/RAM (or NVIDIA GPU); initial download of model weights.

---

## Option 1: Local model (CPU baseline)

```bash
# Build
docker build -t qwen-webapi:cpu -f Dockerfile .

# Run
docker run --rm -p 8000:8000 qwen-webapi:cpu

# Test
curl "http://localhost:8000/ask?question=What%20is%20the%20capital%20of%20Poland%3F"
```

### Notes
- Set a different model via `-e MODEL_ID=Qwen/Qwen2.5-3B`.
- For GPU: use an NVIDIA CUDA base and `--gpus all` (see docs or adapt Dockerfile).

---

## Option 2: Hugging Face Inference API (lighter)

```bash
# Build
docker build -t qwen-webapi:inference -f Dockerfile.inference .

# Run (provide your token)
docker run --rm -p 8000:8000 -e HUGGINGFACE_API_TOKEN=hf_XXXX qwen-webapi:inference

# Test
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question":"Explain transformers briefly."}'
```

---

## Endpoints
- `GET /health` – basic health.
- `GET /ask?question=...` – query param.
- `POST /ask` – JSON body `{ "question": "..." }`.

## Troubleshooting
- Slow on CPU? Reduce `max_new_tokens` or switch to a smaller model.
- Model download failures inside Docker? Add `git` and ensure internet access.
- GPU: install NVIDIA Container Toolkit on host and use CUDA-enabled PyTorch.
