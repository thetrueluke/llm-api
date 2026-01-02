import os
import time
import logging
import traceback
from typing import Callable

from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------
# Logging configuration
# -----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("qwen-webapi")

# -----------------------
# App & model globals
# -----------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Qwen2.5-7B WebAPI", version="1.0")


# -----------------------
# Request logging middleware
# -----------------------
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "-"
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""
    body_preview = ""

    try:
        if path == "/ask" and method == "POST":
            body_bytes = await request.body()
            body_preview = body_bytes.decode("utf-8", errors="ignore")
            if len(body_preview) > 500:
                body_preview = body_preview[:500] + "...[truncated]"
    except Exception:
        body_preview = "<unreadable>"

    logger.info(f">>> {client_ip} {method} {path} {('?' + query) if query else ''}")
    if body_preview:
        logger.info(f"    body: {body_preview}")

    try:
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"<<< {client_ip} {method} {path} status={response.status_code} time_ms={elapsed_ms:.1f}")
        return response
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(f"xxx {client_ip} {method} {path} error={type(e).__name__} time_ms={elapsed_ms:.1f}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Unhandled server error: {type(e).__name__}"})


# -----------------------
# Model initialization
# -----------------------
try:
    logger.info(f"Initializing model: {MODEL_ID}")
    logger.info(f"Detected device: {DEVICE} (cuda_available={torch.cuda.is_available()})")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    t_tok = (time.perf_counter() - t0) * 1000
    logger.info(f"Tokenizer loaded in {t_tok:.1f} ms")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    logger.info(f"Selected dtype: {dtype}")

    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None
    )
    if DEVICE != "cuda":
        model.to(DEVICE)
    t_model = (time.perf_counter() - t1) * 1000
    t_total = (time.perf_counter() - t0) * 1000
    logger.info(f"Model loaded in {t_model:.1f} ms (total init {t_total:.1f} ms)")
except Exception as e:
    logger.error(f"Failed to load model {MODEL_ID}: {e}")
    logger.error(traceback.format_exc())
    raise RuntimeError(f"Failed to load model {MODEL_ID}: {e}")


# -----------------------
# Schemas & helpers
# -----------------------
class AskRequest(BaseModel):
    question: str


def build_prompt(user_question: str) -> str:
    return f"### Question:\n{user_question}\n\n### Answer:\n"


# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    logger.debug("Health check invoked")
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.get("/ask")
def ask_get(question: str = Query(..., min_length=1)):
    logger.debug("GET /ask invoked")
    return ask(AskRequest(question=question))


@app.post("/ask")
def ask(body: AskRequest):
    logger.debug("POST /ask invoked")
    question = (body.question or "").strip()
    if not question:
        logger.warning("Validation failed: empty question")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    prompt = build_prompt(question)
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    try:
        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[-1]
        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        gen_ms = (time.perf_counter() - t0) * 1000
        total_ids = output_ids.shape[-1]
        new_tokens = total_ids - prompt_len

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = full_text.split("### Answer:")[-1].strip()

        logger.info(
            f"Generation ok | prompt_tokens={prompt_len} "
            f"new_tokens={new_tokens} total_tokens={total_ids} time_ms={gen_ms:.1f}"
        )
        return {"question": question, "answer": answer, "model": MODEL_ID}
    except Exception as e:
        logger.error(f"Inference error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
