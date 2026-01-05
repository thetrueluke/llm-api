
import os
import time
import logging
import traceback
import asyncio
import contextlib
import re
import json
from typing import Callable, Tuple, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("llm-webapi")

# -----------------------------------------------------------------------------
# Model/config constants (after torch import)
# -----------------------------------------------------------------------------
try:
    import torch
except Exception:
    # Provide a clear error early if torch isn't available
    raise RuntimeError("PyTorch must be installed to run this service.")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_classifier.txt")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
logger.info("Initializing API")
app = FastAPI(title="LLM WebAPI", version="1.0")


def _load_models_blocking(model_id: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model in a worker thread so it doesn't block the event loop."""
    t0 = time.perf_counter()
    logger.info(f"[startup] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"[startup] Loading model: {model_id} (dtype={dtype}, device={device})")

    # 'dtype' is the modern arg name (instead of deprecated torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    t_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[startup] Model+tokenizer ready in {t_ms:.1f} ms")
    return tokenizer, model


async def _background_model_loader(app: FastAPI) -> None:
    """Background task that loads the Transformers stack and flips readiness to True."""
    try:
        tok, mdl = await asyncio.to_thread(_load_models_blocking, MODEL_ID, DEVICE)
        # Publish to app.state (shared app-wide state)
        app.state.tokenizer = tok
        app.state.model = mdl
        app.state.ready = True
        logger.info("[startup] Readiness set to True")
    except Exception:
        app.state.ready = False
        app.state.startup_error = traceback.format_exc()
        logger.error("[startup] Model load failed:\n%s", app.state.startup_error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize state quickly (so the server becomes responsive immediately)
    app.state.ready: bool = False
    app.state.startup_error: Optional[str] = None
    app.state.tokenizer: Optional[AutoTokenizer] = None
    app.state.model: Optional[AutoModelForCausalLM] = None

    # Load system prompt once
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            app.state.system_text = f.read()
        logger.info(
            f"[startup] Loaded system prompt from {SYSTEM_PROMPT_PATH} ({len(app.state.system_text)} chars)"
        )
    except Exception as e:
        app.state.system_text = (
            "You are a strict JSON classifier. Output ONLY JSON with the required keys."
        )
        app.state.startup_error = f"Failed to read SYSTEM_PROMPT_PATH: {e}"
        logger.warning(f"[startup] Using fallback system prompt. Error: {e}")

    # Fire-and-forget background loading (non-blocking)
    app.state.model_task = asyncio.create_task(_background_model_loader(app))

    # Yield immediately: startup completes, app starts serving
    yield

    # Shutdown: cancel background task if still running
    task = getattr(app.state, "model_task", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app.router.lifespan_context = lifespan  # register lifespan


# -----------------------------------------------------------------------------
# Schemas & helpers
# -----------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


def build_prompt(user_question: str) -> str:
    return f"### Question:\n{user_question}\n\n### Answer:\n"


# -----------------------------------------------------------------------------
# Request logging middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "-"
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""
    body_preview = ""
    try:
        if path in ("/ask", "/classify") and method == "POST":
            body_bytes = await request.body()
            body_preview = body_bytes.decode("utf-8", errors="ignore")
            if len(body_preview) > 500:
                body_preview = body_preview[:500] + "...[truncated]"
    except Exception:
        body_preview = "<unreadable>"

    logger.info(f">>> {client_ip} {method} {path} {('?' + query) if query else ''}")
    if body_preview:
        logger.info(f" body: {body_preview}")

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


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Liveness + readiness snapshot. Always 200 so liveness checks pass.
    Clients can inspect 'ready'=='True' to decide if they can call /classify.
    """
    return {
        "ready": getattr(app.state, "ready", False),
        "model": MODEL_ID,
        "device": DEVICE,
        "error": getattr(app.state, "startup_error", None),
    }


@app.get("/classify")
def classify_get(question: str = Query(..., min_length=1)):
    logger.debug("GET /classify invoked")
    return classify(AskRequest(question=question))


@app.post("/classify")
def classify(body: AskRequest):
    # Ensure model is ready
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    tokenizer: AutoTokenizer = app.state.tokenizer
    model: AutoModelForCausalLM = app.state.model
    system_text: str = getattr(app.state, "system_text", "")

    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Build messages for chat template (few-shot included)
    messages = [
        {"role": "system", "content": system_text},
        # Few-shot guidance example
        {
            "role": "user",
            "content": "Lightweight trail-running shoes with breathable mesh, rock plate, 4mm drop, and Vibram outsole for wet terrain.",
        },
        {
            "role": "assistant",
            "content": """{
                "is_description_type": true,
                "type": "product",
                "confidence": 0.92,
                "rationale": "Lists product features and specifications.",
                "key_signals": ["feature list", "materials", "technical terms (drop, outsole)"]
            }""",
        },
        # Actual input
        {"role": "user", "content": question},
    ]

    # Prefer tokenizer.apply_chat_template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: simple system + user concatenation
        prompt_text = (system_text or "") + "\n" + build_prompt(question)

    inputs = tokenizer([prompt_text], return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 256,  # plenty for the JSON
        "temperature": 0.0,     # deterministic classification
        "top_p": 1.0,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only newly generated tokens
    new_token_ids = output_ids[0, inputs.input_ids.shape[-1]:]
    raw = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    # Harden: trim anything before/after JSON braces
    m = re.search(r'\{.*\}', raw, flags=re.S)
    if not m:
        raise HTTPException(status_code=422, detail="Model did not return valid JSON.")
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON from model: {e}")

    return payload
