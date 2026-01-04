import os
import time
import logging
import traceback
from typing import Callable

# -----------------------
# Logging configuration
# -----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# -----------------------
# FastAPI
# -----------------------
from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("qwen-webapi")

logger.info("Initializing API")
app = FastAPI(title="Qwen2.5-7B WebAPI", version="1.0")

# -----------------------
# App & model globals
# -----------------------
logger.info("Importing transformers")
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        if (path == "/ask" or "/classify") and method == "POST":
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


@app.get("/classify")
def classify_get(question: str = Query(..., min_length=1)):
    logger.debug("GET /classify invoked")
    return classify(AskRequest(question=question))

@app.post("/classify")
def classify(body: AskRequest):
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    system_text = """
You are a strict JSON classifier. 
Your task: decide whether the user input is a *description* and, if so, classify its type.

Definitions:
- "Description": A text primarily conveying attributes, characteristics, properties, or details about a subject (e.g., product features, person bio, process steps, place details, event overview, error message explanation). It is not asking a question, not issuing an instruction, and not general chit-chat.
- Types (choose one):
  product   - describes a product/service (features, specs, benefits)
  process   - describes steps, workflows, procedures
  person    - biography, role, achievements, traits
  place     - location, facilities, environment, geography
  event     - occurrence, schedule, agenda, outcomes
  error     - error/issue description, stack trace, incident symptom
  other     - descriptive but doesn't fit above

Output requirements:
- Respond ONLY with JSON following this schema:
  {
    "is_description_type": boolean,
    "type": "product|process|person|place|event|error|other",
    "confidence": number,
    "rationale": "string",
    "key_signals": ["string", "string"]
  }
- Do NOT include any text outside JSON. No markdown, no commentary.

Edge cases:
- If the input is a single noun phrase or bullet list that still conveys attributes, treat it as description.
- If the input mixes question + description, prefer description if ~70% of tokens are descriptive.
- If it is command/instruction (imperatives), set is_description_type=false.

"""

    messages = [
        {"role": "system", "content": system_text},
        # Include the few-shot pairs BEFORE the real input if you want stronger guidance:
        {"role": "user", "content": "Lightweight trail-running shoes with breathable mesh, rock plate, 4mm drop, and Vibram outsole for wet terrain."},
        {"role": "assistant", "content": '''{
            "is_description_type": true,
            "type": "product",
            "confidence": 0.92,
            "rationale": "Lists product features and specifications.",
            "key_signals": ["feature list", "materials", "technical terms (drop, outsole)"]
        }'''},
        # ... add other examples ...
        {"role": "user", "content": question},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([prompt_text], return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 256,      # plenty for the JSON
        "temperature": 0.0,         # deterministic classification
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
    import re, json
    m = re.search(r'\{.*\}', raw, flags=re.S)
    if not m:
        raise HTTPException(status_code=422, detail="Model did not return valid JSON.")
    payload = json.loads(m.group(0))
    return payload