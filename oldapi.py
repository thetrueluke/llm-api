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

