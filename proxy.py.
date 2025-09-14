from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()

# Hugging Face API key is set as an environment variable on Render
HF_API_KEY = os.getenv("HF_API_KEY")

# Pick your Hugging Face LLM and moderation model
LLM_MODEL = "google/gemma-2-2b-it"       # can swap to any Hugging Face text model
MOD_MODEL = "meta-llama/LlamaGuard-7b"   # moderation model

HF_BASE = "https://api-inference.huggingface.co/models"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def check_safe(text: str) -> bool:
    """Return True if safe, False if sexual/unsafe."""
    resp = requests.post(
        f"{HF_BASE}/{MOD_MODEL}",
        headers=headers,
        json={"inputs": text},
        timeout=30
    )
    try:
        out = resp.json()
        # LlamaGuard returns moderation decisions as generated text
        if isinstance(out, dict) and "generated_text" in out:
            flagged = "sexual" in out["generated_text"].lower()
            return not flagged
        elif isinstance(out, list) and "generated_text" in out[0]:
            flagged = "sexual" in out[0]["generated_text"].lower()
            return not flagged
    except Exception:
        # Fail open = treat as safe if moderation fails
        return True
    return True


@app.post("/")
async def root(req: Request):
    body = await req.json()
    prompt = body.get("inputs") or body.get("prompt") or ""

    # Step 1: Check user input
    if not check_safe(prompt):
        return {"generated_text": "Sorry, I can’t discuss sexual content."}

    # Step 2: Forward safe prompt to Hugging Face LLM
    resp = requests.post(
        f"{HF_BASE}/{LLM_MODEL}",
        headers=headers,
        json={"inputs": prompt},
        timeout=60
    )
    data = resp.json()

    # Step 3: Check model output
    if isinstance(data, list) and "generated_text" in data[0]:
        text_out = data[0]["generated_text"]
        if not check_safe(text_out):
            return {"generated_text": "Sorry, I can’t generate that content."}
        return {"generated_text": text_out}

    return data
