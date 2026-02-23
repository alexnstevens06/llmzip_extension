
from experiments.natural_language_entropy.llmzip_qwen import LLMzipQwen
import os
import sys
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import asyncio

# Override for Navi 22 (RX 6700 XT) support on ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Add core to path so we can import necessary modules if needed
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../../"))

app = FastAPI()

# Global variables for model and tokenizer
model = None
tokenizer = None
llmzip = None


class AnalyzeRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    global model, tokenizer, llmzip
    model_name = "Qwen/Qwen2.5-3B"
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        llmzip = LLMzipQwen(model, tokenizer, device=model.device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)


async def surprise_generator(text: str):
    """Generates chunks of (token, surprise_score)"""
    if not text:
        yield json.dumps({"token": "", "surprise": 0.0}) + "\n"
        return

    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)[0]
    tokens = input_ids

    # Initial setup
    past_kv = None

    # First token has no context, assign 0 surprise
    first_token_str = tokenizer.decode([tokens[0].item()])
    yield json.dumps({"token": first_token_str, "surprise": 0.0, "probability": 1.0}) + "\n"

    for i in range(1, len(tokens)):
        # Allow interruption or slight delay to not block event loop completely if processing is heavy
        # though usually model inference is blocking on GPU, but Python is GIL bound.
        # minimal sleep to allow other async tasks if any
        await asyncio.sleep(0)

        if past_kv is None:
            input_token = tokens[i-1].unsqueeze(0).unsqueeze(0)
            probs, past_kv = llmzip._get_probs(
                input_token, past_key_values=None)
        else:
            input_token = tokens[i-1].unsqueeze(0).unsqueeze(0)
            probs, past_kv = llmzip._get_probs(
                input_token, past_key_values=past_kv)

        token_id = tokens[i].item()
        p = probs[token_id]
        surprise = -np.log2(p + 1e-9)  # Avoid log(0)

        token_str = tokenizer.decode([token_id])

        data = {
            "token": token_str,
            "surprise": float(surprise),
            "probability": float(probs[token_id])
        }
        yield json.dumps(data) + "\n"


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    return StreamingResponse(surprise_generator(request.text), media_type="application/x-ndjson")

# Mount static files
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__),
          "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
