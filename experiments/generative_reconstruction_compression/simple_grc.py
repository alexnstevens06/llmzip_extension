from core import LLMzip
import os
import sys
import torch
import difflib
import numpy as np
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def get_ratcliff_similarity(a: str, b: str) -> float:
    """Calculates the Ratcliff-Obershelp similarity ratio."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    global model, tokenizer, llmzip
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        llmzip = LLMzip(model, tokenizer, device=model.device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    with open("experiments/generative_reconstruction_compression/test_text.txt", "r") as f:
        text = f.read()
        compressed_bytes = llmzip.encode(text)

    # Round-trip verification using the raw bits from memory
    decoded_text = llmzip.decode(compressed_bytes)
    print("\n--- DECODED TEXT ---")
    print(decoded_text)
    print("--------------------")


if __name__ == "__main__":
    main()
