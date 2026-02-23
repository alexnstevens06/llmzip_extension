import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.llmzip_utils import LLMzip


def main():
    model_name = "Qwen/Qwen2.5-3B"
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    llmzip = LLMzip(model, tokenizer, device=str(model.device))

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)[0]

    print(f"Text: {text}")
    print("Computing surprise...")

    # We will simulate the loop in encode but just get probabilities
    # Initial setup
    past_kv = None
    surprise_values = []
    tokens = input_ids  # No EOS for this test

    # First token has no context, let's skip or assign 0 surprise
    # In LLMzip, they use uniform for first token.
    surprise_values.append((tokens[0].item(), 0.0))

    for i in range(1, len(tokens)):
        # Context is everything before i
        # To be efficient, we should use past_kv like LLMzipQwen.encode

        # input to model is the previous token if we have past_kv
        if past_kv is None:
            # This happens on first iteration (i=1) if we don't prepopulate,
            # OR if we flush.
            # For i=1, prev token is tokens[0]
            input_token = tokens[i-1].unsqueeze(0).unsqueeze(0)  # (1, 1)
            probs, past_kv = llmzip._get_probs(
                input_token, past_key_values=None)
        else:
            input_token = tokens[i-1].unsqueeze(0).unsqueeze(0)
            probs, past_kv = llmzip._get_probs(
                input_token, past_key_values=past_kv)

        # probs is numpy array of shape (vocab_size,)
        # We want prob of actual token tokens[i]
        token_id = tokens[i].item()
        p = probs[token_id]
        surprise = -np.log2(p + 1e-9)  # Avoid log(0)
        surprise_values.append((token_id, surprise))

    print("\nResults:")
    for tid, sur in surprise_values:
        tok = tokenizer.decode([tid])
        print(f"Token: '{tok}' \tSurprise: {sur:.4f}")


if __name__ == "__main__":
    main()
