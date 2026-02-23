import sys
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-Coder-0.5B"
sys.path.append(os.path.join(os.path.dirname(__file__), '../../experiments/natural_language_entropy'))
from llmzip_qwen import LLMzipQwen

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
compressor = LLMzipQwen(model, tokenizer, max_window=100)

device = next(model.parameters()).device

with open("experiments/generative_reconstruction_compression/nbody_prompt.txt", "r") as f:
    prompt_text = f.read()
    
with open("experiments/generative_reconstruction_compression/nbody_candidate.py", "r") as f:
    candidate_text = f.read()

candidate_ids = tokenizer.encode(candidate_text, return_tensors='pt')[0].tolist()
prompt_tokens = torch.tensor(tokenizer.encode(prompt_text, return_tensors='pt')[0].tolist() + [tokenizer.eos_token_id], device=device)

past_kv = None
for i in range(1, len(prompt_tokens)):
    if past_kv is None:
        start_idx = max(0, i - compressor.max_window)
        context = prompt_tokens[start_idx:i].unsqueeze(0)
        probs, past_kv = compressor._get_probs(context)
    else:
        last_token = prompt_tokens[i - 1:i].unsqueeze(0)
        probs, past_kv = compressor._get_probs(last_token, past_kv)

print(f"Prompt KV Cache Context State Built")

kept_tokens = candidate_ids[0:10]
batch = torch.tensor([kept_tokens], device=device)
_, past_kv = compressor._get_probs(batch, past_kv)
reconstructed_ids = kept_tokens

tokens_to_add = [tokenizer.eos_token_id]

for tidx, token in enumerate(tokens_to_add):
    if not reconstructed_ids:
        print("Using uniform")
        past_kv = None
    else:
        last_tok = torch.tensor([[reconstructed_ids[-1]]], device=device)
        probs, past_kv = compressor._get_probs(last_tok, past_kv)
        
    print(f"Prob for EOS: {probs[tokenizer.eos_token_id]:0.5f}")
    reconstructed_ids.append(token)
