from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import sys
import os

# Override for Navi 22 (RX 6700 XT) support on ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# ROCm uses 'cuda' as the device string
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model_path = "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16 if device == "cuda" else "auto", # Use fp16 on GPU
    device_map=device
)

seed_text = "print('Hello, "
inputs = tokenizer(seed_text, return_tensors="pt").to(device)
max_new_tokens = 100

def benchmark_no_cache(input_ids, num_tokens):
    print(f"\nRunning WITHOUT cache for {num_tokens} tokens (GPU):")
    print(f"{'Token #':<10} | {'Token':<15} | {'Time (s)':<10}")
    print("-" * 40)
    
    current_ids = input_ids.clone()
    start_time = time.time()
    
    for i in range(num_tokens):
        token_start = time.time()
        with torch.no_grad():
            outputs = model(current_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        token_end = time.time()
        token_time = token_end - token_start
        token_str = tokenizer.decode(next_token[0])
        print(f"{i+1:<10} | {repr(token_str):<15} | {token_time:<10.4f}")
        sys.stdout.flush()
            
    end_time = time.time()
    duration = end_time - start_time
    return duration, tokenizer.decode(current_ids[0], skip_special_tokens=True)

def benchmark_with_cache(input_ids, num_tokens):
    print(f"\nRunning WITH KV cache for {num_tokens} tokens (GPU):")
    print(f"{'Token #':<10} | {'Token':<15} | {'Time (s)':<10}")
    print("-" * 40)
    
    current_ids = input_ids.clone()
    past_key_values = None
    start_time = time.time()
    
    for i in range(num_tokens):
        token_start = time.time()
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(current_ids, use_cache=True)
            else:
                outputs = model(current_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        token_end = time.time()
        token_time = token_end - token_start
        token_str = tokenizer.decode(next_token[0])
        print(f"{i+1:<10} | {repr(token_str):<15} | {token_time:<10.4f}")
        sys.stdout.flush()
            
    end_time = time.time()
    duration = end_time - start_time
    return duration, tokenizer.decode(current_ids[0], skip_special_tokens=True)

# Run Benchmarks
duration_no_cache, text_no_cache = benchmark_no_cache(inputs.input_ids, max_new_tokens)
duration_with_cache, text_with_cache = benchmark_with_cache(inputs.input_ids, max_new_tokens)

tps_no_cache = max_new_tokens / duration_no_cache
tps_with_cache = max_new_tokens / duration_with_cache

print("\n" + "="*40)
print(f"{'Method':<20} | {'Total Time':<10} | {'Avg TPS':<10}")
print("-" * 40)
print(f"{'No Cache':<20} | {duration_no_cache:<10.2f} | {tps_no_cache:<10.2f}")
print(f"{'With KV Cache':<20} | {duration_with_cache:<10.2f} | {tps_with_cache:<10.2f}")
print("="*40)
