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

print(f"\nGenerating {max_new_tokens} tokens with per-token timing (GPU):\n")
print(f"{'Token #':<10} | {'Token':<15} | {'Time (s)':<10}")
print("-" * 40)

current_ids = inputs.input_ids
start_time = time.time()

with torch.no_grad():
    for i in range(max_new_tokens):
        token_start = time.time()
        
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
tps = max_new_tokens / duration

print("\n" + "="*30)
print("Benchmark Results")
print("="*30)
print(f"Total time: {duration:.2f} seconds")
print(f"Tokens generated: {max_new_tokens}")
print(f"Average TPS: {tps:.2f} TPS")
print("="*30)

generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
print(f"\nFinal Output:\n{generated_text}")
