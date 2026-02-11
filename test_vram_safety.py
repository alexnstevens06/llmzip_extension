import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Override for Navi 22 (RX 6700 XT) support on ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

from llmzip_qwen import LLMzipQwen

MODELS = [
    {"name": "Qwen 2.5-3B", "path": "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"},
    {"name": "Llama 3.2-1B", "path": "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"},
    {"name": "Llama 3.2-3B", "path": "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"},
    {"name": "LiquidAI 1.2B", "path": "/home/alexn/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Base/snapshots/1e601c5c9d33bcc8da794c253243d6b258a4d38b"},
    {"name": "Gemma 3-1B", "path": "/home/alexn/.cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"}
]

DEVICE = 'cuda'
TEST_TEXT = "This is a test sentence that will be repeated many times to create a large context. " * 50

def test_vram_safety():
    print("Starting VRAM Safety Tests...")
    print("Threshold: 0.5 GB headroom required.")
    
    for model_info in MODELS:
        print(f"\n--- Testing Model: {model_info['name']} ---")
        try:
            # Clear before each model
            torch.cuda.empty_cache()
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_info['path'])
            # Use bfloat16 for Gemma to avoid instability, otherwise default
            dtype = torch.bfloat16 if "gemma" in model_info['name'].lower() else torch.float16
            
            print(f"Loading {model_info['name']} in {dtype}...")
            # Explicit device map for ROCm compatibility on single GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_info['path'],
                torch_dtype=dtype,
                device_map={"": "cuda:0"} # Explicitly map to GPU 0
            )
            
            # Initialize LLMzip with a large max_window to intentionally push VRAM
            # We want to see if the safety logic kicks in
            llmzip = LLMzipQwen(model, tokenizer, device=DEVICE, max_window=2000)
            
            # Temporary output file
            temp_output = "temp_safety_test.llmzip"
            
            print("Starting encoding (stress test)...")
            # We'll run a few steps to see if symptoms appear
            llmzip.encode(TEST_TEXT, temp_output)
            
            # Check VRAM during/after
            free, total = torch.cuda.mem_get_info()
            print(f"Final VRAM Free: {free/1e9:.3f} GB / {total/1e9:.3f} GB")
            
            if free < 0.5 * 1024**3:
                print(f"FAILED: VRAM headroom below 0.5 GB for {model_info['name']}")
            else:
                print(f"PASSED: VRAM headroom maintained for {model_info['name']}")
                
            # Cleanup - VERY IMPORTANT for ROCm
            del model
            del tokenizer
            del llmzip
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.synchronize()
            
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"ERROR testing {model_info['name']}: {e}")
            torch.cuda.empty_cache()
            import gc
            gc.collect()

if __name__ == "__main__":
    test_vram_safety()
