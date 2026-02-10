import torch
import time
import os
import sys

# Set environment variable for ROCm compatibility if needed
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected by PyTorch.")
    # sys.exit(1)

def test_tensor_ops():
    print("\n--- Testing Tensor Operations on GPU ---")
    try:
        device = torch.device("cuda")
        
        # Create tensors
        x = torch.randn(4096, 4096, device=device)
        y = torch.randn(4096, 4096, device=device)
        
        # Warmup
        torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"10x 4096*4096 Matmul time: {end - start:.4f}s")
        print("Tensor operations working.")
    except Exception as e:
        print(f"Tensor operations FAILED: {e}")

def test_transformers():
    print("\n--- Testing Transformers on GPU ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
        
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, # Use fp16 for fair GPU comparison
            device_map="cuda"
        )
        
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        print("Generating...")
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=20)
        end = time.time()
        
        print(f"Generation time for 20 tokens: {end - start:.4f}s")
        print(f"Output: {tokenizer.decode(outputs[0])}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Transformers test FAILED: {e}")

def test_vllm():
    print("\n--- Testing vLLM on GPU ---")
    try:
        from vllm import LLM, SamplingParams
        
        model_path = "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
        
        print("Initializing vLLM...")
        llm = LLM(model=model_path, enforce_eager=True) # Try without enforce_eager if this works
        
        print("Generating...")
        start = time.time()
        outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=20))
        end = time.time()
        
        print(f"Generation time for 20 tokens: {end - start:.4f}s")
        print(f"Output: {outputs[0].outputs[0].text}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"vLLM test FAILED: {e}")

if __name__ == "__main__":
    test_tensor_ops()
    test_transformers()
    # Uncomment to test vLLM separately if needed, preventing memory conflicts if transformers expects to hold GPU
    # It might fail if transformers doesn't release memory, so maybe run separately or clear cache.
    torch.cuda.empty_cache() 
    test_vllm()
