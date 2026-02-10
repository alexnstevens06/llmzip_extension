import torch
import time
import os
import sys

# Set environment variable for ROCm compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected by PyTorch.")
    sys.exit(1)

def test_tensor_ops():
    print("\n--- Testing Tensor Operations on GPU ---")
    try:
        device = torch.device("cuda")
        
        # Create tensors
        x = torch.randn(8192, 8192, device=device, dtype=torch.float16)
        y = torch.randn(8192, 8192, device=device, dtype=torch.float16)
        
        # Warmup
        torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"10x 8192*8192 Matmul (fp16) time: {end - start:.4f}s")
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
            torch_dtype=torch.float16, 
            device_map="cuda"
        )
        
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        print("Generating...")
        start = time.time()
        # Generate 50 tokens
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        end = time.time()
        
        duration = end - start
        tps = 50 / duration
        print(f"Generation time for 50 tokens: {duration:.4f}s")
        print(f"Approx TPS: {tps:.2f}")
        print(f"Output: {tokenizer.decode(outputs[0])}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Transformers test FAILED: {e}")

if __name__ == "__main__":
    test_tensor_ops()
    test_transformers()
