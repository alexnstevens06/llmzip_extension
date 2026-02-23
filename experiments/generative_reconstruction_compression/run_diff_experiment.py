import os
import sys
import torch
import numpy as np
import difflib
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "core"))
sys.path.append(os.path.join(PROJECT_ROOT, "experiments/natural_language_entropy"))

from utils.myers_diff import myers_diff
from llmzip_qwen import LLMzipQwen

# ROCm environment setup
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
CANDIDATE_PATH = "experiments/binary_search_candidate.py"
TARGET_PATH = "experiments/binary_search_target.py"
PROMPT = "Write a simple Python binary search function that takes a sorted list and a target value. Only output the code, no explanation."

def load_data():
    with open(CANDIDATE_PATH, "r") as f:
        candidate = f.read()
    with open(TARGET_PATH, "r") as f:
        target = f.read()
    return candidate, target

def get_diff_script(opcodes, candidate_ids, target_ids, tokenizer):
    """
    Convert opcodes (tag, i1, i2, j1, j2) into a human-readable/compressible script.
    """
    script = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            script.append(f"K{i2-i1}") # Keep
        elif tag == 'insert':
            tokens = target_ids[j1:j2]
            text = tokenizer.decode(tokens)
            script.append(f"I[{text}]") # Insert
        elif tag == 'delete':
            script.append(f"D{i2-i1}") # Delete
        elif tag == 'replace':
            tokens = target_ids[j1:j2]
            text = tokenizer.decode(tokens)
            script.append(f"R{i2-i1}[{text}]") # Replace
    return " ".join(script)

def calculate_context_aware_bits(ops, candidate_ids, target_ids, model, tokenizer):
    """
    Simulates encoding the target content using the 'Candidate' as a reference 
    and the reconstructed 'Target' prefix as context.
    
    KEEP tokens are batched into a single forward pass to build KV cache efficiently.
    Only INSERT/REPLACE tokens are encoded one-by-one (these are the actual bits transmitted).
    """
    total_bits = 0.0
    device = next(model.parameters()).device
    
    from llmzip_qwen import LLMzipQwen
    compressor = LLMzipQwen(model, tokenizer, max_window=100)
    
    reconstructed_ids = []
    
    # Metadata overhead per operation (simulating a dense binary protocol)
    # Tag (2 bits) + Length (variable, ~10 bits average)
    META_TAX = 12 
    
    print("  Calculating context-aware bits...")
    
    past_kv = None
    op_count = len(ops)
    
    for op_idx, (tag, i1, i2, j1, j2) in enumerate(ops):
        total_bits += META_TAX
        print(f"    Op {op_idx+1}/{op_count}: {tag} (recon len: {len(reconstructed_ids)})", end='\r')
        
        if tag == 'equal':
            # KEEP: Receiver copies from candidate. No content bits transmitted.
            # But we must update KV cache so the model knows this context.
            # BATCH: feed all kept tokens at once instead of one-by-one.
            tokens_to_add = candidate_ids[i1:i2]
            if tokens_to_add:
                batch = torch.tensor([tokens_to_add], device=device)
                _, past_kv = compressor._get_probs(batch, past_kv)
                reconstructed_ids.extend(tokens_to_add)
                
        elif tag == 'insert' or tag == 'replace':
            # INSERT/REPLACE: These tokens must be transmitted.
            # Encode each one-by-one using the model's prediction from context.
            tokens_to_encode = target_ids[j1:j2]
            
            for tidx, token in enumerate(tokens_to_encode):
                if not reconstructed_ids:
                    # No context yet, uniform distribution
                    total_bits += 16  # ~log2(vocab)
                else:
                    # Feed the last reconstructed token to get prediction
                    last_tok = torch.tensor([[reconstructed_ids[-1]]], device=device)
                    probs, past_kv = compressor._get_probs(last_tok, past_kv)
                    
                    # Shannon information content: -log2(p)
                    p = float(probs[token])
                    if p <= 0: p = 1e-10
                    bits = -np.log2(p)
                    total_bits += bits
                
                reconstructed_ids.append(token)
            
            # For DELETE in a replace: skip candidate tokens (no action needed)
        
        elif tag == 'delete':
            # DELETE: Receiver skips candidate tokens. No content bits.
            pass

    print()  # newline after progress
    # Cleanup
    del past_kv
    torch.cuda.empty_cache()
    
    return total_bits

def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    candidate_text, target_text = load_data()
    
    # Baseline: Tokenize target
    target_encoding = tokenizer(target_text, return_tensors="pt")
    target_ids = target_encoding.input_ids[0].tolist()
    
    candidate_encoding = tokenizer(candidate_text, return_tensors="pt")
    candidate_ids = candidate_encoding.input_ids[0].tolist()

    print(f"Candidate tokens: {len(candidate_ids)}, Target tokens: {len(target_ids)}")

    # --- Baseline 1: Direct Compression with Base Model ---
    print("\n[Baseline 1] Standard LLMzip Compression...")
    llmzip = LLMzipQwen(model, tokenizer, max_window=100)
    
    output_llmzip = "experiments/target.llmzip"
    llmzip.encode(target_text, output_llmzip)
    base_compressed_size = os.path.getsize(output_llmzip) * 8 # bits
    
    # Cleanup after baseline
    torch.cuda.empty_cache()
    
    # --- Diffing ---
    print("\nGenerating Diffs...")
    # Ratcliff/Obershelp
    matcher = difflib.SequenceMatcher(None, candidate_ids, target_ids)
    ratcliff_ops = matcher.get_opcodes()
    
    # Myers
    myers_ops = myers_diff(candidate_ids, target_ids)
    
    print(f"Ratcliff ops: {len(ratcliff_ops)}, Myers ops: {len(myers_ops)}")
    
    # Generate Scripts (for the "Naive GRC" baseline)
    ratcliff_script = get_diff_script(ratcliff_ops, candidate_ids, target_ids, tokenizer)
    myers_script = get_diff_script(myers_ops, candidate_ids, target_ids, tokenizer)

    # --- Naive GRC Compression (Naked Scripts) ---
    print("\n[Baseline 2] Naive GRC (Naked Diff Scripts)...")
    def get_compressed_bits(text):
        out_file = "experiments/temp_script.llmzip"
        llmzip.encode(text, out_file)
        return os.path.getsize(out_file) * 8

    prompt_bits = get_compressed_bits(PROMPT)
    naive_ratcliff_bits = prompt_bits + get_compressed_bits(ratcliff_script)
    naive_myers_bits = prompt_bits + get_compressed_bits(myers_script)
    
    # Cleanup before context-aware phase
    torch.cuda.empty_cache()
    
    # --- Context-Aware GRC ---
    print("\n[Phase 3] Context-Aware GRC (Pre-seeded context)...")
    
    print("  Running Ratcliff context-aware...")
    context_ratcliff_diff = calculate_context_aware_bits(ratcliff_ops, candidate_ids, target_ids, model, tokenizer)
    
    print("  Running Myers context-aware...")
    context_myers_diff = calculate_context_aware_bits(myers_ops, candidate_ids, target_ids, model, tokenizer)
    
    total_context_ratcliff = prompt_bits + context_ratcliff_diff
    total_context_myers = prompt_bits + context_myers_diff
    
    # Original Size
    original_size = len(target_text) * 8
    
    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"Original Bits:       {original_size}")
    print(f"Standard LLMzip:     {base_compressed_size} bits ({original_size/base_compressed_size:.2f}x)")
    print("-" * 40)
    print(f"Naive GRC (Rat):     {naive_ratcliff_bits} bits ({original_size/naive_ratcliff_bits:.2f}x)")
    print(f"Naive GRC (Myers):   {naive_myers_bits} bits ({original_size/naive_myers_bits:.2f}x)")
    print("-" * 40)
    print(f"Context-Aware (Rat): {total_context_ratcliff:.0f} bits ({original_size/total_context_ratcliff:.2f}x)")
    print(f"Context-Aware (Mye): {total_context_myers:.0f} bits ({original_size/total_context_myers:.2f}x)")
    print("="*60)

    # Save detailed data
    with open("experiments/results_v2.json", "w") as f:
        import json
        json.dump({
            "original_bits": int(original_size),
            "base_llmzip_bits": int(base_compressed_size),
            "naive_ratcliff_bits": int(naive_ratcliff_bits),
            "naive_myers_bits": int(naive_myers_bits),
            "context_ratcliff_bits": float(total_context_ratcliff),
            "context_myers_bits": float(total_context_myers),
            "prompt_bits": int(prompt_bits)
        }, f)

if __name__ == "__main__":
    main()
