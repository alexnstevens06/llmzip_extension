"""
Multi-Complexity GDT Benchmark (Robust Version)
Compares Standard LLMzip, Naive GRC, and Context-Aware GRC
across programs of varying complexity and length.

Designed to run unattended in a tmux session with:
  - try/except around each benchmark case
  - aggressive GPU memory cleanup between phases
  - results saved incrementally so partial runs are recoverable
  - post-analysis summary table at the end
"""
import os
import sys
import gc
import torch
import numpy as np
import difflib
import json
import time
import traceback

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments/natural_language_entropy"))

os.chdir(PROJECT_ROOT)  # so relative data paths work

# ── ROCm / HIP env ───────────────────────────────────────────────────
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.myers_diff import myers_diff
from llmzip_qwen import LLMzipQwen

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
RESULTS_FILE = "experiments/multi_complexity_results.json"
TEMP_FILE = "experiments/_temp_bench.llmzip"

TEST_CASES = [
    {
        "name": "Binary Search (trivial)",
        "candidate": "experiments/binary_search_candidate.py",
        "target": "experiments/binary_search_target.py",
        "prompt": "Write a simple Python binary search function that takes a sorted list and a target value. Only output the code, no explanation.",
    },
    {
        "name": "Skip List (medium)",
        "candidate": "experiments/skiplist_candidate.py",
        "target": "experiments/skiplist_target.py",
        "prompt": "Implement a skip list data structure in Python with insert, search, delete, and display methods. Only output the code.",
    },
    {
        "name": "N-Body Sim (complex)",
        "candidate": "experiments/nbody_candidate.py",
        "target": "experiments/nbody_target.py",
        "prompt": "Write an N-body gravitational simulation in Python. Include energy calculation and a simulation loop. Only output the code.",
    },
    {
        "name": "Inventory System (complex+long)",
        "candidate": "experiments/inventory_candidate.py",
        "target": "experiments/inventory_target.py",
        "prompt": "Write a warehouse inventory management system in Python with Product, Warehouse, and InventoryReport classes. Only output the code.",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────
def gpu_cleanup():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_diff_script(opcodes, candidate_ids, target_ids, tokenizer):
    """Convert opcodes to a diff script string."""
    parts = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            parts.append(f"K{i2-i1}")
        elif tag == "insert":
            parts.append(f"I[{tokenizer.decode(target_ids[j1:j2])}]")
        elif tag == "delete":
            parts.append(f"D{i2-i1}")
        elif tag == "replace":
            parts.append(f"R{i2-i1}[{tokenizer.decode(target_ids[j1:j2])}]")
    return " ".join(parts)


def compress_text_to_bits(text, llmzip):
    """Compress text with LLMzip and return size in bits."""
    llmzip.encode(text, TEMP_FILE)
    return os.path.getsize(TEMP_FILE) * 8


def context_aware_bits(ops, candidate_ids, target_ids, model, tokenizer):
    """
    Calculate bits for context-aware diff encoding.
    KEEP tokens batched, INSERT/REPLACE encoded token-by-token with LLM context.
    """
    total = 0.0
    device = next(model.parameters()).device
    compressor = LLMzipQwen(model, tokenizer, max_window=50)
    recon = []
    META = 12  # per-op overhead (tag + length)
    past_kv = None
    n_ops = len(ops)

    for idx, (tag, i1, i2, j1, j2) in enumerate(ops):
        total += META
        if idx % 10 == 0:
            print(f"      ctx op {idx+1}/{n_ops}", end="\r", flush=True)

        if tag == "equal":
            toks = candidate_ids[i1:i2]
            if toks:
                # Batch feed — single forward pass for all KEEP tokens
                inp = torch.tensor([toks], device=device)
                with torch.no_grad():
                    _, past_kv = compressor._get_probs(inp, past_kv)
                recon.extend(toks)

        elif tag in ("insert", "replace"):
            for token in target_ids[j1:j2]:
                if not recon:
                    total += 16  # uniform (no context)
                else:
                    inp = torch.tensor([[recon[-1]]], device=device)
                    with torch.no_grad():
                        probs, past_kv = compressor._get_probs(inp, past_kv)
                    p = max(float(probs[token]), 1e-10)
                    total += -np.log2(p)
                recon.append(token)
        # "delete" → 0 content bits

    print(f"      ctx done ({n_ops} ops, {len(recon)} recon tokens)    ")

    # cleanup
    del past_kv, compressor, recon
    gpu_cleanup()
    return total


# ── Main benchmark ────────────────────────────────────────────────────
def run_one(case, model, tokenizer, llmzip):
    """Run all methods on one test case.  Returns result dict."""
    name = case["name"]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    with open(case["candidate"]) as f:
        cand_text = f.read()
    with open(case["target"]) as f:
        tgt_text = f.read()

    cand_ids = tokenizer.encode(cand_text)
    tgt_ids = tokenizer.encode(tgt_text)
    orig_bits = len(tgt_text) * 8

    print(f"  Candidate: {len(cand_ids)} tok / {len(cand_text)} chars")
    print(f"  Target:    {len(tgt_ids)} tok / {len(tgt_text)} chars")

    # 1) Standard LLMzip ──────────────────────────────────
    print("  [1/4] Standard LLMzip …")
    llmzip.encode(tgt_text, TEMP_FILE)
    std_bits = os.path.getsize(TEMP_FILE) * 8
    gpu_cleanup()

    # 2) Diffs ─────────────────────────────────────────────
    print("  [2/4] Diffs …")
    matcher = difflib.SequenceMatcher(None, cand_ids, tgt_ids)
    rat_ops = matcher.get_opcodes()
    mye_ops = myers_diff(cand_ids, tgt_ids)
    rat_script = get_diff_script(rat_ops, cand_ids, tgt_ids, tokenizer)
    mye_script = get_diff_script(mye_ops, cand_ids, tgt_ids, tokenizer)
    print(f"    Ratcliff ops: {len(rat_ops)}, Myers ops: {len(mye_ops)}")

    # 3) Naive GRC ─────────────────────────────────────────
    print("  [3/4] Naive GRC …")
    prompt_bits = compress_text_to_bits(case["prompt"], llmzip)
    gpu_cleanup()
    naive_rat = prompt_bits + compress_text_to_bits(rat_script, llmzip)
    gpu_cleanup()
    naive_mye = prompt_bits + compress_text_to_bits(mye_script, llmzip)
    gpu_cleanup()

    # 4) Context-Aware GRC ─────────────────────────────────
    print("  [4/4] Context-Aware GRC …")
    print("    Ratcliff:")
    ctx_rat_diff = context_aware_bits(rat_ops, cand_ids, tgt_ids, model, tokenizer)
    ctx_rat = prompt_bits + ctx_rat_diff
    gpu_cleanup()

    print("    Myers:")
    ctx_mye_diff = context_aware_bits(mye_ops, cand_ids, tgt_ids, model, tokenizer)
    ctx_mye = prompt_bits + ctx_mye_diff
    gpu_cleanup()

    result = {
        "name": name,
        "target_tokens": len(tgt_ids),
        "target_chars": len(tgt_text),
        "original_bits": orig_bits,
        "standard_bits": int(std_bits),
        "naive_rat_bits": int(naive_rat),
        "naive_mye_bits": int(naive_mye),
        "ctx_rat_bits": float(ctx_rat),
        "ctx_mye_bits": float(ctx_mye),
        "prompt_bits": int(prompt_bits),
        "standard_ratio": round(orig_bits / std_bits, 2),
        "naive_rat_ratio": round(orig_bits / naive_rat, 2),
        "ctx_rat_ratio": round(orig_bits / ctx_rat, 2),
    }

    print(f"\n  ── {name} ──")
    print(f"  Standard LLMzip : {std_bits:>8} bits  ({result['standard_ratio']:>6.2f}x)")
    print(f"  Naive GRC (Rat) : {naive_rat:>8} bits  ({result['naive_rat_ratio']:>6.2f}x)")
    print(f"  Context GRC(Rat): {ctx_rat:>8.0f} bits  ({result['ctx_rat_ratio']:>6.2f}x)")
    return result


def post_analysis(results):
    """Print comparison table and analysis."""
    print("\n\n" + "=" * 85)
    print("  CROSS-COMPLEXITY COMPARISON")
    print("=" * 85)
    header = f"{'Program':<32} {'Tok':>5} {'Chars':>5} {'LLMzip':>9} {'NaiveGRC':>9} {'CtxGRC':>9} {'Gap':>7}"
    print(header)
    print("-" * 85)
    for r in results:
        gap_pct = (r["ctx_rat_ratio"] / r["standard_ratio"]) * 100
        print(
            f"  {r['name']:<30} "
            f"{r['target_tokens']:>5} "
            f"{r['target_chars']:>5} "
            f"{r['standard_ratio']:>8.2f}x "
            f"{r['naive_rat_ratio']:>8.2f}x "
            f"{r['ctx_rat_ratio']:>8.2f}x "
            f"{gap_pct:>5.1f}%"
        )
    print("=" * 85)
    print("  Gap = Context-Aware GRC as a % of Standard LLMzip performance.")
    print("  100% = equal performance, higher = GRC outperforms.")

    # Trend analysis
    if len(results) >= 2:
        std_ratios = [r["standard_ratio"] for r in results]
        ctx_ratios = [r["ctx_rat_ratio"] for r in results]
        print("\n  TREND ANALYSIS:")
        print(f"    LLMzip compression ratio range: {min(std_ratios):.2f}x – {max(std_ratios):.2f}x")
        print(f"    Ctx GRC compression ratio range: {min(ctx_ratios):.2f}x – {max(ctx_ratios):.2f}x")
        # As LLMzip ratio decreases (harder files), does GRC close the gap?
        gaps = [ctx / std for ctx, std in zip(ctx_ratios, std_ratios)]
        print(f"    Gap ratio range: {min(gaps)*100:.1f}% – {max(gaps)*100:.1f}%")
        if gaps[-1] > gaps[0]:
            print("    ✓ Context-Aware GRC closes the gap on harder programs!")
        else:
            print("    ✗ Standard LLMzip maintains its lead across complexity levels.")


def main():
    start = time.time()
    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    gpu_cleanup()

    llmzip = LLMzipQwen(model, tokenizer, max_window=50)
    results = []

    for case in TEST_CASES:
        try:
            r = run_one(case, model, tokenizer, llmzip)
            results.append(r)
            # Save incrementally
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"\n  !! FAILED: {case['name']}: {e}")
            traceback.print_exc()
            gpu_cleanup()
            continue

    # ── Post-analysis ─────────────────────────────────────
    post_analysis(results)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
