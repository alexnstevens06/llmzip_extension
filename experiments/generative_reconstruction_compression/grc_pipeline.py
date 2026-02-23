"""
grc_pipeline.py
===============
Generative Reconstruction Compression (GRC) — correct implementation.

Flow:
  1. Generate raw LLM output from prompt + seed via OpenRouter
  2. Apply developer edits to produce the final program (simulated or manual)
  3. Diff(raw_output, edited_output) on token IDs
  4. Compress: LLMzip(prompt) + seed + entropy(opcodes) + ctx-aware(diff content)
  5. Reconstruct: re-generate from prompt+seed, apply diff → verify exact match
  6. Compare total GRC bits vs standard LLMzip of the final program

Usage:
  python grc_pipeline.py [--prompt "..."] [--model MODEL] [--seed N] [--manual]

  --manual: skip simulated edits, load edited version from data/grc/edited_output.py
"""

import os
import sys
import json
import struct
import pickle
import subprocess
import argparse
import difflib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from utils.myers_diff import myers_diff
except ImportError:
    from utils.myers_diff import myers_diff

try:
    from experiments.natural_language_entropy.llmzip_qwen import LLMzipQwen
except ImportError:
    sys.path.append(os.path.join(
        PROJECT_ROOT, "experiments/natural_language_entropy"))
    from llmzip_qwen import LLMzipQwen

try:
    from experiments.generative_reconstruction_compression.openrouter_client import OpenRouterClient
    from experiments.generative_reconstruction_compression.simulate_edits import simulate_developer_edits, describe_edits
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from openrouter_client import OpenRouterClient
    from simulate_edits import simulate_developer_edits, describe_edits

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

LOCAL_MODEL = "Qwen/Qwen2.5-0.5B"
GRC_DATA_DIR = os.path.join(PROJECT_ROOT, "data/grc")
os.makedirs(GRC_DATA_DIR, exist_ok=True)

DEFAULT_PROMPT = (
    "Write a Python binary search function. "
    "The function should take a sorted list and a target value and return the index, or -1 if not found."
)
DEFAULT_MODEL = "moonshotai/kimi-k2.5"


# ── Compression helpers ────────────────────────────────────────────────────────

def llmzip_compress(text: str, model, tokenizer, label: str) -> tuple[int, str]:
    """LLMzip-compress text. Returns (bits, path)."""
    path = os.path.join(GRC_DATA_DIR, f"{label}.llmzip")
    c = LLMzipQwen(model, tokenizer, max_window=100)
    c.encode(text, path)
    bits = os.path.getsize(path) * 8
    print(f"  [{label}] {len(text)} chars → {bits} bits ({bits/len(text):.3f} bpc)")
    return bits, path


def llmzip_decompress(path: str, model, tokenizer) -> str:
    """LLMzip-decompress from path. Returns text."""
    c = LLMzipQwen(model, tokenizer, max_window=100)
    return c.decode(path)


def opcode_entropy_bits(ops: list) -> float:
    """Entropy estimate for the opcode stream (tag + length)."""
    if not ops:
        return 0.0
    tag_counts = {}
    for tag, *_ in ops:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    n = len(ops)
    tag_bits = sum(-c/n * np.log2(c/n)
                   for c in tag_counts.values() if c > 0) * n
    lengths = []
    for tag, i1, i2, j1, j2 in ops:
        lengths.append(max(i2-i1, j2-j1) if tag == 'replace' else
                       (i2-i1 if tag in ('equal', 'delete') else j2-j1))
    mean_len = max(np.mean(lengths), 1.0)
    q = 1.0 / mean_len
    len_bits = sum(-np.log2(max((1-q)**max(l-1, 0)*q, 1e-10)) for l in lengths)
    total = tag_bits + len_bits
    print(f"  Opcodes: {n} ops → {total:.1f} bits "
          f"({tag_bits:.1f} tag + {len_bits:.1f} length)")
    return total


def context_aware_content_bits(ops, candidate_ids, target_ids, model, tokenizer, label="") -> float:
    """
    Entropy of inserted/replaced tokens conditioned on the reconstructed prefix.
    Equal ops update the KV cache (free). Delete ops are skipped.
    """
    device = next(model.parameters()).device
    compressor = LLMzipQwen(model, tokenizer, max_window=100)
    reconstructed_ids = []
    total_bits = 0.0
    past_kv = None
    n = len(ops)

    print(f"  [{label}] Context-aware content bits...")
    for idx, (tag, i1, i2, j1, j2) in enumerate(ops):
        print(f"    {idx+1}/{n}: {tag}", end='\r')
        if tag == 'equal':
            toks = candidate_ids[i1:i2]
            if toks:
                _, past_kv = compressor._get_probs(
                    torch.tensor([toks], device=device), past_kv)
                reconstructed_ids.extend(toks)
        elif tag == 'delete':
            pass
        elif tag in ('insert', 'replace'):
            for tok in target_ids[j1:j2]:
                if reconstructed_ids:
                    probs, past_kv = compressor._get_probs(
                        torch.tensor([[reconstructed_ids[-1]]], device=device), past_kv)
                    p = max(float(probs[tok]), 1e-10)
                    total_bits += -np.log2(p)
                else:
                    total_bits += 16
                reconstructed_ids.append(tok)
    print()

    ok = reconstructed_ids == target_ids
    status = "✓ match" if ok else f"✗ MISMATCH ({len(reconstructed_ids)} vs {len(target_ids)} tokens)"
    print(f"  [{label}] Reconstruction: {status}")

    del past_kv
    torch.cuda.empty_cache()
    return total_bits


# ── Reconstruction (decode path) ───────────────────────────────────────────────

def reconstruct_from_grc(raw_text_path, ops_path, content_ids_path, model, tokenizer) -> str:
    """Full decode: load stored raw output → apply diff → return final text.

    In a real deployment the receiver would deterministically re-generate the
    raw output from the same LLM + prompt + seed. For verification we reuse
    the stored raw_output.py to avoid extra API calls.
    """
    with open(raw_text_path) as f:
        raw_text = f.read()
    candidate_ids = tokenizer(
        raw_text, return_tensors="pt").input_ids[0].tolist()

    with open(ops_path, "rb") as f:
        ops = pickle.load(f)

    # Load inserted token IDs directly (lossless — avoids decode/re-encode issues)
    if content_ids_path and os.path.exists(content_ids_path):
        with open(content_ids_path, "rb") as f:
            inserted_ids = pickle.load(f)
    else:
        inserted_ids = []

    reconstructed_ids = []
    cursor = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            reconstructed_ids.extend(candidate_ids[i1:i2])
        elif tag == 'delete':
            pass
        elif tag in ('insert', 'replace'):
            n = j2 - j1
            reconstructed_ids.extend(inserted_ids[cursor:cursor+n])
            cursor += n

    return tokenizer.decode(reconstructed_ids)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRC Pipeline (correct model)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--diff", choices=["myers", "ratcliff", "both"], default="both")
    parser.add_argument("--manual", action="store_true",
                        help="Load edited version from data/grc/edited_output.py instead of simulating")
    args = parser.parse_args()

    print("=" * 60)
    print("GRC Pipeline — Correct Model")
    print("=" * 60)
    print(f"  Generator:  {args.model}")
    print(f"  Compressor: {LOCAL_MODEL}")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    # 1. Load local model
    print(f"\n[1/6] Loading local compression model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL, dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"      Device: {next(model.parameters()).device}")

    # 2. Generate raw LLM output
    print(f"\n[2/6] Generating raw LLM output via {args.model}...")
    client = OpenRouterClient(model=args.model)
    raw_text = client.generate(args.prompt, seed=args.seed, max_tokens=2048)
    if not raw_text:
        print("ERROR: Generation failed.")
        return

    raw_path = os.path.join(GRC_DATA_DIR, "raw_output.py")
    with open(raw_path, "w") as f:
        f.write(raw_text)
    print(f"      Raw output: {len(raw_text)} chars → saved to {raw_path}")
    print("-" * 40)
    print(raw_text[:400] + ("..." if len(raw_text) > 400 else ""))
    print("-" * 40)

    # 3. Apply developer edits
    if args.manual:
        edited_path = os.path.join(GRC_DATA_DIR, "edited_output.py")
        if not os.path.exists(edited_path):
            print(f"ERROR: --manual specified but {edited_path} not found.")
            return
        with open(edited_path) as f:
            edited_text = f.read()
        print(
            f"\n[3/6] Loaded manual edits from {edited_path} ({len(edited_text)} chars)")
    else:
        print(f"\n[3/6] Applying simulated developer edits...")
        edited_text = simulate_developer_edits(raw_text)
        edited_path = os.path.join(GRC_DATA_DIR, "edited_output.py")
        with open(edited_path, "w") as f:
            f.write(edited_text)
        changes = describe_edits(raw_text, edited_text)
        for c in changes:
            print(f"      - {c}")
        print(
            f"      Edited: {len(edited_text)} chars → saved to {edited_path}")

    # 4. Tokenize both
    raw_ids = tokenizer(raw_text, return_tensors="pt").input_ids[0].tolist()
    edited_ids = tokenizer(
        edited_text, return_tensors="pt").input_ids[0].tolist()
    print(f"\n      Tokens — raw: {len(raw_ids)}, edited: {len(edited_ids)}")

    # 5. Compute diffs
    print(f"\n[4/6] Computing diffs...")
    algos = {}
    if args.diff in ("myers", "both"):
        algos["myers"] = myers_diff(raw_ids, edited_ids)
        print(f"      Myers:    {len(algos['myers'])} ops")
    if args.diff in ("ratcliff", "both"):
        algos["ratcliff"] = list(difflib.SequenceMatcher(
            None, raw_ids, edited_ids).get_opcodes())
        print(f"      Ratcliff: {len(algos['ratcliff'])} ops")

    # 6. Compress all components
    print(f"\n[5/6] Compressing components...")

    # Prompt + seed
    prompt_bits, prompt_path = llmzip_compress(
        args.prompt, model, tokenizer, "prompt")
    seed_bits = 32
    print(f"  [seed] 32 bits (fixed)")

    # Store seed
    seed_path = os.path.join(GRC_DATA_DIR, "seed.bin")
    with open(seed_path, "wb") as f:
        f.write(struct.pack("<I", args.seed))

    # Raw LLM output — stored to disk; receiver is assumed to deterministically
    # re-generate it from the same model + prompt + seed (no extra API call here).
    # already written above
    raw_path = os.path.join(GRC_DATA_DIR, "raw_output.py")

    # Per-algo: opcodes + content
    results_by_algo = {}
    for algo_name, ops in algos.items():
        print(f"\n  --- {algo_name.upper()} ---")
        op_bits = opcode_entropy_bits(ops)

        # Collect inserted/replaced tokens as text for LLMzip
        inserted_ids = []
        for tag, i1, i2, j1, j2 in ops:
            if tag in ('insert', 'replace'):
                inserted_ids.extend(edited_ids[j1:j2])

        if inserted_ids:
            inserted_text = tokenizer.decode(inserted_ids)
            content_bits, content_path = llmzip_compress(
                inserted_text, model, tokenizer, f"content_{algo_name}")
            # Also store IDs directly for lossless round-trip
            content_ids_path = os.path.join(
                GRC_DATA_DIR, f"content_ids_{algo_name}.pkl")
            with open(content_ids_path, "wb") as f:
                pickle.dump(inserted_ids, f)
        else:
            content_bits, content_path, content_ids_path = 0, None, None
            print(
                f"  [content_{algo_name}] No inserted tokens (perfect match)")

        # Store opcodes
        ops_path = os.path.join(GRC_DATA_DIR, f"opcodes_{algo_name}.pkl")
        with open(ops_path, "wb") as f:
            pickle.dump(ops, f)

        # Context-aware content bits (entropy estimate)
        ctx_bits = context_aware_content_bits(
            ops, raw_ids, edited_ids, model, tokenizer, label=algo_name)

        total = prompt_bits + seed_bits + op_bits + ctx_bits
        results_by_algo[algo_name] = {
            "opcode_bits": op_bits,
            "content_bits_llmzip": content_bits,
            "content_bits_ctx": ctx_bits,
            "total": total,
            "ops_path": ops_path,
            "content_path": content_path,
            "content_ids_path": content_ids_path,
            "raw_path": raw_path,
        }

    # Standard LLMzip baseline on the edited (final) program
    print(f"\n[6/6] Standard LLMzip baseline on final program...")
    llmzip_bits, _ = llmzip_compress(
        edited_text, model, tokenizer, "llmzip_baseline")
    original_bits = len(edited_text) * 8

    # ── Results ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"  Original (raw bytes):  {original_bits} bits  ({len(edited_text)} chars)")
    print(
        f"  Standard LLMzip:       {llmzip_bits} bits  ({original_bits/llmzip_bits:.2f}x)")
    for algo_name, r in results_by_algo.items():
        ratio = original_bits / r["total"]
        print(f"  GRC-{algo_name.capitalize()}:")
        print(f"    Prompt (LLMzip):     {prompt_bits} bits")
        print(f"    Seed:                {seed_bits} bits")
        print(f"    Opcodes (entropy):   {r['opcode_bits']:.0f} bits")
        print(f"    Content (ctx-aware): {r['content_bits_ctx']:.0f} bits")
        print(
            f"    TOTAL:               {r['total']:.0f} bits  ({ratio:.2f}x)")
        print(f"    (Raw output assumed free — deterministically re-generated by receiver)")
    print("=" * 60)

    # ── Round-trip verification ────────────────────────────────────────────────
    print("\n[Verify] Round-trip reconstruction...")
    for algo_name, r in results_by_algo.items():
        reconstructed = reconstruct_from_grc(
            r["raw_path"], r["ops_path"], r["content_ids_path"],
            model, tokenizer)
        recon_ids = tokenizer(
            reconstructed, return_tensors="pt").input_ids[0].tolist()
        ok = recon_ids == edited_ids
        print(f"  {algo_name}: {'✓ PASS' if ok else '✗ FAIL'} "
              f"({len(recon_ids)} vs {len(edited_ids)} tokens)")

    # ── Save results JSON ──────────────────────────────────────────────────────
    results = {
        "generator_model": args.model,
        "compressor_model": LOCAL_MODEL,
        "prompt": args.prompt,
        "seed": args.seed,
        "original_bits": original_bits,
        "original_chars": len(edited_text),
        "standard_llmzip_bits": llmzip_bits,
        "breakdown": {
            "prompt_bits": prompt_bits,
            "seed_bits": seed_bits,
        },
    }
    for algo_name, r in results_by_algo.items():
        results[f"grc_{algo_name}_bits"] = r["total"]
        results["breakdown"][f"opcode_bits_{algo_name}"] = r["opcode_bits"]
        results["breakdown"][f"content_bits_{algo_name}"] = r["content_bits_ctx"]

    out_json = os.path.join(PROJECT_ROOT, "experiments/grc_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Auto-plot
    for script in ["plot_bit_breakdown.py", "plot_bpc.py"]:
        p = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(p):
            subprocess.run([sys.executable, p], check=False)


if __name__ == "__main__":
    main()
