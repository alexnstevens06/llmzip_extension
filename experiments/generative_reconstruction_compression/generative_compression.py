#!/usr/bin/env python3
"""
Generative Reconstruction Compression (GRC) Prototype.

This script simulates a compression scheme where:
1. Sender generates a prompt P describing the target file T.
2. Sender & Receiver share a seed S.
3. Both generate candidate C = LLM(P, S).
4. Sender transmits P, S, and the diff D = Diff(T, C).

The total size is Size(P) + Size(S) + Size(D).
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../core"))
import torch
import difflib
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model Path (Qwen 2.5-3B)
MODEL_PATH = "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def generate_candidate(model, tokenizer, prompt, seed=42, max_new_tokens=512):
    """
    Generate a candidate reconstruction deterministically.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Deterministic generation: greedy decoding (do_sample=False)
    # or sampling with fixed seed if we want variety.
    # For now, let's use greedy for maximum stability.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None
        )

    # Only return the NEW tokens
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


def calculate_token_diff_cost(target_ids, candidate_ids):
    """
    Calculate the cost of encoding the diff between target and candidate token sequences.
    Simulates an edit script:
    - KEEP: Log2(avg_segment_len) bits
    - INSERT: estimated 16 bits per token (or model entropy)
    - DELETE: Log2(avg_segment_len) bits
    """
    matcher = difflib.SequenceMatcher(None, candidate_ids, target_ids)

    total_cost_bits = 0
    ops = matcher.get_opcodes()

    # Basic estimation block
    # Opcode format: tag, i1, i2, j1, j2
    # replace: candidate[i1:i2] should be replaced by target[j1:j2]
    # delete: candidate[i1:i2] should be deleted
    # insert: target[j1:j2] should be inserted
    # equal: candidate[i1:i2] == target[j1:j2]

    # Constants for simulation
    OP_CODE_BITS = 2  # 4 ops -> 2 bits
    LENGTH_BITS = 8  # 0-255 length encoding
    TOKEN_BITS = 16  # Average entropy for unpredicted tokens

    for tag, i1, i2, j1, j2 in ops:
        total_cost_bits += OP_CODE_BITS  # Encode the operation type

        if tag == 'equal':
            # Cost: Encode length of match
            # If matches are long, this is cheap.
            # Ideally log2(length), but let's say 8 bits for length
            total_cost_bits += LENGTH_BITS
            pass

        elif tag == 'replace':
            # Delete + Insert
            # Cost: Encode length to delete + raw tokens to insert
            # number of tokens to delete
            del_len = i2 - i1
            ins_len = j2 - j1
            total_cost_bits += LENGTH_BITS  # encode del length
            total_cost_bits += LENGTH_BITS  # encode ins length
            total_cost_bits += ins_len * TOKEN_BITS  # encode raw tokens

        elif tag == 'delete':
            # Cost: Encode length to delete
            total_cost_bits += LENGTH_BITS

        elif tag == 'insert':
            # Cost: Encode length + raw tokens
            ins_len = j2 - j1
            total_cost_bits += LENGTH_BITS
            total_cost_bits += ins_len * TOKEN_BITS

    return total_cost_bits


def estimate_llmzip_size(text):
    """
    Estimate standard LLMzip size (roughly).
    Assume average 12 bits per token for code? 
    Or just use gzip as a baseline for "standard compression".
    """
    # Quick proxy: use gzip size * 8
    import gzip
    return len(gzip.compress(text.encode('utf-8'))) * 8


def main():
    parser = argparse.ArgumentParser(
        description="Generative Reconstruction Compression")
    parser.add_argument("--prompt", type=str, help="Prompt to generate code")
    parser.add_argument("--target", type=str, help="Target file path")
    args = parser.parse_args()

    if not args.prompt or not args.target:
        print("Usage: ./generative_compression.py --prompt '...' --target file.py")
        sys.exit(1)

    print(f"Reading target file: {args.target}")
    with open(args.target, 'r') as f:
        target_text = f.read()

    model, tokenizer = load_model()

    # 1. Generate Candidate
    print(f"Generating candidate from prompt: '{args.prompt}'...")
    candidate_text, candidate_ids_pt = generate_candidate(
        model, tokenizer, args.prompt)

    print("-" * 40)
    print("CANDIDATE START")
    print(candidate_text[:200] + "...")
    print("CANDIDATE END")
    print("-" * 40)

    # 2. Tokenize Target
    target_ids_pt = tokenizer(target_text, return_tensors="pt").input_ids[0]

    # Convert to lists for difflib
    target_ids = target_ids_pt.tolist()
    candidate_ids = candidate_ids_pt.tolist()

    # 3. Calculate Diff Cost
    diff_bits = calculate_token_diff_cost(target_ids, candidate_ids)

    # 4. Calculate Prompt Cost
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    prompt_bits = len(prompt_ids) * 16  # estimate

    total_grc_bits = prompt_bits + diff_bits

    # 5. Baselines
    original_bits = len(target_text) * 8
    gzip_bits = estimate_llmzip_size(target_text)

    print("\nRESULTS:")
    print(f"Original Size: {original_bits} bits")
    print(f"Gzip Size:     {gzip_bits} bits")
    print(
        f"GRC Size:      {total_grc_bits} bits (Prompt: {prompt_bits} + Diff: {diff_bits})")
    print(f"GRC Ratio:     {original_bits / total_grc_bits:.2f}x")

    if total_grc_bits < gzip_bits:
        print("\nSUCCESS: GRC beat Gzip!")
    else:
        print("\nFAIL: GRC was larger than Gzip.")


if __name__ == "__main__":
    main()
