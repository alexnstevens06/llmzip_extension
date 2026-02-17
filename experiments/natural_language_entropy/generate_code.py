#!/usr/bin/env python3
"""Generate content using an instruct model for self-compression experiments.

Usage:
    .venv/bin/python3 generate_code.py --model <path> --slug <name> --count 8
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

PROMPTS = [
    # Coding prompts
    "Write a Python program that implements a binary search tree with insert, search, and in-order traversal methods. Include a main block that demonstrates the functionality.",
    "Write a Python program that reads a CSV file and computes statistics (mean, median, standard deviation) for each numeric column. Use only the standard library (csv and statistics modules).",
    "Write a Python program that implements a simple LRU cache using an OrderedDict. Include put, get, and display methods, and demonstrate it with a main block.",
    "Write a Python program that implements the Sieve of Eratosthenes to find all prime numbers up to a given limit. Include a function to check if a number is prime and a main block.",
    "Write a Python program that implements a basic HTTP server using the http.server module that serves JSON responses for GET and POST requests to different endpoints.",
    # Natural language prompts
    "Write a persuasive essay arguing that space exploration is a necessary investment for the future of humanity. Discuss technological spin-offs, inspiration, and long-term survival.",
    "Write a short story about a watchmaker who discovers a way to manipulate time, but realizes that every second he saves costs him a memory.",
    "Explain the concept of 'Plate Tectonics' to a high school student. Cover the structure of the earth, types of plate boundaries, and the resulting geological features.",
]


def generate_programs(model_path, slug, count, out_dir="data/source_text", bf16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else (
            torch.float16 if device == "cuda" else "auto"),
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    generated_files = []

    for i in range(min(count, len(PROMPTS))):
        prompt = PROMPTS[i]
        print(f"\n[{i+1}/{count}] Generating: {prompt[:60]}...")

        # Build chat messages for instruct model
        # Modified to be generic, not just for Python code
        messages = [
            {"role": "user", "content": prompt +
                "\n\nRespond with ONLY the requested content, no conversational filler or markdown fences."}
        ]

        # Try using chat template; fall back to raw prompt
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = f"User: {prompt}\n\nRespond with ONLY the requested content.\n\nAssistant:\n"

        input_ids = tokenizer.encode(
            input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (strip the prompt)
        generated_tokens = output_ids[0][input_ids.shape[1]:]
        content = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Strip markdown fences if present (generic, not just python)
        content = content.strip()
        if content.startswith("```"):
            # Find first newline to skip language identifier (e.g. ```python or ```text)
            newline_idx = content.find("\n")
            if newline_idx != -1:
                content = content[newline_idx+1:].strip()
            else:
                content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        # Save with generic .txt extension
        filename = f"generated_{slug}_{i+1}.txt"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  Saved: {filepath} ({len(content)} chars)")
        generated_files.append(filepath)

    # Clean up GPU memory
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate content with instruct model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to instruct model")
    parser.add_argument("--slug", type=str, required=True,
                        help="Short name for file naming")
    parser.add_argument("--count", type=int, default=8,
                        help="Number of items to generate")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--out-dir", type=str, default="data/source_text",
                        help="Output directory for generated files")
    args = parser.parse_args()

    files = generate_programs(args.model, args.slug,
                              args.count, args.out_dir, args.bf16)
    print(f"\nGenerated {len(files)} files:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
