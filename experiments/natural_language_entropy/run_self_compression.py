#!/usr/bin/env python3
"""Self-compression experiment orchestrator.

Waits for instruct model downloads, generates code, compresses with all base
models, and produces plots + tables.

Usage:
    ../.venv/bin/python3 run_self_compression.py
"""

import json
import os
import signal
import subprocess
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../core"))

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

experiment_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(experiment_dir)  # Root of the repo
PYTHON = os.path.join(PROJECT, ".venv/bin/python3")

SOURCE_DIR = os.path.join(PROJECT, "data", "source_text")
ENCODED_DIR = os.path.join(PROJECT, "data", "encoded_documents")
RESULTS_FILE = os.path.join(PROJECT, "data", "self_compression_results.json")

# Base models used for compression (same as run_all_benchmarks.py)
BASE_MODELS = [
    {
        "name": "Qwen 2.5-3B",
        "path": "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        "args": [],
    },
    {
        "name": "LiquidAI 1.2B",
        "path": "/home/alexn/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Base/snapshots/1e601c5c9d33bcc8da794c253243d6b258a4d38b",
        "args": [],
    },
    {
        "name": "Gemma 3-1B",
        "path": "/home/alexn/.cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
        "args": ["--bf16"],
    },
    {
        "name": "Gemma 3-4B",
        "path": "/home/alexn/.cache/huggingface/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
        "args": ["--bf16"],
    },
    {
        "name": "Qwen 3-1.7B",
        "path": "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "args": [],
    },
    {
        "name": "Llama 3.2-1B",
        "path": "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "args": [],
    },
    {
        "name": "Llama 3.2-3B",
        "path": "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062",
        "args": [],
    },
]

# Instruct generators and their corresponding "self" base model
GENERATORS = [
    {
        "name": "Llama 3.2-3B-Instruct",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "slug": "llama32_3b",
        "self_base": "Llama 3.2-3B",
        "bf16": False,
    },
    {
        "name": "Gemma 3-4B-IT",
        "hf_id": "google/gemma-3-4b-it",
        "slug": "gemma3_4b",
        "self_base": "Gemma 3-4B",
        "bf16": True,
    },
    {
        "name": "Qwen 2.5-3B-Instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "slug": "qwen25_3b",
        "self_base": "Qwen 2.5-3B",
        "bf16": False,
    },
]

WINDOW = 50


def get_env():
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    return env


def make_slug(name):
    return name.replace(" ", "_").replace(".", "").replace("-", "")


def clean_gpu():
    """Kill stale GPU processes (same logic as run_all_benchmarks.py)."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["fuser", "/dev/kfd"],
            capture_output=True, text=True, timeout=5,
        )
        output = (result.stdout + " " + result.stderr).strip()
        pids = set()
        for token in output.split():
            token = token.strip().rstrip("m")
            if token.isdigit():
                pid = int(token)
                if pid != my_pid:
                    pids.add(pid)
        if pids:
            print(f"  Cleaning {len(pids)} stale GPU process(es): {pids}")
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(2)
    except Exception:
        pass


def find_model_snapshot(hf_id):
    """Return the local snapshot path for a HuggingFace model, or None."""
    slug = hf_id.replace("/", "--")
    base = os.path.join(os.path.expanduser(
        "~"), ".cache/huggingface/hub", f"models--{slug}")
    snap_dir = os.path.join(base, "snapshots")
    if not os.path.isdir(snap_dir):
        return None
    snaps = os.listdir(snap_dir)
    if not snaps:
        return None
    # Return the latest snapshot
    return os.path.join(snap_dir, sorted(snaps)[-1])


def is_model_ready(hf_id):
    """Check if a model is fully downloaded by verifying all weight shards exist and are accessible."""
    path = find_model_snapshot(hf_id)
    if path is None:
        return False

    files = os.listdir(path)
    has_config = "config.json" in files
    if not has_config:
        return False

    # Check if there's a sharded model (index file lists all required shards)
    index_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        try:
            with open(index_file, "r") as f:
                index = json.load(f)
            # Get unique shard filenames from the weight map
            required_shards = set(index.get("weight_map", {}).values())
            for shard in required_shards:
                shard_path = os.path.join(path, shard)
                # Check the file exists AND is accessible (not a broken symlink)
                if not os.path.isfile(shard_path):
                    return False
                # Also verify the file has non-zero size
                try:
                    if os.path.getsize(shard_path) == 0:
                        return False
                except OSError:
                    return False
            return True
        except (json.JSONDecodeError, IOError):
            return False

    # Single-file model (no index)
    single = os.path.join(path, "model.safetensors")
    if os.path.isfile(single) and os.path.getsize(single) > 0:
        return True

    # Check for .bin files as fallback
    has_bins = any(f.endswith(".bin") and os.path.getsize(os.path.join(path, f)) > 0
                   for f in files if f.startswith("pytorch_model"))
    return has_bins


def wait_for_model(hf_id, poll_interval=30, timeout=7200):
    """Wait until a model is fully downloaded, polling every poll_interval seconds."""
    print(f"\nWaiting for {hf_id} to finish downloading...")
    start = time.time()
    while time.time() - start < timeout:
        if is_model_ready(hf_id):
            path = find_model_snapshot(hf_id)
            print(f"  ✓ {hf_id} is ready at {path}")
            return path
        elapsed = int(time.time() - start)
        print(f"  Still waiting... ({elapsed}s elapsed)", end="\r")
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {hf_id} after {timeout}s")


def download_model_in_tmux(hf_id, session="research"):
    """Start downloading a model in the tmux session."""
    cmd = f"hf download {hf_id}"
    tmux_cmd = f"tmux send-keys -t {session} '{cmd}' Enter"
    print(f"\nStarting download of {hf_id} in tmux session '{session}'...")
    subprocess.run(tmux_cmd, shell=True, check=True)


def generate_code(model_path, slug, bf16=False):
    """Generate Python programs using an instruct model."""
    cmd = [
        PYTHON, "-u", os.path.join(experiment_dir, "generate_code.py"),
        "--model", model_path,
        "--slug", slug,
        "--count", "8",
        # Note: generate_code.py now defaults to writing to data/source_text
        # because we will update it to default to the correct location or respect cwd
    ]
    if bf16:
        cmd.append("--bf16")

    print(f"\nGenerating code with {slug}...")
    subprocess.run(cmd, cwd=PROJECT, env=get_env(), check=True)

    # Find generated files
    files = sorted([
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if f.startswith(f"generated_{slug}_") and f.endswith(".txt")
    ])
    print(f"  Generated {len(files)} files")
    return files


def compress_file(model, input_path, output_file, env):
    """Compress a single file with a single base model."""
    clean_gpu()
    cmd = [
        PYTHON, "-u", "core/encode_story.py",
        "--model", model["path"],
        "--input", input_path,
        "--output", output_file,
        "--window-size", str(WINDOW),
    ] + model["args"]
    subprocess.run(cmd, cwd=PROJECT, env=env, check=True)


def run_compression_benchmark(generated_files, generator_slug, generator_name):
    """Compress all generated files with all base models and collect results."""
    env = get_env()
    results = []
    total = len(BASE_MODELS) * len(generated_files)
    current = 0

    print(f"\n{'='*70}")
    print(f"Compressing {len(generated_files)} files from {generator_name}")
    print(f"with {len(BASE_MODELS)} base models (window={WINDOW})")
    print(f"{'='*70}\n")

    for model in BASE_MODELS:
        for gen_file in generated_files:
            current += 1
            basename = os.path.splitext(os.path.basename(gen_file))[0]
            model_slug = make_slug(model["name"])
            output = os.path.join(
                ENCODED_DIR,
                f"selfcomp_{model_slug}_{basename}_w{WINDOW}.llmzip",
            )
            os.makedirs(os.path.dirname(output), exist_ok=True)

            print(
                f"[{current}/{total}] {model['name']} | {os.path.basename(gen_file)}")

            try:
                compress_file(model, gen_file, output, env)

                # Read metrics
                with open(gen_file, "r") as f:
                    text = f.read()
                original_size = len(text)
                compressed_size = os.path.getsize(output)
                bpc = (compressed_size * 8) / \
                    original_size if original_size > 0 else 0
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

                result = {
                    "generator_model": generator_name,
                    "generator_slug": generator_slug,
                    "compressor_model": model["name"],
                    "compressor_path": model["path"],
                    "source_file": gen_file,
                    "context_window": WINDOW,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "bpc": round(bpc, 4),
                    "compression_ratio": round(compression_ratio, 4),
                }
                results.append(result)
                print(f"  BPC: {bpc:.4f} | Ratio: {compression_ratio:.4f}")

            except subprocess.CalledProcessError as e:
                print(f"  ERROR: {e}")
            print()

    return results


def save_results(results):
    """Append results to the JSON file."""
    existing = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    existing.extend(results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(
        f"Saved {len(results)} results to {RESULTS_FILE} (total: {len(existing)})")


def main():
    print("=" * 70)
    print("SELF-COMPRESSION EXPERIMENT")
    print("Testing: Do base models compress their instruct outputs better?")
    print("=" * 70)

    all_results = []

    for gen_idx, gen in enumerate(GENERATORS):
        print(f"\n{'#'*70}")
        print(f"# Phase {gen_idx+1}: {gen['name']}")
        print(f"{'#'*70}")

        # Step 1: Wait for or download the model
        if is_model_ready(gen["hf_id"]):
            model_path = find_model_snapshot(gen["hf_id"])
            print(f"Model already available: {model_path}")
        else:
            if gen_idx > 0:
                # For second generator, start the download in tmux
                download_model_in_tmux(gen["hf_id"])
            model_path = wait_for_model(gen["hf_id"])

        # Step 2: Generate code
        generated_files = generate_code(model_path, gen["slug"], gen["bf16"])

        if not generated_files:
            print(f"ERROR: No files generated for {gen['name']}, skipping.")
            continue

        # Step 3: Compress with all base models
        results = run_compression_benchmark(
            generated_files, gen["slug"], gen["name"]
        )
        all_results.extend(results)
        save_results(results)

        # Step 4: Plot after each phase
        print(f"\nGenerating plots after {gen['name']} phase...")
        try:
            subprocess.run(
                [PYTHON, os.path.join(experiment_dir, "plot_self_compression.py")],
                cwd=PROJECT, env=get_env(), check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Plotting error: {e}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Total results: {len(all_results)}")
    print(f"Results file: {RESULTS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
