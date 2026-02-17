import os
import subprocess
import sys

MODELS = [
    {
        "name": "Qwen 2.5-3B",
        "path": "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        "args": []
    },
    {
        "name": "LiquidAI 1.2B",
        "path": "/home/alexn/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Base/snapshots/1e601c5c9d33bcc8da794c253243d6b258a4d38b",
        "args": []
    },
    {
        "name": "Gemma 3-1B",
        "path": "/home/alexn/.cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
        "args": ["--bf16"]
    },
    {
        "name": "Gemma 3-4B",
        "path": "/home/alexn/.cache/huggingface/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
        "args": ["--bf16"]
    },
    {
        "name": "Qwen 3-1.7B",
        "path": "/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "args": []
    },
    {
        "name": "Llama 3.2-1B",
        "path": "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "args": []
    },
    {
        "name": "Llama 3.2-3B",
        "path": "/home/alexn/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062",
        "args": []
    }
]

WINDOWS = list(range(5, 105, 5))
INPUT_FILE = "source_text/short_story.txt"


def run_sweep():
    python_exec = ".venv/bin/python3"
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

    total_runs = len(MODELS) * len(WINDOWS)
    current_run = 0

    for model in MODELS:
        for window in WINDOWS:
            current_run += 1
            print(
                f"[{current_run}/{total_runs}] Testing {model['name']} with window size {window}...")

            output_file = f"encoded_documents/sweep_{model['name'].replace(' ', '_').replace('.', '')}_w{window}.llmzip"

            cmd = [
                python_exec, "-u", "encode_story.py",
                "--model", model["path"],
                "--input", INPUT_FILE,
                "--output", output_file,
                "--window-size", str(window)
            ] + model["args"]

            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(
                    f"Error running {model['name']} with window {window}: {e}")
                # Continue with next combination
                continue


if __name__ == "__main__":
    run_sweep()
