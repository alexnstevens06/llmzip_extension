import os
import subprocess
import argparse
import signal
import time

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

SOURCE_DIR = "source_text"
IGNORE_FILES = {"text8.txt"}
PYTHON_EXEC = ".venv/bin/python3"


def get_env():
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    return env


def make_slug(name):
    return name.replace(" ", "_").replace(".", "")


def clean_gpu():
    """Kill any stale processes holding the GPU and wait for VRAM to free."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["fuser", "/dev/kfd"],
            capture_output=True, text=True, timeout=5
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


def run_encode(model, input_path, output_file, window, env):
    clean_gpu()
    cmd = [
        PYTHON_EXEC, "-u", "encode_story.py",
        "--model", model["path"],
        "--input", input_path,
        "--output", output_file,
        "--window-size", str(window)
    ] + model["args"]
    subprocess.run(cmd, env=env, check=True)


def run_texts(args):
    """Run all models against each text file at a fixed window size."""
    env = get_env()
    window = args.window

    txt_files = sorted([
        f for f in os.listdir(SOURCE_DIR)
        if f.endswith(".txt") and f not in IGNORE_FILES
    ])

    total = len(MODELS) * len(txt_files)
    current = 0

    print(
        f"=== Text Benchmark: {len(MODELS)} models x {len(txt_files)} texts, window={window} ===")
    print(f"Texts: {txt_files}\n")

    for model in MODELS:
        for txt_file in txt_files:
            current += 1
            input_path = os.path.join(SOURCE_DIR, txt_file)
            basename = os.path.splitext(txt_file)[0]
            slug = make_slug(model["name"])
            output = f"encoded_documents/{slug}_{basename}_w{window}.llmzip"

            print(
                f"[{current}/{total}] {model['name']} | {txt_file} | window {window}")
            try:
                run_encode(model, input_path, output, window, env)
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: {e}")
            print()

    print("TEXT BENCHMARK COMPLETE")


def run_sweep(args):
    """Run all models across a range of window sizes on a single text file."""
    env = get_env()
    input_file = args.input
    windows = list(range(args.start, args.end + 1, args.step))

    total = len(MODELS) * len(windows)
    current = 0

    print(
        f"=== Window Sweep: {len(MODELS)} models x {len(windows)} windows on {input_file} ===")
    print(f"Windows: {windows}\n")

    for model in MODELS:
        for window in windows:
            current += 1
            slug = make_slug(model["name"])
            output = f"encoded_documents/sweep_{slug}_w{window}.llmzip"

            print(f"[{current}/{total}] {model['name']} | window {window}")
            try:
                run_encode(model, input_file, output, window, env)
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: {e}")
            print()

    print("WINDOW SWEEP COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="LLMzip benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # texts: all models x all texts at fixed window
    texts_parser = subparsers.add_parser(
        "texts", help="Run all models against each text file")
    texts_parser.add_argument(
        "--window", type=int, default=100, help="Context window size (default: 100)")

    # sweep: all models x window range on one text
    sweep_parser = subparsers.add_parser(
        "sweep", help="Sweep window sizes for all models on one text")
    sweep_parser.add_argument(
        "--input", type=str, default="source_text/short_story.txt", help="Input text file")
    sweep_parser.add_argument(
        "--start", type=int, default=5, help="Start window size (default: 5)")
    sweep_parser.add_argument(
        "--end", type=int, default=100, help="End window size (default: 100)")
    sweep_parser.add_argument(
        "--step", type=int, default=5, help="Window step (default: 5)")

    args = parser.parse_args()

    if args.command == "texts":
        run_texts(args)
    elif args.command == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()
