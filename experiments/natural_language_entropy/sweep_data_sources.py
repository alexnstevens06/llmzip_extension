#!/usr/bin/env python3
"""Sweep all models against all data_sources texts from the database.

Runs each of the 7 models against each of the 35 data source texts at window=100.
Results are saved to benchmarks.db via encode_story.py. Progress is printed live.

Usage:
    cd ~/Documents/code/research/LLMzip_from_scratch
    source .venv/bin/activate
    HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True python -u sweep_data_sources.py
"""

import os
import subprocess
import sqlite3
import signal
import time
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

DB_FILE = "benchmarks.db"
PYTHON_EXEC = ".venv/bin/python3"
WINDOW = 100


def get_env():
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    return env


def make_slug(name):
    return name.replace(" ", "_").replace(".", "")


def clean_gpu():
    """Kill stale GPU processes before each run."""
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


def get_data_source_files():
    """Get the list of data source text files from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT category, title, char_count
        FROM data_sources
        ORDER BY category, title
    """)
    rows = cursor.fetchall()
    conn.close()

    # Build file paths matching the slug pattern from populate_sources.py
    files = []
    for category, title, char_count in rows:
        slug = f"{category}_{title.lower().replace(' ', '_').replace('/', '_')[:40]}"
        filepath = os.path.join("source_text", f"{slug}.txt")
        if os.path.exists(filepath):
            files.append({
                "path": filepath,
                "category": category,
                "title": title,
                "chars": char_count
            })
        else:
            print(f"  WARNING: {filepath} not found, skipping")
    return files


def is_already_done(model_path, source_path, window):
    """Check if this combination already exists in the benchmarks DB."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bpc FROM benchmarks
            WHERE model_path = ? AND source_text_path = ? AND context_window = ?
        """, (model_path, source_path, window))
        row = cursor.fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def main():
    env = get_env()
    source_files = get_data_source_files()

    if not source_files:
        print("No data source files found. Run populate_sources.py first.")
        return

    total = len(MODELS) * len(source_files)
    done_count = 0
    skip_count = 0
    error_count = 0
    current = 0

    print(f"{'='*70}")
    print(
        f"DATA SOURCES SWEEP: {len(MODELS)} models x {len(source_files)} texts, window={WINDOW}")
    print(f"{'='*70}")
    print(f"Models: {', '.join(m['name'] for m in MODELS)}")
    print(
        f"Categories: {', '.join(sorted(set(f['category'] for f in source_files)))}")
    print(f"Total runs: {total}")
    print(f"{'='*70}\n")

    for model in MODELS:
        print(f"\n{'─'*60}")
        print(f"MODEL: {model['name']}")
        print(f"{'─'*60}")

        for src in source_files:
            current += 1

            # Skip if already done
            if is_already_done(model["path"], src["path"], WINDOW):
                skip_count += 1
                print(
                    f"  [{current}/{total}] SKIP (exists) | {src['category']}/{src['title'][:30]}")
                continue

            slug = make_slug(model["name"])
            src_slug = os.path.splitext(os.path.basename(src["path"]))[0]
            output = f"encoded_documents/{slug}_{src_slug}_w{WINDOW}.llmzip"

            print(
                f"  [{current}/{total}] {src['category']} | {src['title'][:35]} ({src['chars']} chars)")
            sys.stdout.flush()

            try:
                clean_gpu()
                cmd = [
                    PYTHON_EXEC, "-u", "encode_story.py",
                    "--model", model["path"],
                    "--input", src["path"],
                    "--output", output,
                    "--window-size", str(WINDOW)
                ] + model["args"]

                start = time.time()
                subprocess.run(cmd, env=env, check=True)
                elapsed = time.time() - start

                done_count += 1
                print(f"    ✓ Done in {elapsed:.1f}s")

            except subprocess.CalledProcessError as e:
                error_count += 1
                print(f"    ✗ ERROR: {e}")

            sys.stdout.flush()

    # Print summary
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"  Completed: {done_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {error_count}")
    print(f"  Total:     {total}")

    # Show results summary from DB
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.model_name, 
                   COUNT(*) as runs,
                   ROUND(AVG(b.bpc), 4) as avg_bpc,
                   ROUND(MIN(b.bpc), 4) as min_bpc,
                   ROUND(MAX(b.bpc), 4) as max_bpc
            FROM benchmarks b
            WHERE b.context_window = ?
              AND b.source_text_path LIKE 'source_text/%_%'
              AND b.source_text_path NOT IN (
                  'source_text/short_story.txt', 'source_text/coding.txt',
                  'source_text/education.txt', 'source_text/social_science.txt',
                  'source_text/sports.txt', 'source_text/wiki_math.txt',
                  'source_text/text8.txt', 'source_text/text8_1mb.txt'
              )
            GROUP BY b.model_name
            ORDER BY avg_bpc
        """, (WINDOW,))
        rows = cursor.fetchall()
        conn.close()

        if rows:
            print(f"\n{'─'*70}")
            print(
                f"{'Model':<20} {'Runs':>5} {'Avg BPC':>10} {'Min BPC':>10} {'Max BPC':>10}")
            print(f"{'─'*70}")
            for name, runs, avg, mn, mx in rows:
                print(f"{name:<20} {runs:>5} {avg:>10.4f} {mn:>10.4f} {mx:>10.4f}")
    except Exception as e:
        print(f"  Could not print summary: {e}")


if __name__ == "__main__":
    main()
