
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmzip_qwen import LLMzipQwen

# Override for Navi 22 (RX 6700 XT) support on ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import argparse

def main():
    parser = argparse.ArgumentParser(description="Encode text using LLMzip.")
    parser.add_argument("--model", type=str, default="/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", help="Path to the model")
    parser.add_argument("--input", type=str, default="short_story.txt", help="Input text file")
    parser.add_argument("--output", type=str, default="short_story.llmzip", help="Output compressed file")
    parser.add_argument("--window-size", type=int, default=50, help="Context window size (default: 50)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16")
    args = parser.parse_args()

    model_path = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if device == "cuda" else "auto"),
            device_map=device,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize LLMzip
    llmzip = LLMzipQwen(model, tokenizer, device=device, max_window=args.window_size)
    
    # Read story
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Ensure output directory exists
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Update default output filename if it doesn't have a path
    if os.path.dirname(args.output) == "":
        args.output = os.path.join(output_dir, args.output)

    output_file = args.output
    
    print(f"Compressing {input_file} to {output_file}...")
    start_time = time.time()
    
    llmzip.encode(text, output_file)
    
    end_time = time.time()
    print(f"Compression finished in {end_time - start_time:.2f} seconds.")

    # JSON Tracking
    import json
    from datetime import datetime

    json_file = "model_results.json"
    results = []
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Calculate metrics
    original_size = len(text)
    compressed_size = os.path.getsize(output_file)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    bpc = (compressed_size * 8) / original_size if original_size > 0 else 0
    
    # Extract model name from path
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_name = os.path.basename(model_path)
    if "models--" in model_path:
        try:
            parts = model_path.split("/")
            for part in parts:
                if part.startswith("models--"):
                    model_name = part.replace("models--", "").replace("--", "/")
                    break
        except:
            pass

    new_entry = {
        "model_name": model_name,
        "model_path": model_path,
        "context_window": args.window_size,
        "source_text_path": input_file,
        "timestamp": datetime.now().isoformat(),
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": round(compression_ratio, 4),
        "bits_per_character": round(bpc, 4),
        "execution_time_seconds": round(end_time - start_time, 2)
    }
    
    results.append(new_entry)
    
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save to SQLite DB
    import sqlite3
    db_file = "benchmarks.db"
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Override logic: Delete previous results for same model, window, and source
        cursor.execute('''
            DELETE FROM benchmarks 
            WHERE model_path = ? AND context_window = ? AND source_text_path = ?
        ''', (model_path, args.window_size, input_file))
        
        cursor.execute('''
            INSERT INTO benchmarks (
                model_name, model_path, context_window, source_text_path,
                timestamp, original_size, compressed_size, 
                compression_ratio, bpc, execution_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            model_path,
            args.window_size,
            input_file,
            new_entry['timestamp'],
            original_size,
            compressed_size,
            round(compression_ratio, 4),
            round(bpc, 4),
            round(end_time - start_time, 2)
        ))
        conn.commit()
        conn.close()
        print(f"Results saved to {db_file}")
    except Exception as e:
        print(f"Error saving to database: {e}")

    print(f"Results saved to {json_file}")

if __name__ == "__main__":
    main()
