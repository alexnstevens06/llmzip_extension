# Project: LLMzip from Scratch

## Overview
This project focuses on research and experimentation with Large Language Models (LLMs), specifically exploring arithmetic encoding (compression) and performance benchmarking. It uses the **Qwen2.5-3B** model.

## Environment
- **OS**: Linux
- **Python**: Running in a virtual environment (`.venv`). YOU MUST RUN COMMANDS FROM THE VIRTUAL ENVIRONMENT.
- **Hardware Acceleration**: AMD ROCm is used. Note the environment variable `HSA_OVERRIDE_GFX_VERSION="10.3.0"` is often required for Navi 22 GPUs (RX 6700 XT).
- **Libraries**: `transformers`, `torch`, `accelerate`.

## Model Information
- **Model**: Qwen2.5-3B
- **Location**: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B`
- **Snapshot Used**: `snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`

## Key Scripts

### 1. Text Generation
- **`qwen_generation.py`**: 
  - Loads the local Qwen model.
  - Generates text with a restriction of 5 new tokens.
  - Outputs the top-5 token probabilities for each step.
  - **Usage**: `python qwen_generation.py`

### 2. Benchmarking
- **`benchmark_tps.py`**: 
  - Measures the generation speed (Tokens Per Second) on the GPU.
- **`benchmark_kv.py`**: 
  - Compares generation performance with vs. without Key-Value (KV) caching.

### 3. Core Logic
- **`arithmetic_encoding.py`**: 
  - Contains experimental logic for arithmetic encoding, likely for the "LLMzip" compression aspect.

### 4. Utilities
- **`test_gpu.py` / `test_gpu_simple.py`**: 
  - Simple scripts to verify that PyTorch can access the GPU and perform basic tensor operations.

## Self-Compression Experiment (Expanded)
This experiment tests the **Self-Affinity Hypothesis**: Does a base model compress its own fine-tuned (Instruct) outputs better than other models?

### Experiment Design
- **Generators**: 
    - Llama 3.2-3B-Instruct
    - Gemma 3-4B-IT
    - Qwen 2.5-3B-Instruct (Added Feb 2026)
- **Domains**: 
    - **Coding**: Binary Search Tree, CSV Stats, LRU Cache, Sieve of Eratosthenes, HTTP Server.
    - **Natural Language**: Argumentative Essay, Short Story, Educational Explanation (Added Feb 2026).
- **Compressors (Base Models)**: 
    - Llama 3.2-3B, Gemma 3-4B, Qwen 2.5-3B, plus smaller variants (1B/1.7B).

### Execution Scripts
- **`run_self_compression.py`**: Orchestrator script. Downloads models, generates content, compresses files, and plots results.
    - **Usage**: `.venv/bin/python3 run_self_compression.py`
    - **Note**: Runs for ~1.5 hours. Best run in `tmux`.
- **`generate_code.py`**: Generates content using instruct models. Now supports mixed content (Code + NL) and saves as `.txt`.
- **`plot_self_compression.py`**: Visualizes compression ratios (BPC) and self-affinity deltas.

### Current Status (Feb 17, 2026)
- The benchmark is running in a detached `tmux` session named `research`.
- **Artifacts**: Results accumulate in `self_compression_results.json`. Plots saved in `results/`.
- **Fixes**: 
    - Resolved OOM during model download by using `huggingface-cli`.
    - Fixed file extension logic (`.py.txt` -> `.txt`) for non-code content.

## Quick Start
1. **Activate Environment**: Ensure you are in the project root and the virtual environment is verified/active.
2. **Run Generation**:
   ```bash
   python qwen_generation.py
   ```
3. **Run Benchmarks**:
   ```bash
   python benchmark_tps.py
   ```
