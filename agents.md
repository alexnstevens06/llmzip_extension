# Project: LLMzip from Scratch

## Overview
This project focuses on research and experimentation with Large Language Models (LLMs), specifically exploring arithmetic encoding (compression) and performance benchmarking. It uses the **Qwen2.5-3B** model.

## Environment
- **OS**: Linux
- **Python**: Running in a virtual environment (`.venv`).
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
