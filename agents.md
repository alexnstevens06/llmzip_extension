# Project: LLMzip from Scratch

## Overview
This project focuses on research and experimentation with Large Language Models (LLMs), specifically exploring arithmetic encoding (compression), generative reconstruction, and performance benchmarking. It uses the **Qwen2.5-3B** model as a primary driver.

## Environment
- **OS**: Linux
- **Python**: Running in a virtual environment (`.venv`). YOU MUST RUN COMMANDS FROM THE VIRTUAL ENVIRONMENT.
- **Hardware Acceleration**: AMD ROCm is used. Note the environment variable `HSA_OVERRIDE_GFX_VERSION="10.3.0"` is often required for Navi 22 GPUs (RX 6700 XT).
- **Libraries**: `transformers`, `torch`, `accelerate`.

## Project Structure
The project is organized into core components and specific experimental tracks:

### `core/`
Contains shared logic and utilities.
- **`AC/arithmeticcoding.py`**: Reference implementation of arithmetic encoding.
- **`encode_story.py`**: Utility for compressing text files using arithmetic encoding.

### `experiments/`
Research is divided into two main tracks:

#### 1. Natural Language Entropy (`experiments/natural_language_entropy/`)
Focuses on measuring the intrinsic entropy of text and code using various models.
- **`run_self_compression.py`**: Orchestrator for the "Self-Affinity" experiment (Does a model compress its own output best?).
- **`llmzip_qwen.py`**: Implements standard token-by-token arithmetic compression.
- **`plot_self_compression.py`**: Generates visualizations for the self-compression results.
- **`run_all_benchmarks.py`**: General purpose benchmarking script.

#### 2. Generative Reconstruction Compression (`experiments/generative_reconstruction_compression/`)
Explores "Generative Transmission" where content is generated on the receiver side and the sender transmits the prompt, seed, and diff.
- **`generative_compression.py`**: Prototype for the Prompt + Seed + Diff workflow.


### `results/`
Contains output plots and data, mirrored by experiment track.
- `results/natural_language_entropy/`: Plots for self-compression and entropy benchmarks.
- `results/generative_reconstruction_compression/`: (Future) Results for GRC experiments.

## Key Experiments

### Self-Compression (Natural Language Entropy)
Tests the **Self-Affinity Hypothesis**: Does a base model compress its own fine-tuned (Instruct) outputs better than other models?
- **Generators**: Llama 3.2-3B-Instruct, Gemma 3-4B-IT, Qwen 2.5-3B-Instruct.
- **Compressors**: Corresponding base models + smaller variants.
- **Status**: Operational. Results are saved in `data/self_compression_results.json`.

### Generative Reconstruction (GRC)
Aims to minimize transmission size by sending `Size(Prompt) + Size(Seed) + Size(Diff)` instead of the raw file.
- **Concept**: Sender generates a "Candidate" using a specific Prompt and Seed. The Sender computes the basic file with the prompt and seed, adjusts the basic file to be satisfactory, and then computes the diff. The Receiver generates the same Candidate and applies the Diff to recreate the file.
- **Status**: Prototype phase. Diff encoding is currently estimated.

## Quick Start
1. **Activate Environment**: Ensure you are in the project root.
2. **Run Self-Compression**:
   ```bash
   .venv/bin/python3 experiments/natural_language_entropy/run_self_compression.py
   ```
3. **Run GRC Prototype**:
   ```bash
   .venv/bin/python3 experiments/generative_reconstruction_compression/generative_compression.py --help
   ```
