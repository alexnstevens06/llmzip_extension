# Open Ends & Research Directions

This document tracks open problems, feature requests, and avenues for future research in the LLMzip project.

## Generative Reconstruction Compression (GRC)

### 1. Arithmetic Encoded Diffs (Critical)
**Problem**: The current `generative_compression.py` script only *estimates* the size of the diffs using fixed bit-width assumptions (e.g., 8 bits for length, 16 bits for tokens). It does not actually generate a compressed binary stream.
**Goal**: Implement a real Arithmetic Encoder for the diff stream.
- **Approach**: 
    - Model the probability of "Keep", "Insert", "Delete" operations based on the context.
    - If `Level=High` match, "Keep" should have very high probability (very low bit cost).
    - Use the `core.AC` library to write the actual bits.

### 2. Token-Level vs. Character-Level Diffs
**Problem**: `difflib` works on lists of standard tokenizer IDs. This is coarse.
- **Issue**: A single character change in a variable name might change the entire token ID, resulting in a full "Replace" operation rather than a small "Edit".
**Goal**: Investigate "sub-token" or character-level diffing for finer granularity, potentially using a byte-level model or a specialized small model for patching.

### 4. Shared Context / RAG
**Idea**: Can the Sender and Receiver share a massive static library (e.g., GitHub code, StackOverflow)? 
- **Goal**: The "Prompt" could have the natural language description of the file and have supporting context built out from the shared library, plus a small diff to customize it.

## Natural Language Entropy

### 1. Cross-Architecture Entropy
**Question**: Do Mamba/RNN models have different entropy characteristics compared to Transformers (Attention) for the same text?

### 2. "Surprise" Metrics
**Idea**: Correlate high-entropy (hard to compress) segments with "human surprise" or "bug density".
- If a line of code requires many bits to encode, is it more likely to be a bug?
