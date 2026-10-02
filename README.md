# llmzip_extension

Experiments on compressing text with a language model's next-token probabilities and arithmetic coding.

The arithmetic coder and encode/decode scripts are in `core/`. Scripts under `experiments/natural_language_entropy/` run compression benchmarks and a self-compression comparison. Result files and plots are in `data/` and `results/`.

## Running

Commands already written in those files:

- `experiments/natural_language_entropy/run_self_compression.py`: `../.venv/bin/python3 run_self_compression.py`
- `experiments/natural_language_entropy/generate_code.py`: `.venv/bin/python3 generate_code.py --model <path> --slug <name> --count 8`
- `experiments/natural_language_entropy/plot_self_compression.py`: `.venv/bin/python3 plot_self_compression.py`
- `core/encode_story.py` and `core/decode_story.py` take `--model`, `--input`, `--output`, `--window-size`, and `--bf16`.

`sweep_data_sources.py` and `plots/plot_data_sources.py` also have usage comments at the top of the file. `agents.md` names `qwen_generation.py`, `benchmark_tps.py`, `benchmark_kv.py`, and `arithmetic_encoding.py`, which are not in this repository.
