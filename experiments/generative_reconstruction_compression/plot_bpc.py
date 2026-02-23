"""Bits-per-character comparison: Standard LLMzip vs GRC-Myers vs GRC-Ratcliff."""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))


def main():
    with open(os.path.join(PROJECT_ROOT, "experiments/grc_results.json")) as f:
        data = json.load(f)
    bd = data["breakdown"]
    original_chars = data["original_chars"]

    prompt_bits = bd["prompt_bits"]
    seed_bits = bd["seed_bits"]
    op_myers = bd["opcode_bits_myers"]
    op_ratcliff = bd["opcode_bits_ratcliff"]
    content_myers = bd["content_bits_myers"]
    content_ratcliff = bd["content_bits_ratcliff"]
    llmzip_bits = data["standard_llmzip_bits"]

    totals = [
        llmzip_bits,
        prompt_bits + seed_bits + op_myers + content_myers,
        prompt_bits + seed_bits + op_ratcliff + content_ratcliff,
    ]
    bpc = [t / original_chars for t in totals]
    methods = ["Standard LLMzip", "GRC (Myers)", "GRC (Ratcliff)"]
    colors = ["#3498db", "#e74c3c", "#e67e22"]
    x = np.arange(len(methods))
    w = 0.45

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, bpc, w, color=colors, edgecolor="white", linewidth=1.2)

    for bar, b in zip(bars, bpc):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{b:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Bits Per Character (BPC)", fontsize=11)
    ax.set_title(
        f"Bits Per Character\n"
        f"Generator: {data['generator_model']}  |  Compressor: {data['compressor_model']}\n"
        f"(Raw ASCII = 8 bpc for reference)",
        fontsize=10, pad=10
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(bpc) * 1.22)

    out = os.path.join(
        PROJECT_ROOT, "results/generative_reconstruction_compression/grc_bpc.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
