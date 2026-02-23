"""Stacked bit breakdown: Standard LLMzip vs GRC-Myers vs GRC-Ratcliff."""
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
    original_bits = data["original_bits"]
    original_chars = data["original_chars"]

    prompt_bits = bd["prompt_bits"]
    seed_bits = bd["seed_bits"]
    op_myers = bd["opcode_bits_myers"]
    op_ratcliff = bd["opcode_bits_ratcliff"]
    content_myers = bd["content_bits_myers"]
    content_ratcliff = bd["content_bits_ratcliff"]
    llmzip_bits = data["standard_llmzip_bits"]

    methods = ["Standard LLMzip", "GRC (Myers)", "GRC (Ratcliff)"]
    x = np.arange(len(methods))
    w = 0.45

    # Stack layers: each row is [llmzip, myers, ratcliff]
    layers = [
        ("LLMzip Content",           [
         llmzip_bits, 0, 0],                          "#3498db"),
        ("Prompt (LLMzip)",          [
         0, prompt_bits, prompt_bits],                "#9b59b6"),
        ("Seed (32-bit)",            [0, seed_bits,
         seed_bits],                    "#95a5a6"),
        ("Opcodes (entropy)",        [
         0, op_myers, op_ratcliff],                   "#e67e22"),
        ("Diff Content (ctx-aware)",
         [0, content_myers, content_ratcliff],         "#e74c3c"),
    ]

    totals = [
        llmzip_bits,
        prompt_bits + seed_bits + op_myers + content_myers,
        prompt_bits + seed_bits + op_ratcliff + content_ratcliff,
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottoms = np.zeros(len(methods))
    for label, vals, color in layers:
        vals = np.array(vals)
        ax.bar(x, vals, w, bottom=bottoms, color=color, label=label,
               edgecolor="white", linewidth=0.8)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 8:
                ax.text(x[i], b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    # Total + ratio labels above each bar
    for i, (tot, ratio) in enumerate(zip(totals, [original_bits / t for t in totals])):
        ax.text(x[i], tot + 6, f"{tot:.0f} bits\n({ratio:.1f}x)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Original size as annotation, not a line crossing the whole chart
    ax.annotate(
        f"Original: {original_bits} bits (1.0x)",
        xy=(x[-1] + w / 2 + 0.05, original_bits),
        xytext=(x[-1] + w / 2 + 0.3, original_bits - max(totals) * 0.15),
        fontsize=8, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    )

    # Y-axis: only go up to just above the tallest bar, not original_bits
    ax.set_ylim(0, max(totals) * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Total Bits", fontsize=11)
    ax.set_title(
        f"Stacked Bit Breakdown\n"
        f"Generator: {data['generator_model']}  |  Compressor: {data['compressor_model']}",
        fontsize=10, pad=10
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    out = os.path.join(
        PROJECT_ROOT, "results/generative_reconstruction_compression/grc_bit_breakdown.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
