"""
Plot multi-complexity GDT benchmark results as a grouped bar chart.
Matches the project's existing figure style (seaborn whitegrid, despine, 300dpi).
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

RESULTS_FILE = "experiments/multi_complexity_results.json"
OUTPUT_DIR = "results/generative_reconstruction_compression"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gdt_multi_complexity.png")

def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    names = [d["name"] for d in data]
    short_names = [n.split("(")[0].strip() for n in names]
    complexity_labels = [n.split("(")[1].rstrip(")") if "(" in n else "" for n in names]

    std_ratios = [d["standard_ratio"] for d in data]
    naive_ratios = [d["naive_rat_ratio"] for d in data]
    ctx_ratios = [d["ctx_rat_ratio"] for d in data]

    # ── Plot setup ────────────────────────────────────────────────
    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1.2]})

    # Colors
    colors = {
        "Standard LLMzip": "#2196F3",
        "Naive GRC (Ratcliff)": "#FF9800",
        "Context-Aware GRC": "#4CAF50",
    }

    # ── Left: Grouped bar chart ──────────────────────────────────
    x = np.arange(len(names))
    width = 0.22

    bars1 = ax1.bar(x - width, std_ratios, width, label="Standard LLMzip",
                    color=colors["Standard LLMzip"], edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax1.bar(x, naive_ratios, width, label="Naive GRC (Ratcliff)",
                    color=colors["Naive GRC (Ratcliff)"], edgecolor="white", linewidth=0.5, zorder=3)
    bars3 = ax1.bar(x + width, ctx_ratios, width, label="Context-Aware GRC",
                    color=colors["Context-Aware GRC"], edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}x',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xlabel("Program", fontsize=13, labelpad=10)
    ax1.set_ylabel("Compression Ratio (higher = better)", fontsize=13, labelpad=10)
    ax1.set_title("Compression Ratio by Method & Program Complexity",
                  fontsize=15, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}\n({c})" for s, c in zip(short_names, complexity_labels)],
                        fontsize=10)
    ax1.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax1.set_ylim(0, max(std_ratios) * 1.25)
    sns.despine(ax=ax1)

    # ── Right: Gap % line chart ──────────────────────────────────
    gaps = [ctx / std * 100 for ctx, std in zip(ctx_ratios, std_ratios)]
    tokens = [d["target_tokens"] for d in data]

    ax2.plot(tokens, gaps, marker="o", linewidth=2.5, color="#E91E63",
             markersize=8, zorder=3)
    ax2.fill_between(tokens, gaps, alpha=0.15, color="#E91E63")

    for i, (tok, gap) in enumerate(zip(tokens, gaps)):
        ax2.annotate(f'{gap:.0f}%', xy=(tok, gap), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=10, fontweight="bold",
                     color="#E91E63")

    ax2.set_xlabel("Target Tokens", fontsize=13, labelpad=10)
    ax2.set_ylabel("Context-Aware GRC vs LLMzip (%)", fontsize=11, labelpad=10)
    ax2.set_title("Gap Convergence Trend",
                  fontsize=15, fontweight="bold", pad=15)
    ax2.axhline(100, linestyle="--", color="gray", alpha=0.5, label="Parity (100%)")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9, loc="upper left")
    sns.despine(ax=ax2)

    # ── Save ─────────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
