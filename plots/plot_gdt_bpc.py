"""
Plot multi-complexity GDT benchmark results as a grouped bar chart (BPC).
Matches the project's existing figure style (seaborn whitegrid, despine).
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.use('Agg')

RESULTS_FILE = "experiments/multi_complexity_results.json"
OUTPUT_DIR = "results/generative_reconstruction_compression"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gdt_bpc_comparison.png")


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    # Prepare Data
    names = []
    std_bpc = []
    gdt_rat_bpc = []
    gdt_mye_bpc = []

    for d in data:
        # Simplified name: "Binary Search (trivial)" -> "Binary Search\n(trivial)"
        name_parts = d["name"].replace(")", "").split("(")
        if len(name_parts) > 1:
            clean_name = f"{name_parts[0].strip()}\n({name_parts[1].strip()})"
        else:
            clean_name = name_parts[0]
        names.append(clean_name)

        chars = d["target_chars"]
        std_bpc.append(d["standard_bits"] / chars)
        gdt_rat_bpc.append(d["ctx_rat_bits"] / chars)
        gdt_mye_bpc.append(d["ctx_mye_bits"] / chars)

    # ── Plot setup ────────────────────────────────────────────────
    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors
    colors = {
        "Standard LLMzip": "#2196F3",   # Blue
        "GDT (Ratcliff)": "#FF9800",    # Orange
        "GDT (Myers)": "#9C27B0",       # Purple
    }

    # ── Grouped bar chart ────────────────────────────────────────
    x = np.arange(len(names))
    width = 0.25

    # Bars
    bars1 = ax.bar(x - width, std_bpc, width, label="Standard LLMzip",
                   color=colors["Standard LLMzip"], edgecolor="white", linewidth=0.5, zorder=3)

    bars2 = ax.bar(x, gdt_rat_bpc, width, label="GDT (Context-Aware Ratcliff)",
                   color=colors["GDT (Ratcliff)"], edgecolor="white", linewidth=0.5, zorder=3)

    bars3 = ax.bar(x + width, gdt_mye_bpc, width, label="GDT (Context-Aware Myers)",
                   color=colors["GDT (Myers)"], edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color="#333333")

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # Styling
    ax.set_ylabel(
        "Bits Per Character (BPC) [Lower is Better]", fontsize=12, labelpad=10)
    ax.set_title("Compression Efficiency: Standard LLMzip vs. GDT Variants",
                 fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11, loc="upper left",
              framealpha=0.95, fancybox=True, shadow=True)

    # Add some headroom for labels
    ax.set_ylim(0, max(gdt_mye_bpc + gdt_rat_bpc) * 1.15)

    # Remove top and right spines
    sns.despine(ax=ax)

    # ── Save ─────────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
