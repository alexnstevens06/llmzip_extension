#!/usr/bin/env python3
"""Plot data source benchmark results from data_source_benchmarks table.

Produces a grouped bar chart showing BPC for each model across all data source
categories and individual sources.

Usage:
    cd ~/Documents/code/research/LLMzip_from_scratch
    source .venv/bin/activate
    python plot_data_sources.py
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

DB_FILE = "benchmarks.db"
OUTPUT_DIR = "results"

# ── Colour palette (hand-picked for readability on dark + light) ────────────
MODEL_COLORS = {
    "Qwen/Qwen2.5-3B":           "#6366f1",
    "LiquidAI/LFM2.5-1.2B-Base": "#f59e0b",
    "google/gemma-3-1b-pt":      "#10b981",
    "google/gemma-3-4b-pt":      "#06b6d4",
    "Qwen/Qwen3-1.7B":          "#ef4444",
    "meta-llama/Llama-3.2-1B":   "#ec4899",
    "meta-llama/Llama-3.2-3B":   "#8b5cf6",
}

# Friendly display names for legends / labels
MODEL_DISPLAY = {
    "Qwen/Qwen2.5-3B":           "Qwen 2.5-3B",
    "LiquidAI/LFM2.5-1.2B-Base": "LiquidAI 1.2B",
    "google/gemma-3-1b-pt":      "Gemma 3-1B",
    "google/gemma-3-4b-pt":      "Gemma 3-4B",
    "Qwen/Qwen3-1.7B":          "Qwen 3-1.7B",
    "meta-llama/Llama-3.2-1B":   "Llama 3.2-1B",
    "meta-llama/Llama-3.2-3B":   "Llama 3.2-3B",
}

CATEGORY_ORDER = [
    "wikipedia", "coding_docs", "source_code", "social_science",
    "sports", "education", "ai_generated",
]

CATEGORY_LABELS = {
    "wikipedia":      "Wikipedia",
    "coding_docs":    "Coding Docs",
    "source_code":    "Source Code",
    "social_science": "Social Science",
    "sports":         "Sports",
    "education":      "Education",
    "ai_generated":   "AI Generated",
}


def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT
            dsb.model_name,
            ds.category,
            ds.title,
            dsb.bpc,
            dsb.compression_ratio,
            ds.char_count
        FROM data_source_benchmarks dsb
        JOIN data_sources ds ON dsb.data_source_id = ds.id
        ORDER BY ds.category, ds.title, dsb.model_name
    """, conn)
    conn.close()
    return df


def shorten_title(title, max_len=22):
    """Shorten long titles for axis labels."""
    if len(title) <= max_len:
        return title
    return title[:max_len - 1] + "…"


# ── Plot 1: Grouped bar chart – BPC by model per source ─────────────────────
def plot_bpc_per_source(df):
    """One bar per model for every individual data source, grouped by category."""
    models = sorted(df["model_name"].unique())
    sources = []
    for cat in CATEGORY_ORDER:
        cat_df = df[df["category"] == cat]
        for t in sorted(cat_df["title"].unique()):
            sources.append((cat, t))

    n_sources = len(sources)
    n_models = len(models)
    bar_w = 0.9 / n_models
    x = np.arange(n_sources)

    fig, ax = plt.subplots(figsize=(max(22, n_sources * 0.7), 9))

    for i, model in enumerate(models):
        vals = []
        for cat, title in sources:
            row = df[(df["model_name"] == model) & (
                df["category"] == cat) & (df["title"] == title)]
            vals.append(row["bpc"].values[0] if len(row) else 0)
        color = MODEL_COLORS.get(model, "#888888")
        display = MODEL_DISPLAY.get(model, model)
        ax.bar(x + i * bar_w, vals, bar_w, label=display,
               color=color, edgecolor="white", linewidth=0.3)

    # x-axis labels
    labels = [shorten_title(t) for _, t in sources]
    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)

    # Category separators & labels
    prev_cat = None
    cat_starts = {}
    for idx, (cat, _) in enumerate(sources):
        if cat != prev_cat:
            if prev_cat is not None:
                ax.axvline(x=idx - 0.5, color="#555555",
                           linewidth=0.8, linestyle="--", alpha=0.5)
            cat_starts[cat] = idx
            prev_cat = cat

    for cat, start_idx in cat_starts.items():
        # Count sources in this category
        count = sum(1 for c, _ in sources if c == cat)
        mid = start_idx + count / 2 - 0.5
        label = CATEGORY_LABELS.get(cat, cat)
        ax.text(mid, ax.get_ylim()[1] * 0.97, label, ha="center", va="top",
                fontsize=10, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.85))

    ax.set_ylabel("Bits Per Character (BPC)", fontsize=13, labelpad=10)
    ax.set_title("LLMzip Compression: BPC by Data Source (window=100)",
                 fontsize=16, fontweight="bold", pad=18)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.5, n_sources)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "data_source_bpc_per_source.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
    return out


# ── Plot 2: Category-level summary (avg BPC per category per model) ─────────
def plot_bpc_by_category(df):
    """Grouped bar chart: average BPC per category, one bar per model."""
    agg = df.groupby(["category", "model_name"])["bpc"].mean().reset_index()
    models = sorted(agg["model_name"].unique())
    cats = [c for c in CATEGORY_ORDER if c in agg["category"].values]

    n_cats = len(cats)
    n_models = len(models)
    bar_w = 0.85 / n_models
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, model in enumerate(models):
        vals = []
        for cat in cats:
            row = agg[(agg["model_name"] == model) & (agg["category"] == cat)]
            vals.append(row["bpc"].values[0] if len(row) else 0)
        color = MODEL_COLORS.get(model, "#888888")
        display = MODEL_DISPLAY.get(model, model)
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=display,
                      color=color, edgecolor="white", linewidth=0.4)

    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    ax.set_ylabel("Average BPC", fontsize=13, labelpad=10)
    ax.set_title("LLMzip Average BPC by Category (window=100)",
                 fontsize=16, fontweight="bold", pad=18)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "data_source_bpc_by_category.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
    return out


# ── Plot 3: Heatmap – model × source ────────────────────────────────────────
def plot_heatmap(df):
    """Heatmap of BPC values: rows=models, columns=sources (sorted by category)."""
    sources = []
    for cat in CATEGORY_ORDER:
        cat_df = df[df["category"] == cat]
        for t in sorted(cat_df["title"].unique()):
            sources.append((cat, t))

    models = sorted(df["model_name"].unique())
    matrix = np.zeros((len(models), len(sources)))
    for mi, model in enumerate(models):
        for si, (cat, title) in enumerate(sources):
            row = df[(df["model_name"] == model) & (
                df["category"] == cat) & (df["title"] == title)]
            matrix[mi, si] = row["bpc"].values[0] if len(row) else np.nan

    fig, ax = plt.subplots(
        figsize=(max(20, len(sources) * 0.6), max(6, len(models) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="BPC")

    ax.set_xticks(range(len(sources)))
    xlabels = [shorten_title(t, 18) for _, t in sources]
    ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=10)

    # Add value annotations
    for mi in range(len(models)):
        for si in range(len(sources)):
            v = matrix[mi, si]
            if not np.isnan(v):
                ax.text(si, mi, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanmedian(matrix) else "black")

    ax.set_title("BPC Heatmap: Models × Data Sources (window=100)",
                 fontsize=15, fontweight="bold", pad=14)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "data_source_bpc_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
    return out


# ── Plot 4: Summary table as image ──────────────────────────────────────────
def plot_summary_table(df):
    """Render model summary table (avg/min/max BPC) as a styled PNG."""
    rows = []
    for model in sorted(df["model_name"].unique()):
        mdf = df[df["model_name"] == model]
        display = MODEL_DISPLAY.get(model, model)
        rows.append((display, f"{mdf['bpc'].mean():.4f}",
                    f"{mdf['bpc'].min():.4f}", f"{mdf['bpc'].max():.4f}"))

    # Sort by avg BPC ascending
    rows.sort(key=lambda r: float(r[1]))

    col_labels = ["Model", "Avg BPC", "Min BPC", "Max BPC"]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.4))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4f46e5")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(rows) + 1):
        color = "#f5f3ff" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)
            table[i, j].set_edgecolor("#e0e0e0")

    # Highlight best (first row)
    for j in range(len(col_labels)):
        table[1, j].set_text_props(fontweight="bold")

    ax.set_title("Model Summary – Data Source Benchmarks (window=100)",
                 fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "data_source_model_summary.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)
    return out


# ── Plot 5: Category summary table as image ─────────────────────────────────
def plot_category_table(df):
    """Render category (document type) summary table as a styled PNG."""
    rows = []
    for cat in CATEGORY_ORDER:
        cdf = df[df["category"] == cat]
        if cdf.empty:
            continue
        label = CATEGORY_LABELS.get(cat, cat)
        rows.append((label, f"{cdf['bpc'].mean():.4f}",
                     f"{cdf['bpc'].min():.4f}", f"{cdf['bpc'].max():.4f}"))

    # Sort by avg BPC ascending
    rows.sort(key=lambda r: float(r[1]))

    col_labels = ["Document Type", "Avg BPC", "Min BPC", "Max BPC"]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.4))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#0e7490")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(rows) + 1):
        color = "#ecfeff" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)
            table[i, j].set_edgecolor("#e0e0e0")

    # Highlight best (first row)
    for j in range(len(col_labels)):
        table[1, j].set_text_props(fontweight="bold")

    ax.set_title("BPC by Document Type – Data Source Benchmarks (window=100)",
                 fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "data_source_category_summary.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    if df.empty:
        print("No data found in data_source_benchmarks. Run sweep_data_sources.py first.")
        return

    print(f"Loaded {len(df)} rows: {df['model_name'].nunique()} models, "
          f"{df['title'].nunique()} sources across {df['category'].nunique()} categories\n")

    plot_bpc_per_source(df)
    plot_bpc_by_category(df)
    plot_heatmap(df)
    plot_summary_table(df)
    plot_category_table(df)

    # Print summary table
    print(f"\n{'─'*70}")
    print(f"{'Model':<20} {'Avg BPC':>10} {'Min BPC':>10} {'Max BPC':>10}")
    print(f"{'─'*70}")
    for model in sorted(df["model_name"].unique()):
        mdf = df[df["model_name"] == model]
        display = MODEL_DISPLAY.get(model, model)
        print(
            f"{display:<20} {mdf['bpc'].mean():>10.4f} {mdf['bpc'].min():>10.4f} {mdf['bpc'].max():>10.4f}")


if __name__ == "__main__":
    main()
