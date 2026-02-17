#!/usr/bin/env python3
"""Generate a grouped bar chart of BPC by document and model for window=100."""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    conn = sqlite3.connect('benchmarks.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT model_name, source_text_path, bpc 
        FROM benchmarks 
        WHERE context_window = 100 
        ORDER BY source_text_path, model_name
    ''')
    rows = cursor.fetchall()
    conn.close()

    # Organize data
    docs = []
    models = []
    data = {}  # (doc, model) -> bpc

    for model, path, bpc in rows:
        doc = os.path.splitext(os.path.basename(path))[0]
        if doc not in docs:
            docs.append(doc)
        if model not in models:
            models.append(model)
        data[(doc, model)] = bpc

    # Sort models by average BPC (best first)
    model_avg = {m: np.mean([data.get((d, m), 0) for d in docs]) for m in models}
    models.sort(key=lambda m: model_avg[m])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(docs))
    n_models = len(models)
    bar_width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        bpc_values = [data.get((doc, model), 0) for doc in docs]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, bpc_values, bar_width * 0.9, label=model, color=colors[i], edgecolor='white', linewidth=0.5)

        # Add value labels on each bar
        for bar, val in zip(bars, bpc_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Document', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bits per Character (BPC)', fontsize=12, fontweight='bold')
    ax.set_title('LLMzip Compression: BPC by Document and Model (Window = 100)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in docs], fontsize=10)
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, max(data.values()) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = 'results/bpc_by_document_w100.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
