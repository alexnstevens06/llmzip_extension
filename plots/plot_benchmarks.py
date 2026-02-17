import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_FILE = "benchmarks.db"
OUTPUT_PLOT = "results/bpc_sweep_plot.png"

def plot_results():
    if not os.path.exists(DB_FILE):
        print(f"Database {DB_FILE} not found.")
        return

    # Connect to DB and load data
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT model_name, context_window, bpc FROM benchmarks", conn)
    conn.close()

    if df.empty:
        print("No data found in database.")
        return

    # Clean up model names for the plot labels
    df['display_name'] = df['model_name'].apply(lambda x: x.split('/')[-1] if '/' in x else x)

    # Sort data for better plotting
    df = df.sort_values(by=["display_name", "context_window"])

    # Setup plot
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})
    
    # Custom color palette
    palette = sns.color_palette("husl", len(df['display_name'].unique()))
    
    # Create line plot
    plot = sns.lineplot(
        data=df, 
        x="context_window", 
        y="bpc", 
        hue="display_name", 
        marker="o",
        linewidth=2.5,
        palette=palette,
        legend=False  # Handled by annotations
    )

    # Annotate model names at the end of each line
    for i, model in enumerate(df['display_name'].unique()):
        model_data = df[df['display_name'] == model]
        last_row = model_data.iloc[-1]
        
        plt.text(
            last_row['context_window'] + 1, 
            last_row['bpc'], 
            f" {model}", 
            color=palette[i],
            va='center',
            fontweight='bold',
            fontsize=10
        )

    # Customize plot aesthetics
    plt.title("LLMzip Compression Efficiency: BPC vs. Context Window", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Context Window Size (Tokens)", fontsize=14, labelpad=10)
    plt.ylabel("Bits Per Character (BPC)", fontsize=14, labelpad=10)
    
    # Restrict axes appropriately
    plt.xlim(0, 115)  # Extra space for labels
    plt.ylim(0, max(df['bpc']) * 1.1)
    
    plt.xticks(range(0, 105, 5))
    plt.grid(True, alpha=0.3)
    
    # Remove top and right spines
    sns.despine()

    # Ensure results directory exists
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_results()

