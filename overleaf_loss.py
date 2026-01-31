import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- STYLE SETTINGS ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2.5
})

# --- COLORS ---
COLORS = {
    "FedAvg": "#7f7f7f",   # Gray
    "FedProx": "#1f77b4",  # Blue
    "MOON": "#2ca02c",     # Green
    "ASTRA": "#9467bd"     # Purple
}

WINDOW_SIZE = 5

def load_data():
    all_data = []
    # SAME FILE LIST AS BEFORE
    files = [
        ("ASTRA", "results_metrics/metrics_Scenario4_Hybrid_Seed10_new.csv"),
        ("ASTRA", "results_metrics/metrics_Scenario4_Hybrid_Seed42_new.csv"),
        ("ASTRA", "results_metrics/metrics_Scenario4_Hybrid_Seed999_new.csv"),
        ("MOON", "results_metrics/metrics_moon_seed10.csv"),
        ("MOON", "results_metrics/metrics_moon_seed42.csv"),
        ("MOON", "results_metrics/metrics_moon_seed999.csv"),
        ("FedProx", "results_metrics/metrics_Scenario2_FedProx_Seed10.csv"),
        ("FedProx", "results_metrics/metrics_Scenario2_FedProx.csv"),
        ("FedProx", "results_metrics/metrics_Scenario2_FedProx_Seed999.csv"),
        ("FedAvg", "results_metrics/metrics_Scenario1_FedAvg_Seed10.csv"),
        ("FedAvg", "results_metrics/metrics_Scenario1_FedAvg.csv"),
        ("FedAvg", "results_metrics/metrics_Scenario1_FedAvg_Seed999.csv"),
    ]

    print("--- Loading Loss Data ---")
    for method, filepath in files:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df.columns = [c.lower() for c in df.columns] 
                # TARGETING 'loss' COLUMN NOW
                if 'round' in df.columns and 'loss' in df.columns:
                    temp_df = df[['round', 'loss']].copy()
                    temp_df['method'] = method
                    all_data.append(temp_df)
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")

    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_loss(df):
    if df.empty: return
    plt.figure()
    
    for method in COLORS.keys():
        subset = df[df['method'] == method]
        if subset.empty: continue

        # Group by round to avg seeds
        grouped = subset.groupby('round')['loss'].agg(['mean', 'std']).reset_index()

        # Apply Smoothing
        grouped['smooth_mean'] = grouped['mean'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        grouped['smooth_std'] = grouped['std'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

        plt.plot(grouped['round'], grouped['smooth_mean'], label=method, color=COLORS[method], alpha=1.0)
        plt.fill_between(grouped['round'], 
                         grouped['smooth_mean'] - grouped['smooth_std'], 
                         grouped['smooth_mean'] + grouped['smooth_std'], 
                         color=COLORS[method], alpha=0.15)

    plt.xlabel('Communication Rounds')
    plt.ylabel('Training Loss')
    plt.title(f'Global Model Loss (Smoothed Window={WINDOW_SIZE})')
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 50)
    plt.tight_layout()
    
    filename = 'fig_loss_comparison.png'
    plt.savefig(filename, format='png', dpi=300)
    print(f"✅ Loss plot saved: {filename}")

if __name__ == "__main__":
    df = load_data()
    plot_loss(df)