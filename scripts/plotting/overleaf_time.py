import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- STYLE ---
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

def load_time_data():
    all_data = []
    
    # LISTA DE ARQUIVOS
    files = [
        # ASTRA
        ("ASTRA", "results_metrics/times_Scenario4_Hybrid_Seed42_new.csvv"),
        ("ASTRA", "results_metrics/times_Scenario4_Hybrid_Seed10_new.csv"),      # Seed 42 (Geralmente sem sufixo)
        ("ASTRA", "results_metrics/times_Scenario4_Hybrid_Seed999_new.csv"),
        
        # MOON
        ("MOON", "results_metrics/moon_training_times_seed10.csv"),
        ("MOON", "results_metrics/moon_training_times_seed42.csv"),
        ("MOON", "results_metrics/moon_training_times_seed999.csv"),
        
        # FedProx
        ("FedProx", "results_metrics/times_Scenario2_FedProx_Seed10.csv"),
        ("FedProx", "results_metrics/times_Scenario2_FedProx_Seed999.csv"),
        ("FedProx", "results_metrics/times_Scenario2_FedProx.csv"),
        
        # FedAvg
        ("FedAvg", "results_metrics/times_Scenario1_FedAvg_Seed10.csv"),
        ("FedAvg", "results_metrics/times_Scenario1_FedAvg_Seed999.csv"),
        ("FedAvg", "results_metrics/times_Scenario1_FedAvg.csv")
    ]

    print("--- Loading Time Data ---")
    for method, filepath in files:
        if os.path.exists(filepath):
            try:
                # FIX: Read without header, then assign names manually
                # Column 0: Round, Column 1: Client, Column 2: Duration
                df = pd.read_csv(filepath, header=None, names=['round', 'client', 'duration'])
                
                # Sum the duration column to get total training time for this seed
                total_time = df['duration'].sum()
                
                # Optional: Filter outliers (like that 399s value if it's an error)
                # if total_time > 10000: continue 

                all_data.append({"Method": method, "Total Time (s)": total_time})
                print(f"✅ Loaded: {filepath} | Time: {total_time:.2f}s")
            
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")
        else:
            print(f"⚠️  File not found: {filepath}")

    if not all_data: 
        print("⛔ No data loaded. Check paths.")
        return pd.DataFrame()
    
    return pd.DataFrame(all_data)

def plot_time_bar(df):
    if df.empty: return

    plt.figure(figsize=(9, 6))
    
    # Calculate average total time per method across seeds
    summary = df.groupby("Method")["Total Time (s)"].mean().reset_index()
    
    # Plot Bar Chart
    ax = sns.barplot(x="Method", y="Total Time (s)", data=summary, 
                     palette=COLORS, order=COLORS.keys(), capsize=.1)
    
    # Add numbers on top of bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.0f s', padding=3)

    plt.ylabel('Total Training Time (seconds)')
    plt.xlabel('')
    plt.title('Computational Efficiency (Total Training Time)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filename = 'fig_time_comparison.png'
    plt.savefig(filename, format='png', dpi=300)
    print(f"\n✅ Plot saved successfully: {filename}")

if __name__ == "__main__":
    df = load_time_data()
    plot_time_bar(df)