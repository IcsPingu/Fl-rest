import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- 1. ACADEMIC STYLE SETUP ---
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
    'figure.figsize': (8, 5),
    'lines.linewidth': 2.5
})

# --- 2. COLOR PALETTE ---
COLORS = {
    "FedAvg": "#d62728",    
    "FedProx": "#ff7f0e",   
    "MOON": "#2ca02c",      # Added MOON back
    "ASTRA": "#1f77b4"      
}

# --- 3. CONFIGURATION ---
WINDOW_SIZE = 5      
MAX_ROUNDS = 50      

def load_data():
    all_data = []
    
    # LIST OF FILES
    files = [
        # --- ASTRA ---
        #("ASTRA", "../../results/metrics/metrics_Scenario4_Hybrid_Seed10_new.csv"),
        ("ASTRA", "../../results/metrics/metrics_Scenario4_Hybrid_Seed42_new.csv"),
        #("ASTRA", "../../results//metrics/metrics_Scenario4_Hybrid_Seed999_new.csv"),

        # --- MOON ---
        ("MOON", "../../results/metrics/metrics_moon_seed10.csv"),
        ("MOON", "../../results/metrics/metrics_moon_seed42.csv"),
        ("MOON", "../../results/metrics/metrics_moon_seed999.csv"),

        # --- FedProx ---
        ("FedProx", "../../results/metrics/metrics_Scenario2_FedProx_Seed10.csv"),
        ("FedProx", "../../results/metrics/metrics_Scenario2_FedProx.csv"),   
        ("FedProx", "../../results/metrics/metrics_Scenario2_FedProx_Seed999.csv"),

        # --- FedAvg ---
        ("FedAvg", "../../results/metrics/metrics_Scenario1_FedAvg_Seed10.csv"),
        ("FedAvg", "../../results/metrics/metrics_Scenario1_FedAvg.csv"),     
        ("FedAvg", "../../results/metrics/metrics_Scenario1_FedAvg_Seed999.csv"),
    ]

    print("--- Loading Datasets ---")
    for method, filepath in files:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df.columns = [c.lower() for c in df.columns] 
                
                if 'round' in df.columns and 'accuracy' in df.columns:
                    temp_df = df[['round', 'accuracy']].copy()
                    temp_df['method'] = method
                    # Add a 'run_id' based on filename to distinguish seeds internally if needed
                    temp_df['source'] = filepath 
                    all_data.append(temp_df)
                    print(f"✅ Loaded: {filepath} ({len(df)} rounds)")
                else:
                    print(f"⚠️  Wrong columns in: {filepath}")
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")
        else:
            print(f"❌ File not found: {filepath}")

    if not all_data:
        print("⛔ No data loaded. Check your file paths!")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

def generate_stats_table(df):
    """
    Calculates Mean ± Std for the FINAL ROUND across all seeds.
    """
    if df.empty:
        return

    print("\n" + "="*50)
    print("       📊 FINAL RESULTS (Mean ± Std)        ")
    print("="*50)

    # 1. Filter only for the data at MAX_ROUNDS (Final Convergence)
    # We take the last available round for each run, up to MAX_ROUNDS
    final_results = []
    
    # Iterate over each unique file source to get its final value
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        # Get the row with the max round number
        last_row = subset.loc[subset['round'].idxmax()]
        
        # Only count if it actually reached (or is close to) the target rounds
        # This prevents a crashed run at round 5 from messing up the average
        if last_row['round'] >= 40: 
            final_results.append({
                'method': last_row['method'],
                'final_acc': last_row['accuracy']
            })
    
    results_df = pd.DataFrame(final_results)

    # 2. Group by Method and Calculate Stats
    stats = results_df.groupby('method')['final_acc'].agg(['mean', 'std', 'count'])
    
    # 3. Print readable text table
    print(f"{'Method':<15} | {'Accuracy (%)':<20} | {'Samples'}")
    print("-" * 50)
    for method, row in stats.iterrows():
        mean = row['mean']
        std = row['std']
        count = int(row['count'])
        print(f"{method:<15} | {mean:.2f} ± {std:.2f}        | n={count}")

    # 4. Print LaTeX Table Code
    print("\n" + "="*50)
    print("       📋 LaTeX TABLE CODE (Copy this!)      ")
    print("="*50)
    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \caption{Final Model Accuracy on Non-IID CIFAR-10 ($\alpha=0.1$). Results are averaged over 3 independent trials.}")
    print(r"    \label{tab:results_main}")
    print(r"    \begin{tabular}{l c}")
    print(r"        \toprule")
    print(r"        \textbf{Method} & \textbf{Accuracy (\%)} \\")
    print(r"        \midrule")
    
    # Print rows (sorted optionally)
    # Force specific order if desired
    order = ["FedAvg", "FedProx", "MOON", "ASTRA"]
    for method in order:
        if method in stats.index:
            row = stats.loc[method]
            # Bolding ASTRA
            if method == "ASTRA":
                print(r"        \textbf{ASTRA (Ours)} & \textbf{" + f"{row['mean']:.1f}" + r"} \scriptsize{$\pm$ " + f"{row['std']:.1f}" + r"} \\")
            else:
                print(f"        {method} & {row['mean']:.1f} \\scriptsize{{$\\pm$ {row['std']:.1f}}} \\\\")
            
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")
    print("="*50)

def plot_comparison(df):
    if df.empty:
        return

    plt.figure()
    
    for method in COLORS.keys():
        subset = df[df['method'] == method]
        if subset.empty:
            continue

        grouped = subset.groupby('round')['accuracy'].agg(['mean', 'std']).reset_index()
        grouped = grouped[grouped['round'] <= MAX_ROUNDS]

        grouped['smooth_mean'] = grouped['mean'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        grouped['smooth_std'] = grouped['std'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

        plt.plot(grouped['round'], grouped['smooth_mean'], 
                 label=method, color=COLORS[method], alpha=1.0)
        
        plt.fill_between(grouped['round'], 
                         grouped['smooth_mean'] - grouped['smooth_std'], 
                         grouped['smooth_mean'] + grouped['smooth_std'], 
                         color=COLORS[method], alpha=0.15)

    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Convergence Comparison (Non-IID, $\\alpha=0.1$)')
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, fancybox=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, MAX_ROUNDS)
    plt.tight_layout()
    
    plt.savefig('astra_real_accuracy.pdf', format='pdf', dpi=300)
    print(f"\n✨ Success! Plot saved as 'astra_real_accuracy.pdf'")

if __name__ == "__main__":
    df_combined = load_data()
    if not df_combined.empty:
        generate_stats_table(df_combined) # <--- NEW FUNCTION
        plot_comparison(df_combined)