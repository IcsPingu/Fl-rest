import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {"New ASTRA (Seed 42)": "#9467bd", "FedProx": "#1f77b4", "Old ASTRA (Avg)": "#7f7f7f"}

def load_data():
    data = []
    
    # 1. NEW ASTRA SEED 42 (The one you just ran)
    # Ensure this file exists! You might need to rename it first if you haven't.
    new_astra_path = "../../results/metrics/metrics_Scenario4_Hybrid_Seed42_new.csv"
    if os.path.exists(new_astra_path):
        df = pd.read_csv(new_astra_path)
        df['method'] = "New ASTRA (Seed 42)"
        data.append(df)
    else:
        # Fallback: Check if it's still named generic 'metrics_Scenario4_Hybrid.csv'
        fallback_path = "../../results/metrics/metrics_Scenario4_Hybrid_Seed42_new.csv"
        if os.path.exists(fallback_path):
            print(f"⚠️  Found generic file '{fallback_path}'. Assuming it is Seed 42.")
            df = pd.read_csv(fallback_path)
            df['method'] = "New ASTRA (Seed 42)"
            data.append(df)
        else:
            print("❌ Could not find Seed 42 data.")

    # 2. FEDPROX (Baseline)
    fedprox_path = "../../results/metrics/metrics_Scenario2_FedProx.csv"
    if os.path.exists(fedprox_path):
        df = pd.read_csv(fedprox_path)
        df['method'] = "FedProx"
        data.append(df)

    if not data: return pd.DataFrame()
    return pd.concat(data, ignore_index=True)

def plot_results(df):
    if df.empty: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. ACCURACY PLOT ---
    sns.lineplot(data=df, x="round", y="accuracy", hue="method", palette=COLORS, ax=ax1, linewidth=2.5)
    ax1.set_title("Accuracy Check (Seed 42)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 2. LOSS PLOT ---
    sns.lineplot(data=df, x="round", y="loss", hue="method", palette=COLORS, ax=ax2, linewidth=2.5)
    ax2.set_title("Loss Check (Seed 42)")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("check_seed42_performance.png", dpi=300)
    print("✅ Performance plot saved: check_seed42_performance.png")

def check_time():
    # Check Training Time for the new run
    times_path = "results_metrics/times_Scenario4_Hybrid_Seed42_new.csv" # Or _Seed42.csv
    if os.path.exists(times_path):
        try:
            df = pd.read_csv(times_path, header=None, names=['round', 'client', 'duration'])
            total_time = df['duration'].sum()
            print(f"\n⏱️  NEW ASTRA TIME (Total): {total_time:.2f} seconds")
            print(f"   (Compare this to FedProx ~1327s)")
        except:
            print("⚠️  Could not read time file.")

if __name__ == "__main__":
    df = load_data()
    plot_results(df)
    check_time()