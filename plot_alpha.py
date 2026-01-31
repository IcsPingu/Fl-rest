import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.figsize": (9, 6)
})

# --- FILE MAPPING ---
# Here we map each scenario specifically to the files you have.
# Note: For ASTRA 0.1, I inferred 'metrics_' based on your 'times_' filename.
FILE_MAP = {
    "0.1": {
        "label": "0.1 (Severe)",
        "FedProx": "results_metrics/metrics_Scenario2_FedProx_Seed999.csv",
        "ASTRA":   "results_metrics/metrics_Scenario4_Hybrid_Seed999_new.csv"
    },
    "0.3": {
        "label": "0.3 (Moderate)",
        "FedProx": "results_metrics/metrics_FedProx_Alpha0.3.csv",
        "ASTRA":   "results_metrics/metrics_ASTRA_Alpha0.3.csv"
    },
    "0.5": {
        "label": "0.5 (Light)",
        "FedProx": "results_metrics/metrics_FedProx_Alpha0.5.csv",
        "ASTRA":   "results_metrics/metrics_ASTRA_Alpha0.5.csv"
    }
}

COLORS = {"FedProx": "#1f77b4", "ASTRA": "#9467bd"}

def get_max_accuracy(filepath):
    """
    Reads a metrics file robustly, handling potential text corruption.
    Returns the maximum accuracy found in the file.
    """
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return None
    try:
        # Read as string first to avoid crashing on "loss..." text
        df = pd.read_csv(filepath, dtype=str)
        
        # Check if 'accuracy' column exists
        if "accuracy" not in df.columns:
            # Fallback: sometimes columns are misplaced, check index -1 or similar if needed
            # But usually header remains. Let's assume header is fine.
            return None

        # Force convert to numeric, turning text garbage into NaN
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
        
        # Drop corrupted rows
        df = df.dropna(subset=["accuracy"])
        
        if df.empty:
            return None

        # Return the absolute best accuracy achieved
        return df["accuracy"].max()
        
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def main():
    print("--- GENERATING HETEROGENEITY ANALYSIS ---")
    print(f"{'Dirichlet α':<15} | {'FedProx':<10} | {'ASTRA (Ours)':<15} | {'Improvement':<10}")
    print("-" * 65)

    plot_data = []
    
    # Process each alpha scenario in order
    for alpha in ["0.1", "0.3", "0.5"]:
        scenario = FILE_MAP[alpha]
        row_label = scenario["label"]
        
        fedprox_path = scenario["FedProx"]
        astra_path = scenario["ASTRA"]
        
        # Get Scores
        fedprox_acc = get_max_accuracy(fedprox_path)
        astra_acc = get_max_accuracy(astra_path)
        
        # Handle missing data (use 0.0 so plot doesn't crash, but warn user)
        if fedprox_acc is None: fedprox_acc = 0.0
        if astra_acc is None:   astra_acc = 0.0
        
        # Add to Plot Data
        if fedprox_acc > 0:
            plot_data.append({"Alpha Label": row_label, "Method": "FedProx", "Accuracy": fedprox_acc})
        if astra_acc > 0:
            plot_data.append({"Alpha Label": row_label, "Method": "ASTRA",   "Accuracy": astra_acc})

        # Calculate Improvement
        if fedprox_acc > 0:
            improvement = ((astra_acc - fedprox_acc) / fedprox_acc) * 100
            imp_str = f"+{improvement:.1f}%"
        else:
            imp_str = "N/A"

        print(f"{row_label:<15} | {fedprox_acc:.2f}%     | {astra_acc:.2f}%          | {imp_str:<10}")

    if not plot_data:
        print("\n❌ No valid data found to plot.")
        return

    df = pd.DataFrame(plot_data)

    # --- PLOTTING ---
    plt.figure(figsize=(9, 6))
    
    # Grouped Bar Chart
    ax = sns.barplot(
        data=df,
        x="Alpha Label",
        y="Accuracy",
        hue="Method",
        palette=COLORS,
        edgecolor="black",
        linewidth=1
    )

    # Add accuracy labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')

    plt.title("Model Robustness to Data Heterogeneity (Dirichlet Distribution)", fontweight='bold')
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Heterogeneity Level (Dirichlet α)")
    plt.ylim(0, df["Accuracy"].max() * 1.2) # Extra headroom
    plt.legend(title="Method", loc='upper left', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("fig_heterogeneity_sensitivity.png", dpi=300)
    print("\n✅ Plot saved: fig_heterogeneity_sensitivity.png")

if __name__ == "__main__":
    main()