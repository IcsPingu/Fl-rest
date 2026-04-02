import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. ACADEMIC STYLE SETUP ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "figure.figsize": (10, 6),
    "lines.linewidth": 1.5
})

# --- 2. EXACT DATA INPUT (From your logs) ---
data = [
    # --- Alpha 0.1 (Severe) ---
    {"Alpha": "0.1 (Severe)", "Method": "FedAvg",  "Accuracy": 45.},
    {"Alpha": "0.1 (Severe)", "Method": "FedProx", "Accuracy": 30.9},
    {"Alpha": "0.1 (Severe)", "Method": "MOON",    "Accuracy": 37.1},
    {"Alpha": "0.1 (Severe)", "Method": "ASTRA",   "Accuracy": 47.0},

    # --- Alpha 0.3 (Moderate) ---
    {"Alpha": "0.3 (Moderate)", "Method": "FedAvg",  "Accuracy": 50.50},
    {"Alpha": "0.3 (Moderate)", "Method": "FedProx", "Accuracy": 48.00},
    {"Alpha": "0.3 (Moderate)", "Method": "MOON",    "Accuracy": 52.10},
    {"Alpha": "0.3 (Moderate)", "Method": "ASTRA",   "Accuracy": 55.40},

    # --- Alpha 0.5 (Mild) ---
    {"Alpha": "0.5 (Mild)", "Method": "FedAvg",  "Accuracy": 58.00},
    {"Alpha": "0.5 (Mild)", "Method": "FedProx", "Accuracy": 60.20},
    {"Alpha": "0.5 (Mild)", "Method": "MOON",    "Accuracy": 63.50},
    {"Alpha": "0.5 (Mild)", "Method": "ASTRA",   "Accuracy": 65.80}
]

df = pd.DataFrame(data)

# --- 3. COLORS & PATTERNS ---
METHODS = ["FedAvg", "FedProx", "MOON", "ASTRA"]
COLORS = {
    "FedAvg":  "#7f7f7f",   # Gray
    "FedProx": "#d62728",   # Red
    "MOON":    "#2ca02c",   # Green
    "ASTRA":   "#1f77b4"    # Blue (Ours)
}
HATCHES = {
    "FedAvg":  "//",
    "FedProx": "..",
    "MOON":    "xx",
    "ASTRA":   ""     # Solid for ours
}

def plot_final_chart():
    plt.figure(figsize=(10, 6))
    
    # Draw Bar Chart
    ax = sns.barplot(
        data=df,
        x="Alpha",
        y="Accuracy",
        hue="Method",
        hue_order=METHODS,
        palette=COLORS,
        edgecolor="black",
        linewidth=1.2,
        errorbar=None
    )

    # Apply Hatches (Patterns)
    for container, method in zip(ax.containers, METHODS):
        for bar in container:
            bar.set_hatch(HATCHES[method])
            
            # Add Value Labels
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Decoration
    plt.title("Model Robustness vs. Data Heterogeneity ($\\alpha$)", fontweight='bold')
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Heterogeneity Level (Dirichlet Distribution)")
    plt.ylim(0, 75) # Set limit slightly higher than max (65.8)
    
    plt.legend(title="Method", loc='upper left', ncol=4, frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save
    plt.savefig("fig_heterogeneity_sensitivity.pdf", dpi=300)
    plt.savefig("fig_heterogeneity_sensitivity.png", dpi=300)
    print("✅ Final Chart Saved: fig_heterogeneity_sensitivity.png")

if __name__ == "__main__":
    plot_final_chart()