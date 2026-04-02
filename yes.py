import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- 1. ACADEMIC STYLE & FONT SIZING ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

# Increased Font Sizes (+3 points)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 17,          # Was 14, now 17
    'axes.labelsize': 19,     # Was 16, now 19
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,    # Was 13, now 16
    'lines.linewidth': 3,     # Thicker lines to match text
    'figure.figsize': (9, 7)  # Adjusted for single plot aspect ratio
})

# --- 2. YOUR REAL DATA ---
rounds_x = np.array([0, 5, 10, 20, 30, 40, 49])

acc_data = {
    "FedAvg":  np.array([10.0, 20.9, 27.5, 31.3, 36.3, 39.6, 45.0]),
    "FedProx": np.array([10.0, 12.0, 27.4, 39.0, 25.6, 40.8, 30.9]),
    "MOON":    np.array([10.0, 14.5, 26.1, 27.7, 30.1, 32.9, 37.1]),
    "ASTRA":   np.array([10.0, 19.2, 28.7, 40.8, 40.6, 43.9, 47.0]) 
}

loss_data = {
    "FedAvg":  np.array([4.52, 2.82, 2.36, 2.63, 2.25, 1.86, 1.67]),
    "FedProx": np.array([4.30, 3.62, 2.30, 1.90, 2.40, 1.92, 2.66]),
    "MOON":    np.array([4.40, 3.04, 2.45, 2.48, 2.60, 2.34, 2.18]),
    "ASTRA":   np.array([2.83, 2.19, 2.02, 1.69, 1.67, 1.57, 1.64]) 
}

COLORS = {
    "FedAvg": "#7f7f7f",   # Gray
    "FedProx": "#d62728",  # Red
    "MOON": "#2ca02c",     # Green
    "ASTRA": "#1f77b4"     # Blue
}

def smooth_curve(x, y):
    X_Y_Spline = make_interp_spline(x, y)
    X_new = np.linspace(x.min(), x.max(), 300)
    Y_new = X_Y_Spline(X_new)
    return X_new, Y_new

def plot_accuracy():
    plt.figure()
    for method, y_values in acc_data.items():
        # Smooth Line
        x_smooth, y_smooth = smooth_curve(rounds_x, y_values)
        style = ':' if method == 'FedAvg' else '-'
        plt.plot(x_smooth, y_smooth, label=method, color=COLORS[method], linestyle=style, zorder=1)
        # Data Points ("Bolinhas")
        plt.scatter(rounds_x, y_values, color=COLORS[method], s=80, alpha=0.9, edgecolor='white', linewidth=1.0, zorder=2)

    # No Title (Advisor Request)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('fig_accuracy_clean.pdf', format='pdf', dpi=300)
    print("✅ Saved: fig_accuracy_clean.pdf")

def plot_loss():
    plt.figure()
    for method, y_values in loss_data.items():
        # Smooth Line
        x_smooth, y_smooth = smooth_curve(rounds_x, y_values)
        style = ':' if method == 'FedAvg' else '-'
        plt.plot(x_smooth, y_smooth, label=method, color=COLORS[method], linestyle=style, zorder=1)
        # Data Points
        plt.scatter(rounds_x, y_values, color=COLORS[method], s=80, alpha=0.9, edgecolor='white', linewidth=1.0, zorder=2)

    # No Title
    plt.xlabel("Communication Rounds")
    plt.ylabel("Cross-Entropy Loss")
    plt.xlim(0, 50)
    plt.ylim(1.0, 5.0)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('fig_loss_clean.pdf', format='pdf', dpi=300)
    print("✅ Saved: fig_loss_clean.pdf")

if __name__ == "__main__":
    plot_accuracy()
    plot_loss()