import re
import matplotlib.pyplot as plt
from datetime import datetime

# --- ⚙️ CONFIGURATION ---
LOG_FILES = {
    # Update these paths to match your folder structure exactly
    "Baseline (FedProx)": "my_logs/Scenario_A_Fedprox_v5.log", 
    "Hybrid (Ours)":      "my_logs/Scenario_C_Hybrid_v16.log"
}
# Metric Constants
MODEL_SIZE_MB = 4.5  
CLIENTS_PER_ROUND = 5
# ------------------------

def parse_full_logs(filename):
    data = {
        "rounds": [],
        "divergence": [],
        "loss": [],
        "accuracy": [],
        "throughput": []
    }
    
    # Regex Patterns
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    
    # Pattern for your custom metric: "--- [METRIC] Weight Divergence: 0.1234"
    drift_pattern = re.compile(r"Weight Divergence: ([\d\.]+)")
    
    # Pattern for Server Stats: "--- Round 10 Acc: 47.02%, Loss: 1.4982 ---"
    # OR Client Stats if server log is missing: "Round 10 finished. Loss: 0.89, Acc: 71%"
    # We'll rely on the Server Summary line usually printed at end of round
    server_stats_pattern = re.compile(r"--- Round (\d+) Acc: ([\d\.]+)%, Loss: ([\d\.]+)")

    start_time = None
    current_round = -1
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found.")
        return None

    for line in lines:
        # 1. Capture Timestamp
        match_ts = ts_pattern.search(line)
        if match_ts:
            now = datetime.strptime(match_ts.group(1), "%Y-%m-%d %H:%M:%S")

        # 2. Detect Round Start
        if "--- Starting Round" in line:
            match_rnd = re.search(r"Round (\d+)", line)
            if match_rnd:
                current_round = int(match_rnd.group(1))
                start_time = now
        
        # 3. Capture Weight Divergence
        if "Weight Divergence" in line:
            match_d = drift_pattern.search(line)
            if match_d:
                val = float(match_d.group(1))
                # Avoid duplicates if logged multiple times
                if not data["divergence"] or len(data["divergence"]) < current_round + 1:
                    data["divergence"].append((current_round, val))

        # 4. Capture Accuracy & Loss (Server Summary)
        if "--- Round" in line and "Acc:" in line:
            match_stats = server_stats_pattern.search(line)
            if match_stats:
                rnd = int(match_stats.group(1))
                acc = float(match_stats.group(2))
                loss = float(match_stats.group(3))
                data["accuracy"].append((rnd, acc))
                data["loss"].append((rnd, loss))

        # 5. Capture Throughput (End of Round)
        if "Aggregation complete" in line and start_time:
            duration = (now - start_time).total_seconds()
            if duration > 0:
                total_data = (MODEL_SIZE_MB * CLIENTS_PER_ROUND) * 2
                mbps = total_data / duration
                data["throughput"].append((current_round, mbps))
            start_time = None

    return data

def plot_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Setup Data Containers
    results = {}
    for label, fname in LOG_FILES.items():
        results[label] = parse_full_logs(fname)

    # --- PLOT 1: Weight Divergence (The Proof) ---
    ax = axes[0]
    for label, res in results.items():
        if res and res["divergence"]:
            x, y = zip(*res["divergence"])
            ax.plot(x, y, marker='o', linewidth=2, label=label)
    ax.set_title("A. Client Drift (Weight Divergence)", fontweight='bold')
    ax.set_ylabel("L2 Distance ($||W_g - W_i||$)")
    ax.set_xlabel("Rounds")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # --- PLOT 2: Loss Stability (The Consequence) ---
    ax = axes[1]
    for label, res in results.items():
        if res and res["loss"]:
            x, y = zip(*res["loss"])
            ax.plot(x, y, marker='x', linewidth=2, label=label)
    ax.set_title("B. Convergence Stability (Loss)", fontweight='bold')
    ax.set_ylabel("Global Loss")
    ax.set_xlabel("Rounds")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # --- PLOT 3: Efficiency (The Cost) ---
    ax = axes[2]
    for label, res in results.items():
        if res and res["throughput"]:
            x, y = zip(*res["throughput"])
            # Align lengths just in case of mismatch
            ax.plot(x, y, marker='s', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
    ax.set_title("C. System Throughput", fontweight='bold')
    ax.set_ylabel("MB/s")
    ax.set_xlabel("Rounds")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig("thesis_final_comparison.png", dpi=300)
    print("\n✅ Validated! Graph saved to 'thesis_final_comparison.png'")
    plt.show()

if __name__ == "__main__":
    plot_comparison()