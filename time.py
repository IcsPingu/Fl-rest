import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV you created in server/app.py
# Columns expected based on your code: Round, ClientID, Seconds
try:
    df = pd.read_csv("training_times.csv", header=None, names=["Round", "Client", "Time"])
    
    plt.figure(figsize=(10, 6))
    
    # 1. Calculate Average Time per Round
    avg_times = df.groupby("Round")["Time"].mean()
    
    # 2. Plot
    plt.plot(avg_times.index, avg_times.values, marker='o', linestyle='-', color='purple', linewidth=2, label='Optimized Hybrid (Cached)')
    
    # Optional: Add a "Baseline" line if you know the old time (e.g., 15s)
    # plt.axhline(y=15.0, color='gray', linestyle='--', label='Original Hybrid (Est.)')
    
    plt.title("Average Training Time per Round (Efficiency Analysis)")
    plt.xlabel("Round")
    plt.ylabel("Training Time (Seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save
    plt.savefig("thesis_time_efficiency.png")
    print("✅ Graph saved to 'thesis_time_efficiency.png'")
    plt.show()
    
    print(f"Average Training Time: {df['Time'].mean():.2f} seconds")

except FileNotFoundError:
    print("❌ 'training_times.csv' not found. Make sure the simulation ran with the new Server code.")