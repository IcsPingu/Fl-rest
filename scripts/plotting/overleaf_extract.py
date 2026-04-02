import pandas as pd
import os

# --- FILE CONFIGURATION ---
# Ensure these match your actual folder structure
files = [
    # ASTRA (Scenario 4)
    ("ASTRA", "../../results/metrics/metrics_Scenario4_Hybrid_Seed10_new.csv"),
    ("ASTRA", "../../results/metrics/metrics_Scenario4_Hybrid_Seed42_new.csv"),
    ("ASTRA", "../../results/metrics/metrics_Scenario4_Hybrid_Seed999_new.csv"),

    # MOON
    ("MOON", "../../results/metrics/metrics_moon_seed10.csv"),
    ("MOON", "../../results/metrics/metrics_moon_seed42.csv"),
    ("MOON", "../../results/metrics/metrics_moon_seed999.csv"),

    # FedProx
    ("FedProx", "../../results/metrics/metrics_Scenario2_FedProx_Seed10.csv"),
    ("FedProx", "../../results/metrics/metrics_Scenario2_FedProx.csv"),
    ("FedProx", "../../results/metrics/metrics_Scenario2_FedProx_Seed999.csv"),

    # FedAvg
    ("FedAvg", "../../results/metrics/metrics_Scenario1_FedAvg_Seed10.csv"),
    ("FedAvg", "../../results/metrics/metrics_Scenario1_FedAvg.csv"),
    ("FedAvg", "../../results/metrics/metrics_Scenario1_FedAvg_Seed999.csv"),
]

def extract_checkpoints():
    all_data = []
    
    print("--- Reading CSV Files ---")
    for method, filepath in files:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df.columns = [c.lower() for c in df.columns] # Lowercase cols
                
                # Find Loss Column
                loss_col = next((c for c in ['loss', 'train_loss', 'global_loss'] if c in df.columns), None)
                
                if 'round' in df.columns and 'accuracy' in df.columns and loss_col:
                    temp = df[['round', 'accuracy', loss_col]].copy()
                    temp.columns = ['round', 'accuracy', 'loss']
                    temp['method'] = method
                    all_data.append(temp)
                else:
                    print(f"⚠️  Missing columns in {filepath}")
            except:
                print(f"❌ Error reading {filepath}")
        else:
            print(f"❌ File not found: {filepath}")

    if not all_data:
        print("No data found!")
        return

    df_combined = pd.concat(all_data, ignore_index=True)
    
    # --- CALCULATE AVERAGES ---
    # Group by Method and Round to get the mean of the 3 seeds
    df_avg = df_combined.groupby(['method', 'round']).mean().reset_index()

    # --- DEFINE CHECKPOINTS ---
    # We only need these specific rounds to reconstruct the curve perfectly
    checkpoints = [0, 5, 10, 20, 30, 40, 49]
    
    print("\n" + "="*60)
    print("📋 COPY THIS TABLE BELOW AND PASTE IT TO ME")
    print("="*60)
    print(f"{'Method':<10} | {'Round':<5} | {'Accuracy (%)':<15} | {'Loss':<15}")
    print("-" * 55)

    for method in ['ASTRA', 'MOON', 'FedProx', 'FedAvg']:
        subset = df_avg[df_avg['method'] == method]
        if subset.empty:
            continue
            
        for r in checkpoints:
            # Find the row for this round
            row = subset[subset['round'] == r]
            if not row.empty:
                acc = row['accuracy'].values[0]
                loss = row['loss'].values[0]
                print(f"{method:<10} | {r:<5} | {acc:.4f}          | {loss:.4f}")
            else:
                # If round missing (e.g., if crashed), print N/A
                print(f"{method:<10} | {r:<5} | N/A             | N/A")
        print("-" * 55)

if __name__ == "__main__":
    extract_checkpoints()