import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION AREA -----------------------------------------------------
# 1. Path to your experiment folders (where the 'events.out.tfevents' files are)
#    Example: "runs/Sep15_12-30-55_Baseline"
LOG_DIR_BASELINE = "fl_logs/tensorboard/Cenario_C_Hibrido_Mu0.01_Alpha0.0_1506"  
LOG_DIR_HYBRID   = "fl_logs/tensorboard/Cenario_C_Hibrido_Mu0.01_Alpha0.1_2131"

# 2. The Tag you want to plot (e.g., 'Test/Accuracy', 'Train/Loss', 'Accuracy')
#    If you don't know it, run the script once; it will print available tags.
TAG_NAME = "Global/Accuracy"  
# ----------------------------------------------------------------------------

def find_event_file(log_dir):
    """Recursively search for the .tfevents file in a directory."""
    if not os.path.exists(log_dir):
        return None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                return os.path.join(root, file)
    return None

def extract_data(log_dir, tag):
    """Parses the tfevents file and extracts step vs value."""
    event_path = find_event_file(log_dir)
    
    if not event_path:
        print(f"❌ No event file found in: {log_dir}")
        return [], []

    print(f"📂 Loading: {event_path}")
    
    # Load the event accumulator
    # size_guidance=0 loads all events (no downsampling)
    ea = EventAccumulator(event_path, size_guidance={ 'scalars': 0 })
    ea.Reload()

    # Check available tags
    valid_tags = ea.Tags()['scalars']
    if tag not in valid_tags:
        print(f"   ⚠️  Tag '{tag}' not found. Available tags: {valid_tags}")
        return [], []

    # Extract data
    events = ea.Scalars(tag)
    steps = [x.step for x in events]
    values = [x.value for x in events]
    
    return steps, values

def plot_comparison():
    # 1. Extract Data
    print("--- Extracting Baseline Data ---")
    steps_base, val_base = extract_data(LOG_DIR_BASELINE, TAG_NAME)
    
    print("\n--- Extracting Hybrid Data ---")
    steps_hybrid, val_hybrid = extract_data(LOG_DIR_HYBRID, TAG_NAME)

    # 2. Plotting
    plt.figure(figsize=(10, 6))

    # Plot Baseline
    if steps_base:
        plt.plot(steps_base, val_base, label='Baseline (FedProx)', 
                 color='gray', linestyle='--', linewidth=2, alpha=0.8)
    
    # Plot Hybrid
    if steps_hybrid:
        plt.plot(steps_hybrid, val_hybrid, label='Hybrid (Ours)', 
                 color='purple', marker='o', markersize=4, linewidth=2)

    # Formatting
    plt.title(f"Performance Comparison: {TAG_NAME}", fontsize=14)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel(TAG_NAME, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Save
    output_filename = "thesis_comparison_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ Graph saved to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    plot_comparison()