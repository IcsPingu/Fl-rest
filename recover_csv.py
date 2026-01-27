import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
# 1. Paste the path to the OLD run folder (e.g., from yesterday/earlier)
#    Example: "fl_logs/tensorboard/Cenario_C_Hibrido_Mu0.01_Alpha0.4_0129"
OLD_RUN_FOLDER = "fl_logs/tensorboard/Cenario_C_Hibrido_Mu0.01_Alpha0.3_1450" 

# 2. Output filename
OUTPUT_FILENAME = "recovered_training_times.csv"
# ---------------------

def recover_training_times(log_dir, output_file):
    # Find the event file
    event_path = None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                event_path = os.path.join(root, file)
                break
        if event_path: break
    
    if not event_path:
        print(f"❌ No event file found in {log_dir}")
        return

    print(f"📂 Reading logs from: {event_path}...")
    ea = EventAccumulator(event_path, size_guidance={'scalars': 0})
    ea.Reload()

    # Find all tags related to Training Time
    # usually "System/TrainingTime/client_001", etc.
    tags = ea.Tags()['scalars']
    time_tags = [t for t in tags if "TrainingTime" in t]

    if not time_tags:
        print("❌ No 'TrainingTime' tags found in this log.")
        print(f"   Available tags: {tags}")
        return

    print(f"✅ Found {len(time_tags)} client logs. Reconstructing CSV...")

    # Extract data
    all_data = []
    
    for tag in time_tags:
        # tag example: "System/TrainingTime/client_001"
        client_id = tag.split('/')[-1] # extract "client_001"
        
        events = ea.Scalars(tag)
        for e in events:
            all_data.append({
                "Round": e.step,
                "Client": client_id,
                "TrainingTime": e.value
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort nicely
    df.sort_values(by=["Round", "Client"], inplace=True)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"🎉 Success! Recovered data saved to '{output_file}'")
    print(df.head())

if __name__ == "__main__":
    recover_training_times(OLD_RUN_FOLDER, OUTPUT_FILENAME)