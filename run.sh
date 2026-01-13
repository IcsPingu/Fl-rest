#!/bin/bash
set -e # Exit immediately if any command fails

#################################################################
# 1. SIMULATION PARAMETERS
#################################################################

# --- Client & Round Config ---
# 1. PHYSICAL CONTAINERS (How many Docker containers to spawn)
# Total = 8 Containers (Heavy RAM usage if all run at once)
CLIENTS_HIGH_PERF=4 
CLIENTS_LOW_PERF=4   

# 2. LOGICAL TRAINING (How many clients train per round)
# We limit this to 4 to save RAM. The other 4 will stay idle/waiting.
MIN_CLIENTS_PER_ROUND=4       
MIN_CLIENTS_FOR_AGGREGATION=4 

TOTAL_ROUNDS=10

# ---------------------------------------------------------------
# (Derived values - DO NOT EDIT)
# This calculates Total Containers (8)
NUM_CLIENTS=$(($CLIENTS_HIGH_PERF + $CLIENTS_LOW_PERF))

# ---------------------------------------------------------------

# --- Client Training Config ---
LOCAL_EPOCHS=3
BATCH_SIZE=32       
LEARNING_RATE=0.01

# --- Data Config (Non-IID) ---
DIRICHLET_ALPHA=0.5

# --- FL Condition Simulation ---
CLIENT_DROPOUT_RATE=0.0 
ROUND_TIMEOUT_SEC=300     

# --- NEW: Slow Sender & Latency Config ---
SLOW_SENDER_RATE=0.0
SLOW_SENDER_DELAY_SEC=30
NETWORK_LATENCY_RATE=0.0
NETWORK_LATENCY_DELAY_SEC=5

# --- Other System Config ---
DEVICE="auto" 
MOMENTUM=0.9
POLL_INTERVAL=10
RANDOM_SEED=42
SAVED_MODEL_NAME="final_global_model.pth"

# ---------------------------------------------------------------
# (Derived values)
# We keep TOTAL_CLIENTS = 8 so Docker knows how many to build
TOTAL_CLIENTS=$NUM_CLIENTS
CONFIG_FILE="config.py"
# ---------------------------------------------------------------


#################################################################
# 2. CONFIGURATION SCRIPT
# (This section automatically updates config.py)
#################################################################

echo "▶️ Starting simulation with $NUM_CLIENTS containers (Training $MIN_CLIENTS_PER_ROUND per round)..."

# Check if config.py exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found. Cannot configure run."
    exit 1
fi

echo "🔄 Updating $CONFIG_FILE with new parameters..."

# IMPORTANT: These lines now use the variables set to '4' at the top,
# NOT the total number of containers (8).
sed -i "s/^TOTAL_ROUNDS = .*/TOTAL_ROUNDS = $TOTAL_ROUNDS/" $CONFIG_FILE
sed -i "s/^MIN_CLIENTS_PER_ROUND = .*/MIN_CLIENTS_PER_ROUND = $MIN_CLIENTS_PER_ROUND/" $CONFIG_FILE
sed -i "s/^MIN_CLIENTS_FOR_AGGREGATION = .*/MIN_CLIENTS_FOR_AGGREGATION = $MIN_CLIENTS_FOR_AGGREGATION/" $CONFIG_FILE

# This stays 8 (Total registered in simulation)
sed -i "s/^TOTAL_CLIENTS = .*/TOTAL_CLIENTS = $TOTAL_CLIENTS/" $CONFIG_FILE

sed -i "s/^LOCAL_EPOCHS = .*/LOCAL_EPOCHS = $LOCAL_EPOCHS/" $CONFIG_FILE
sed -i "s/^BATCH_SIZE = .*/BATCH_SIZE = $BATCH_SIZE/" $CONFIG_FILE
sed -i "s/^LEARNING_RATE = .*/LEARNING_RATE = $LEARNING_RATE/" $CONFIG_FILE
sed -i "s/^MOMENTUM = .*/MOMENTUM = $MOMENTUM/" $CONFIG_FILE
sed -i "s/^POLL_INTERVAL = .*/POLL_INTERVAL = $POLL_INTERVAL/" $CONFIG_FILE
sed -i "s/^DIRICHLET_ALPHA = .*/DIRICHLET_ALPHA = $DIRICHLET_ALPHA/" $CONFIG_FILE
sed -i "s/^RANDOM_SEED = .*/RANDOM_SEED = $RANDOM_SEED/" $CONFIG_FILE
sed -i "s/^SAVED_MODEL_NAME = .*/SAVED_MODEL_NAME = \"$SAVED_MODEL_NAME\"/" $CONFIG_FILE
sed -i "s/^DEVICE = .*/DEVICE = \"$DEVICE\"/" $CONFIG_FILE
sed -i "s/^CLIENT_DROPOUT_RATE = .*/CLIENT_DROPOUT_RATE = $CLIENT_DROPOUT_RATE/" $CONFIG_FILE
sed -i "s/^ROUND_TIMEOUT_SEC = .*/ROUND_TIMEOUT_SEC = $ROUND_TIMEOUT_SEC/" $CONFIG_FILE
sed -i "s/^SLOW_SENDER_RATE = .*/SLOW_SENDER_RATE = $SLOW_SENDER_RATE/" $CONFIG_FILE
sed -i "s/^SLOW_SENDER_DELAY_SEC = .*/SLOW_SENDER_DELAY_SEC = $SLOW_SENDER_DELAY_SEC/" $CONFIG_FILE
sed -i "s/^NETWORK_LATENCY_RATE = .*/NETWORK_LATENCY_RATE = $NETWORK_LATENCY_RATE/" $CONFIG_FILE
sed -i "s/^NETWORK_LATENCY_DELAY_SEC = .*/NETWORK_LATENCY_DELAY_SEC = $NETWORK_LATENCY_DELAY_SEC/" $CONFIG_FILE

echo "✅ $CONFIG_FILE updated."


#################################################################
# 3. GENERATE DOCKER-COMPOSE FILE
#################################################################

echo "🔄 Generating docker-compose.yml for $NUM_CLIENTS clients..."
python generate_compose.py --high $CLIENTS_HIGH_PERF --low $CLIENTS_LOW_PERF
echo "✅ docker-compose.yml generated."


#################################################################
# 4. EXECUTE SIMULATION
#################################################################

echo "🧹 Cleaning up old containers..."
docker compose down --remove-orphans

LOG_FILE="fl_logs/simulation_$(date +'%Y%m%d_%H%M%S').log"

echo "🚀 Building images..."
docker compose build

echo "📦 Preparing Data (Downloading & Partitioning)..."
docker compose run --rm server python prepare_data.py

echo "▶️ Starting Simulation..."
echo "🪵 Log file will be saved to: $LOG_FILE"

docker compose up --remove-orphans --exit-code-from server | tee "$LOG_FILE"

echo "---"
echo "✅ Simulation complete. Log saved to $LOG_FILE"