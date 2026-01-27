#!/bin/bash
set -e 

# ==========================================
# 1. CONFIGURATION (EDIT THIS BEFORE RUNNING)
# ==========================================

# --- Choose your Scenario Name (Change this manually every run!) ---
SCENARIO_NAME="Scenario_C_Hybrid_v16"
# --- SCENARIO SETTINGS (Uncomment the one you want) ---

# [SCENARIO A] Only FedProx
#FEDPROX_MU=0.01
#KD_ALPHA=0.0
#KD_TYPE="logits"

# [SCENARIO B] Only KD (Logits)
# FEDPROX_MU=0.0
# KD_ALPHA=0.5
# KD_TYPE="logits"

# [SCENARIO C] Hybrid
FEDPROX_MU=0.01
KD_ALPHA=0.1
KD_TYPE="logits"

# [SCENARIO D] KD Layer Based (New)
# FEDPROX_MU=0.0
# KD_ALPHA=0.5
# KD_TYPE="layer"


# --- System Settings ---
LOG_DIR="my_logs"  # Saving here to avoid permission errors
CONFIG_FILE="config.py"
CLIENTS_HIGH_PERF=10 
CLIENTS_LOW_PERF=10   
TOTAL_ROUNDS=50

# ==========================================
# 2. AUTOMATION LOGIC
# ==========================================

echo "------------------------------------------------------------------"
echo "▶️ PREPARING: $SCENARIO_NAME"
echo "   Settings: MU=$FEDPROX_MU | ALPHA=$KD_ALPHA | TYPE=$KD_TYPE"
echo "------------------------------------------------------------------"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found."
    exit 1
fi

# 1. Update config.py
# Base params
sed -i "s/^TOTAL_ROUNDS = .*/TOTAL_ROUNDS = $TOTAL_ROUNDS/" $CONFIG_FILE
sed -i "s/^TOTAL_CLIENTS = .*/TOTAL_CLIENTS = $(($CLIENTS_HIGH_PERF + $CLIENTS_LOW_PERF))/" $CONFIG_FILE

# Scenario params
sed -i "s/^FEDPROX_MU = .*/FEDPROX_MU = $FEDPROX_MU/" $CONFIG_FILE
sed -i "s/^KD_ALPHA = .*/KD_ALPHA = $KD_ALPHA/" $CONFIG_FILE
sed -i "s/^SCENARIO_NAME = .*/SCENARIO_NAME = \"$SCENARIO_NAME\"/" $CONFIG_FILE

# Handle KD_TYPE (Add it if missing, replace it if present)
if grep -q "KD_TYPE =" "$CONFIG_FILE"; then
    sed -i "s/^KD_TYPE = .*/KD_TYPE = \"$KD_TYPE\"/" $CONFIG_FILE
else
    echo "KD_TYPE = \"$KD_TYPE\"" >> $CONFIG_FILE
fi

echo "✅ Configuration updated."

# 2. Docker Setup
echo "🧹 Cleaning up containers..."
docker-compose stop > /dev/null 2>&1
docker-compose rm -f > /dev/null 2>&1

echo "🔄 Generating docker-compose..."
python generate_compose.py --high $CLIENTS_HIGH_PERF --low $CLIENTS_LOW_PERF > /dev/null

echo "🚀 Building images..."
docker-compose build > /dev/null

echo "📦 Preparing Data..."
docker-compose run --rm server python prepare_data.py > /dev/null

# 3. Execution
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${SCENARIO_NAME}.log"

echo "▶️ STARTING SIMULATION..."
echo "📄 Logs will be saved to: $LOG_FILE"
echo "   (Use 'tail -f $LOG_FILE' to watch in real-time)"

# Run and save output
docker-compose up --exit-code-from server > $LOG_FILE 2>&1

echo "------------------------------------------------------------------"
echo "✅ FINISHED: $SCENARIO_NAME"
echo "------------------------------------------------------------------"