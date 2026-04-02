#!/bin/bash
set -e 

# ==========================================
# 1. CONFIGURATION (EDIT THIS BEFORE RUNNING)
# ==========================================

# --- Choose your Scenario (Uncomment ONE block below) ---

# [CENÁRIO 1] FedAvg (Baseline Puro)
 #SCENARIO_NAME="1_FedAvg"
 #FEDPROX_MU=0.0
 #KD_ALPHA=0.0
 #ENABLE_GRADUATION="False"
 #MOON_MU=0.0

# [CENÁRIO 2] FedProx (Baseline Forte)
 #SCENARIO_NAME="2_FedProx"
 #FEDPROX_MU=0.01
 #KD_ALPHA=0.0
 #ENABLE_GRADUATION="False"
 #MOON_MU=0.0

# [CENÁRIO 3] KD Constante (Sem Graduação - O "Vilão" da sua tese)
 #SCENARIO_NAME="3_KD_Constant"
 #FEDPROX_MU=0.01
 #KD_ALPHA=0.3
 #ENABLE_GRADUATION="False"

# [CENÁRIO 4] Híbrido (O "Herói" - Seu Método)
#SCENARIO_NAME="4_Hybrid_Ours"
#FEDPROX_MU=0.01
#KD_ALPHA=0.3
#ENABLE_GRADUATION="True"
#MOON_MU=0.0 

# [CENÁRIO 5] MOON (Contrastive Learning)
 SCENARIO_NAME="5_MOON"
 FEDPROX_MU=0.0
 KD_ALPHA=0.0
 ENABLE_GRADUATION="False"
 MOON_MU=1.0           #<-- High weight is common for MOsON
 MOON_TEMPERATURE=0.5


# --- System Settings ---
LOG_DIR="results/logs"
CONFIG_FILE="config.py"
CLIENTS_HIGH_PERF=10 
CLIENTS_LOW_PERF=10   
TOTAL_ROUNDS=50

# ==========================================
# 2. AUTOMATION LOGIC
# ==========================================

echo "------------------------------------------------------------------"
echo "▶️ PREPARING: $SCENARIO_NAME"
echo "   Settings: MU=$FEDPROX_MU | ALPHA=$KD_ALPHA | GRAD=$ENABLE_GRADUATION"
echo "------------------------------------------------------------------"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found."
    exit 1
fi

# 1. Update config.py with SED
# Base params
sed -i "s/^SCENARIO_NAME = .*/SCENARIO_NAME = \"$SCENARIO_NAME\"/" $CONFIG_FILE
sed -i "s/^FEDPROX_MU = .*/FEDPROX_MU = $FEDPROX_MU/" $CONFIG_FILE
sed -i "s/^KD_ALPHA = .*/KD_ALPHA = $KD_ALPHA/" $CONFIG_FILE
sed -i "s/^ENABLE_GRADUATION = .*/ENABLE_GRADUATION = $ENABLE_GRADUATION/" $CONFIG_FILE
sed -i "s/^MOON_MU = .*/MOON_MU = $MOON_MU/" $CONFIG_FILE

echo "✅ Configuration updated."

# 2. Docker Setup (Critical for clean experiments)
echo "🧹 Cleaning up containers..."
docker-compose down --remove-orphans > /dev/null 2>&1

#echo "🔄 Generating docker-compose..."
#python3 generate_compose.py --high $CLIENTS_HIGH_PERF --low $CLIENTS_LOW_PERF > /dev/null

echo "🚀 Building images..."
docker-compose build > /dev/null

echo "📦 Preparing Data..."
docker-compose run --rm server python3 scripts/setup/prepare_data.py > /dev/null

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
echo "💾 Saving metrics..."
docker cp fl-rest_server_1:/app/results/metrics.csv "${LOG_DIR}/${SCENARIO_NAME}_metrics.csv"
docker cp fl-rest_server_1:/app/results/training_times.csv "${LOG_DIR}/${SCENARIO_NAME}_times.csv"
echo "------------------------------------------------------------------"