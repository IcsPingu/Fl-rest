import logging
from flask import Flask, request, jsonify, send_file
from collections import OrderedDict
import os
import io
import signal
import threading
from threading import Timer, Lock
import torch.nn.functional as F 
import random
import json
import time

# --- ML Imports ---
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from shared.models import get_model
from server.strategies import get_strategy 
import config
from server.aggregator import federated_average

# --- Add this near the top (after imports) ---
RESULTS_DIR = "results"  # Matches the /app/results volume
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.csv")
TIMES_FILE = os.path.join(RESULTS_DIR, "training_times.csv")

# Initialize metrics file with headers if it doesn't exist
if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w") as f:
        f.write("round,accuracy,loss\n")


# --- 1. CONFIGURAÇÃO DE LOG (MOVIDO PARA O TOPO) ---
logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

werkzeug_logger = logging.getLogger('werkzeug')

# Filtro para limpar logs do Flask
class StatusFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "GET /status" in msg or "POST /register" in msg:
            return False 
        return True

werkzeug_logger.addFilter(StatusFilter())

# --- 2. TENSORBOARD DINÂMICO ---
# Define o nome da pasta com base no cenário ativo no config.py

if config.FEDPROX_MU > 0 and config.ENABLE_PARAMETER_BASED_KD:
    # Cenário C: Híbrido (Mostra os valores de Mu e Alpha no nome)
    run_name = f"Cenario_C_Hibrido_Mu{config.FEDPROX_MU}_Alpha{config.KD_ALPHA}"

elif config.ENABLE_PARAMETER_BASED_KD:
    # Cenário A: Só KD
    run_name = "Cenario_A_Parametros_OnlyKD"

elif config.FEDPROX_MU > 0:
    # Cenário: Só FedProx
    run_name = f"Cenario_FedProx_Mu{config.FEDPROX_MU}"

elif config.ENABLE_FEATURE_BASED_KD:
    # Cenário B: Features
    run_name = "Cenario_B_Features"

else:
    run_name = "Standard_FedAvg"

# Adiciona hora para garantir unicidade
timestamp = time.strftime("%H%M") 
folder_name = f"{run_name}_{timestamp}"

log_dir = os.path.join("fl_logs", "tensorboard", folder_name)
tb_writer = SummaryWriter(log_dir=log_dir)

# Agora o logger já existe e esta linha vai funcionar!
logger.info(f"Writing TensorBoard logs to: {log_dir}")

# --- Configuration ---
fl_state = {
    "status": "WAITING", 
    "current_round": 0,
    "client_updates": [],
    "registered_clients": set(),
    "aggregation_lock": Lock(), 
    "round_timer": None          
}

global_models_by_round = {}

global_test_logits = None
fl_strategy = None

def setup_initial_model():
    global global_models_by_round, fl_strategy
    
    initial_model = get_model(config.MODEL_NAME)
    
    fl_strategy = get_strategy(
        config.AGGREGATION_STRATEGY, 
        global_model=initial_model,
        lr=config.SERVER_LEARNING_RATE,
        momentum=config.SERVER_MOMENTUM
    )
    
    global_models_by_round[0] = initial_model.state_dict()
    
    logger.info(f"Initial model for round 0 created.")
    
    # [SCENARIO B] If enabled, pre-calculate logits for Round 0
    if config.ENABLE_FEATURE_BASED_KD:
        # We need to make sure test_loader is available
        if 'TEST_LOADER' in app.config:
             update_global_logits(global_models_by_round[0], app.config['TEST_LOADER'])

def state_dict_to_bytes(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return buffer.read()

# --- Helper Functions ---


def load_test_data():
    logger.info("Loading CIFAR-10 test dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=2)
    logger.info("Test dataset loaded.")
    return test_loader

def update_global_logits(model_state, test_loader):
    """[SCENARIO B] Pre-calculate global logits on public/test data."""
    global global_test_logits
    
    if not config.ENABLE_FEATURE_BASED_KD:
        return

    logger.info("Generating global logits for Feature-based KD...")
    
    model = get_model(config.MODEL_NAME)
    model.load_state_dict(model_state)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            all_logits.append(outputs.cpu())
    
    if all_logits:
        global_test_logits = torch.cat(all_logits)
        logger.info(f"Global logits updated. Shape: {global_test_logits.shape}")

def evaluate_model(model_state_tensors, test_loader):
    forced_device = config.DEVICE.lower()
    if forced_device == "cpu":
        device = torch.device("cpu")
    elif forced_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available! Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config.MODEL_NAME)
    model.load_state_dict(model_state_tensors)
    model.to(device) 
    model.eval() 
    
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad(): 
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss

def aggregate_models(updates):
    if not updates:
        return None

    first_update = updates[0]
    ref_keys = first_update['model_update'].keys()
    
    param_sum = {k: 0.0 for k in ref_keys}
    weight_sum = {k: 0 for k in ref_keys}

    for update in updates:
        client_samples = update['num_samples']
        client_state = update['model_update']
        
        for key in ref_keys:
            if key in client_state:
                if isinstance(param_sum[key], float):
                    param_sum[key] = client_state[key].float() * client_samples
                else:
                    param_sum[key] += client_state[key].float() * client_samples
                
                weight_sum[key] += client_samples

    avg_state_dict = OrderedDict()
    for key in ref_keys:
        count = weight_sum[key]
        if count > 0:
            avg_state_dict[key] = param_sum[key] / count
        else:
            avg_state_dict[key] = first_update['model_update'][key]
                        
    return avg_state_dict

def check_and_aggregate(test_loader):
    global fl_strategy
    
    if fl_state["status"] == "AGGREGATING":
        return
        
    if fl_state["round_timer"]:
        fl_state["round_timer"].cancel()
        fl_state["round_timer"] = None
        
    fl_state["status"] = "AGGREGATING"
    current_round = fl_state["current_round"]
    
    if len(fl_state["client_updates"]) == 0:
        logger.warning(f"No updates received. Skipping aggregation.")
    else:
        logger.info(f"--- Aggregating {len(fl_state['client_updates'])} updates ---")
        
        valid_updates = []
        for update in fl_state["client_updates"]:
            valid_updates.append(update)
        
        # --- FIX: Convert to Dictionary format for your Custom Aggregator ---
        # Your aggregator expects: {client_id: {'state_dict': ..., 'num_samples': ...}}
        updates_dict = {
            u['client_id']: {
                'state_dict': u['model_update'], 
                'num_samples': u['num_samples']
            }
            for u in valid_updates
        }

        # CALL YOUR CUSTOM FUNCTION
        logger.info("⚡ Using Custom Aggregator with Drift Metric...")
        new_global_model_tensors = federated_average(updates_dict)
        # -------------------------------------------------------------------
        logger.info(f"Aggregation complete.")
        
        if new_global_model_tensors:
            accuracy, loss = evaluate_model(new_global_model_tensors, test_loader)
            tb_writer.add_scalar("Global/Accuracy", accuracy, fl_state["current_round"])
            tb_writer.add_scalar("Global/Loss", loss, fl_state["current_round"])
            tb_writer.flush()

            # --- NEW CODE (ADD THIS TO SAVE CSV) ---
            logger.info(f"💾 Saving metrics to {METRICS_FILE}...")
            with open(METRICS_FILE, "a") as f:
                f.write(f"{fl_state['current_round']},{accuracy},{loss}\n")

            logger.info(f"--- Round {current_round} Acc: {accuracy:.2f}%, Loss: {loss:.4f} ---")
            
            global_models_by_round[current_round + 1] = new_global_model_tensors
            
            if config.ENABLE_FEATURE_BASED_KD:
                update_global_logits(new_global_model_tensors, test_loader)
            
            if current_round == config.TOTAL_ROUNDS - 1:
                try:
                    save_path = os.path.join("fl_logs", config.SAVED_MODEL_NAME)
                    os.makedirs("fl_logs", exist_ok=True)
                    torch.save(new_global_model_tensors, save_path)
                    logger.info(f"---Final model saved ---")
                except Exception as e:
                    logger.error(f"Failed to save final model: {e}")

    fl_state["current_round"] += 1
    
    if fl_state["current_round"] >= config.TOTAL_ROUNDS:
        logger.info(f"--- Training complete. ---")
        fl_state["status"] = "TRAINING_COMPLETE"
        threading.Timer(2.0, shutdown_server).start()
    else:
        start_next_round_timer(test_loader)

def trigger_aggregation(test_loader):
    with fl_state["aggregation_lock"]:
        if fl_state["status"] == "WAITING": 
            check_and_aggregate(test_loader) 

def start_next_round_timer(test_loader): 
    global fl_state
    round_num = fl_state["current_round"]
    logger.info(f"--- Starting Round {round_num}. Timeout: {config.ROUND_TIMEOUT_SEC}s ---")
    fl_state["status"] = "WAITING"
    fl_state["client_updates"] = []
    
    fl_state["round_timer"] = Timer(config.ROUND_TIMEOUT_SEC, trigger_aggregation, args=[test_loader])
    fl_state["round_timer"].start()

def shutdown_server():
    os.kill(os.getpid(), signal.SIGINT)

# --- Routes ---

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    client_id = data.get("client_id")
    with fl_state["aggregation_lock"]:
        if client_id not in fl_state["registered_clients"]:
            fl_state["registered_clients"].add(client_id)
            logger.info(f"Client {client_id} registered.")
        
        if fl_state["current_round"] == 0 and fl_state["round_timer"] is None:
            # Init logits for round 0 if needed
            if config.ENABLE_FEATURE_BASED_KD and global_models_by_round.get(0):
                 with app.app_context():
                    update_global_logits(global_models_by_round[0], app.config['TEST_LOADER'])
            
            start_next_round_timer(app.config['TEST_LOADER'])    
    return jsonify({"status": fl_state["status"]})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": fl_state["status"], "current_round": fl_state["current_round"]})

@app.route('/download_model', methods=['GET'])
def download_model():
    requested_round = request.args.get('round', type=int)
    state_dict = global_models_by_round.get(requested_round)
    if state_dict:
        return send_file(
            io.BytesIO(state_dict_to_bytes(state_dict)),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='model.pth'
        )
    return jsonify({"error": "Not found"}), 404

@app.route('/get_global_logits', methods=['GET'])
def get_global_logits():
    global global_test_logits
    if global_test_logits is None:
        return jsonify({"error": "Logits not ready"}), 404
    buffer = io.BytesIO()
    torch.save(global_test_logits, buffer)
    buffer.seek(0)
    return send_file(buffer, mimetype='application/octet-stream', as_attachment=True, download_name='logits.pth')

@app.route('/submit_update', methods=['POST'])
def submit_update():
    if fl_state["status"] != "WAITING":
        return jsonify({"error": "Not accepting updates"}), 400
    
    try:
        metadata = json.loads(request.form['json'])
        client_id = metadata.get('client_id')
        num_samples = metadata.get('num_samples')
        metrics = metadata.get('metrics', {}) 

        # --- START OF NEW CODE ---
        # Extract and Log the time explicitly to the terminal and a CSV
        train_time = metrics.get('training_time_sec', 0.0)
        
        # 1. Print to Docker Logs (so you see it in "docker-compose up")
        print(f"⏱️ [TIME] Client {client_id} finished training in {train_time:.2f}s")
        
        # 2. Save to CSV (for easy graphing later)
        # Saves as: Round, ClientID, Seconds
        with open(TIMES_FILE, "a") as f:
            f.write(f"{fl_state['current_round']},{client_id},{train_time}\n")
        # --- END OF NEW CODE ---

        file_bytes = request.files['model'].read()
        client_state_dict = torch.load(io.BytesIO(file_bytes), map_location='cpu')
        
        with fl_state["aggregation_lock"]:
            if fl_state["status"] == "WAITING":
                if not any(u['client_id'] == client_id for u in fl_state["client_updates"]):
                    fl_state["client_updates"].append({
                        "client_id": client_id,
                        "num_samples": num_samples,
                        "model_update": client_state_dict,
                        "metrics": metrics
                    })
                    logger.info(f"Received update from {client_id}")
                    
                    # Log metrics
                    step = fl_state["current_round"]
                    tb_writer.add_scalar(f"System/TrainingTime/{client_id}", metrics.get('training_time_sec', 0), step)
                    
                    if len(fl_state["client_updates"]) >= config.MIN_CLIENTS_FOR_AGGREGATION:
                        check_and_aggregate(app.config['TEST_LOADER'])
        return jsonify({"status": "Received"})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        test_loader = load_test_data()
        app.config['TEST_LOADER'] = test_loader
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        exit(1)
        
    setup_initial_model() 
    app.run(host='0.0.0.0', port=5000, debug=False)