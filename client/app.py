import io
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import requests
import config
# Ensure we import the function correctly
from client.trainer import train_model
from client.model_utils import get_model_bytes, set_model_from_bytes
from shared.models import get_model
import logging
import time
import os
import json
import random

SERVER_URL = os.getenv("SERVER_URL", "http://server:5000")
CLIENT_ID = os.getenv("CLIENT_ID", "default_client")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def download_global_model(round_number):
    try:
        response = requests.get(f"{SERVER_URL}/download_model", params={"round": round_number})
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def register_client():
    """Registers the client with the central server."""
    attempts = 0
    while attempts < 5:
        try:
            response = requests.post(f"{SERVER_URL}/register", json={"client_id": CLIENT_ID})
            if response.status_code == 200:
                logger.info("✅ Registered with server successfully.")
                return True
        except Exception as e:
            logger.warning(f"Registration attempt {attempts+1} failed: {e}")
        
        time.sleep(3)
        attempts += 1
    return False

def submit_model_update(model, num_samples, metrics):
    """Sends trained weights and metrics back to the server."""
    url = f"{SERVER_URL}/submit_update"
    model_bytes = get_model_bytes(model)
    
    # Metadata includes the metrics dict returned by the trainer
    metadata = {
        "client_id": CLIENT_ID, 
        "num_samples": num_samples, 
        "metrics": metrics
    }
    
    files = {
        'model': ('model.pth', model_bytes, 'application/octet-stream'), 
        'json': (None, json.dumps(metadata), 'application/json')
    }
    
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            logger.info("🚀 Model update submitted successfully.")
            return True
        else:
            logger.error(f"Update submission failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error submitting update: {e}")
        return False

def check_server_status():
    try: 
        return requests.get(f"{SERVER_URL}/status").json()
    except: 
        return None

def get_client_dataloader(client_id_str):
    """Loads specific partition based on Client ID (e.g., client_001)."""
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Dataset path matches Docker volume map
    data_root = './data'
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
        
        # Load partition file created by prepare_data.py
        with open(f'{data_root}/partitions.json', 'r') as f:
            partitions = json.load(f)
            
        # Parse "client_001" -> "1"
        client_idx = str(int(client_id_str.split('_')[-1]))
        
        indices = partitions.get(client_idx)
        if not indices:
            raise ValueError(f"No partition found for client index {client_idx}")

        subset = Subset(train_dataset, indices)
        return DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=True)
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None

# --- Main Loop ---

def main():
    logger.info(f"--- Starting FL Client: {CLIENT_ID} ---")
    
    # 1. Register
    if not register_client():
        logger.error("Could not register. Exiting.")
        return

    # 2. Setup Model & Data
    global_model = get_model(config.MODEL_NAME)
    client_dataloader = get_client_dataloader(CLIENT_ID)
    
    if client_dataloader is None:
        logger.error("Failed to load dataloader. Exiting.")
        return

    current_round = -1
    model_bytes = None

    # 3. Training Loop
    while True:
        # A. Poll for Server Status
        try:
            status = check_server_status()
        except:
            time.sleep(config.POLL_INTERVAL)
            continue

        if not status:
            time.sleep(config.POLL_INTERVAL)
            continue

        server_round = status.get("current_round", 0)
        server_state = status.get("status")

        if server_state == "TRAINING_COMPLETE":
            logger.info("Server finished training. Exiting.")
            break

        # B. Check if it's a new round
        if server_round > current_round:
            logger.info(f"--- New Round Detected: {server_round} ---")
            
            # Download new Global Model
            attempts = 0
            while attempts < 3:
                model_bytes = download_global_model(server_round)
                if model_bytes:
                    break
                time.sleep(2)
                attempts += 1
            
            if not model_bytes:
                logger.error("Failed to download global model. Retrying later.")
                time.sleep(5)
                continue

            # Load weights into local model
            set_model_from_bytes(global_model, model_bytes)
            current_round = server_round

            # C. Train (Here is the FIX)
            start_time = time.time()
            
            # Note: We pass 'config' so the trainer knows if it should use FedProx/KD
            # We DO NOT pass teacher_model or logits here anymore (Trainer handles it)
            metrics = train_model(
                client_id=CLIENT_ID,
                model=global_model,
                train_loader=client_dataloader,
                config=config
            )
            
            training_time = metrics.get("training_time_sec", time.time() - start_time)
            logger.info(f"Round {current_round} finished. Loss: {metrics.get('loss'):.4f}, Acc: {metrics.get('accuracy'):.2f}%")

            # D. Submit Update
            num_samples = len(client_dataloader.dataset)
            submit_model_update(global_model, num_samples, metrics)
            
            logger.info(f"Update submitted for Round {current_round}. Waiting...")

        # Sleep before polling again
        time.sleep(config.POLL_INTERVAL)

if __name__ == "__main__":
    main()