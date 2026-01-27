import torch
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)

def calculate_weight_divergence(global_state, client_updates):
    """
    Calculates the average Euclidean distance between the new global model
    and the individual client models.
    """
    divergences = []

    # Helper to flatten a state_dict into a single 1D tensor
    def flatten_model(state_dict):
        # Concatenate all tensors into one long vector
        return torch.cat([param.view(-1) for param in state_dict.values()])

    # 1. Flatten the global model
    # (Move to CPU to ensure compatibility if mixing devices)
    global_flat = flatten_model(global_state).cpu()

    # 2. Calculate distance for each client
    for client_id, data in client_updates.items():
        client_state = data['state_dict']
        client_flat = flatten_model(client_state).cpu()
        
        # Euclidean distance (L2 Norm)
        dist = torch.norm(global_flat - client_flat).item()
        divergences.append(dist)

    # 3. Return the average distance
    if not divergences:
        return 0.0
    return sum(divergences) / len(divergences)

def federated_average(client_updates):
    """
    Performs the Federated Averaging (FedAvg) algorithm.
    
    Args:
        client_updates (dict): A dictionary where keys are client_ids and
                               values are dicts {'state_dict': ..., 'num_samples': ...}
    
    Returns:
        dict: The new, aggregated global model state_dict.
    """
    print("DEBUG: ⚠️ MY CUSTOM AGGREGATOR IS RUNNING ⚠️") 
    
    if not client_updates:
        logger.warning("No client updates to aggregate.")
        return None

    logger.info(f"Starting FedAvg aggregation for {len(client_updates)} clients.")

    # 1. Calculate the total number of samples
    total_samples = sum(data['num_samples'] for data in client_updates.values())
    if total_samples == 0:
        logger.warning("No samples reported by clients. Cannot aggregate.")
        return None

    # 2. Get the keys from the first model to initialize a new one
    first_update = next(iter(client_updates.values()))['state_dict']
    
    # 3. Create a new state_dict, initialized with zeros
    new_global_state = {key: torch.zeros_like(tensor) for key, tensor in first_update.items()}
    
    # 4. Perform the weighted average
    for client_id, data in client_updates.items():
        weight = data['num_samples'] / total_samples
        state_dict = data['state_dict']
        
        for key in new_global_state:
            if key in state_dict:
                new_global_state[key] += state_dict[key] * weight
            else:
                logger.warning(f"Key {key} missing from client {client_id}. Skipping.")

    # --- NEW: Calculate and Log Drift Metric ---
    try:
        drift = calculate_weight_divergence(new_global_state, client_updates)
        # We use a specific prefix "--- [METRIC]" so it's easy to grep/plot later
        logger.info(f"--- [METRIC] Weight Divergence: {drift:.4f}")
    except Exception as e:
        logger.error(f"Failed to calculate weight divergence: {e}")
    # -------------------------------------------

    logger.info("FedAvg aggregation complete.")
    return new_global_state