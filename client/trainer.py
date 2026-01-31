import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import gc
import re  # Added missing import
from shared.models import get_model
from torch.cuda.amp import autocast, GradScaler # Import at top
from config import ENABLE_GRADUATION

# --- RTX 4090 OPTIMIZATION ---
# We keep this to ensure we are as fast as possible even with 5 epochs
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

class ModelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ModelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_local, z_global, z_prev):
        # Calculate similarity with Global (Positive - we want to be close)
        sim_global = self.cosine_similarity(z_local, z_global) / self.temperature
        
        # Calculate similarity with Previous Local (Negative - we want to move away)
        sim_prev = self.cosine_similarity(z_local, z_prev) / self.temperature
        
        # We want to maximize sim_global and minimize sim_prev
        logits = torch.cat((sim_global.unsqueeze(1), sim_prev.unsqueeze(1)), dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(z_local.device)
        
        return self.criterion(logits, labels)

def get_device(device_config):
    if device_config == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_config == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. CLASSE DE LOSS INTELIGENTE (Support for Logits & Layer) ---
class DoubleRegLoss(nn.Module):
    def __init__(self, mu=0.01, alpha=0.5, temperature=2.0, confidence_threshold=0.0, kd_type='logits', device='cpu'):
        super(DoubleRegLoss, self).__init__()
        self.mu = mu
        self.alpha = alpha
        self.T = temperature
        self.conf_thresh = confidence_threshold  # <--- This is the new part we added
        self.kd_type = kd_type
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, labels, student_params, global_params, teacher_logits=None, student_features=None, teacher_features=None):
        # 1. Standard Task Loss (CrossEntropy)
        loss = self.ce_loss(student_logits, labels)

        # 2. FedProx Regularization (Gravity)
        if self.mu > 0 and global_params:
            prox_term = 0.0
            for w, w_t in zip(student_params, global_params):
                prox_term += (w - w_t).norm(2)**2
            loss += (self.mu / 2) * prox_term

        # 3. Knowledge Distillation (The Teacher)
        if self.alpha > 0 and teacher_logits is not None:
            # --- CONFIDENCE FILTERING LOGIC ---
            if self.conf_thresh > 0:
                # Calculate Teacher's confidence
                teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
                max_probs, _ = torch.max(teacher_probs, dim=1)
                
                # Create a mask: 1 if Confident, 0 if Confused
                mask = (max_probs >= self.conf_thresh).float().unsqueeze(1)
                
                # Calculate standard KL Divergence
                distillation_loss = nn.KLDivLoss(reduction='none')(
                    F.log_softmax(student_logits / self.T, dim=1),
                    F.softmax(teacher_logits / self.T, dim=1)
                )
                
                # Apply the mask: Only learn from confident examples
                # We sum the loss and divide by the number of confident samples to keep scale correct
                distillation_loss = torch.sum(distillation_loss * mask) / (torch.sum(mask) + 1e-8)
                
            else:
                # Standard KD without filtering
                distillation_loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_logits / self.T, dim=1),
                    F.softmax(teacher_logits / self.T, dim=1)
                )

            loss += self.alpha * (self.T * self.T) * distillation_loss

        return loss

def train_model(client_id, model, train_loader, config):
    device = get_device(config.DEVICE)
    model.to(device)
    model.train() 

    # --- CONFIGS ---
    current_round = getattr(config, 'CURRENT_ROUND', 0)
    global_rounds = getattr(config, 'GLOBAL_ROUNDS', 50)
    
    # --- "TEST & CORRECT" SCHEDULE ---
    # Logic: 
    # 1. Bootcamp (Rounds 0-10): Teacher ON to prevent early divergence.
    # 2. Semester (Rounds 11+):  Teacher ON every 2 rounds (Correction).
    #    - Even rounds (12, 14...): Teacher ON (Correction)
    #    - Odd rounds  (11, 13...): Teacher OFF (Test/Speed)
    
    BOOTCAMP_ROUNDS = 10
    KD_INTERVAL = 2 
    
    is_teacher_active = False

    if current_round <= BOOTCAMP_ROUNDS:
        is_teacher_active = True
    elif current_round % KD_INTERVAL == 0:
        is_teacher_active = True
    else:
        is_teacher_active = False

    # --- ALPHA CONFIG ---
    initial_alpha = getattr(config, 'KD_ALPHA', 0.0)
    mu_val = getattr(config, 'FEDPROX_MU', 0.0)
    enable_graduation = getattr(config, 'ENABLE_GRADUATION', True)
    
    if initial_alpha == 0.0:
        current_alpha = 0.0
    elif not is_teacher_active:
        # ⚡ SPEEDUP: Teacher inactive -> Alpha 0 -> Skip computation
        current_alpha = 0.0
    elif enable_graduation:
        decayed_alpha = initial_alpha * (1 - (current_round / global_rounds))
        current_alpha = max(0.0, decayed_alpha)
    else:
        current_alpha = initial_alpha

    # Initialize Criterion
    criterion = DoubleRegLoss(
        mu=mu_val,
        alpha=current_alpha,
        temperature=3.0,
        confidence_threshold=0.0,
        kd_type='logits',
        device=device
    )
    
    # --- MOON CONFIG ---
    moon_mu = getattr(config, 'MOON_MU', 0.0)
    if moon_mu is None: moon_mu = 0.0
    moon_temp = getattr(config, 'MOON_TEMPERATURE', 0.5)
    
    contrastive_criterion = None
    global_model_moon = None
    prev_model_moon = None

    if moon_mu > 0:
        global_model_moon = copy.deepcopy(model)
        global_model_moon.eval()
        global_model_moon.to(device)
        for param in global_model_moon.parameters():
            param.requires_grad = False 
        prev_model_moon = global_model_moon 
        contrastive_criterion = ModelContrastiveLoss(temperature=moon_temp)

    # --- OPTIMIZER ---
    base_lr = config.LEARNING_RATE
    if current_round >= 30: base_lr *= 0.1
    if current_round >= 40: base_lr *= 0.1
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=config.MOMENTUM)

    # --- TEACHER LOADING (With Memory Fix) ---
    teacher_model = None
    
    # MEMORY FIX: Only load if Alpha > 0. If not, force garbage collection.
    if criterion.alpha > 0: 
        teacher_model = copy.deepcopy(model)
        teacher_model.eval() 
        teacher_model.to(device)
        for param in teacher_model.parameters():
            param.requires_grad = False
    else:
        # ⚡ CRITICAL: Ensure no previous teacher lingers in memory
        teacher_model = None
        gc.collect()
        torch.cuda.empty_cache()
            
    # Handle FedProx Parameters
    global_params = []
    if criterion.mu > 0:
        if teacher_model:
            global_params = list(teacher_model.parameters())
        else:
            temp_model = copy.deepcopy(model)
            global_params = list(temp_model.parameters())

    scaler = torch.cuda.amp.GradScaler() 
    
    # --- TRAINING LOOP ---
    start_time = time.time()
    
    for epoch in range(config.LOCAL_EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # 1. Student Forward
                if moon_mu > 0:
                    student_logits, student_features = model(images, return_features=True)
                else:
                    student_logits = model(images)
                    student_features = None

                # 2. Teacher Forward (Only if Active)
                teacher_logits = None
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_model(images)
                
                # 3. MOON Loss
                loss_moon = 0.0
                if moon_mu > 0:
                    with torch.no_grad():
                        _, global_features = global_model_moon(images, return_features=True)
                        _, prev_features = prev_model_moon(images, return_features=True)
                    loss_moon = contrastive_criterion(student_features, global_features, prev_features)

                # 4. Standard Loss
                loss = criterion(
                    student_logits=student_logits, 
                    labels=labels, 
                    student_params=model.parameters(), 
                    global_params=global_params,
                    teacher_logits=teacher_logits
                )
                
                if moon_mu > 0:
                    loss += moon_mu * loss_moon
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    total_time = time.time() - start_time
    
    # Final Cleanup
    if teacher_model: del teacher_model
    if global_model_moon: del global_model_moon
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "loss": 0.0, 
        "accuracy": 0.0, 
        "training_time_sec": total_time
    }