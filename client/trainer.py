import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import re  # Added missing import
from shared.models import get_model
from torch.cuda.amp import autocast, GradScaler # Import at top

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

    # --- 1. Client Adaptivity ---
    try:
        c_num = int(re.search(r'\d+', client_id).group())
    except:
        c_num = 999 

    limit_high_perf = getattr(config, 'CLIENTS_HIGH_PERF', 4)
    is_high_perf = (c_num <= limit_high_perf)

    if is_high_perf:
        actual_epochs = config.LOCAL_EPOCHS + 2 
    else:
        actual_epochs = max(1, config.LOCAL_EPOCHS - 1) 

    # --- 2. THE GRADUATION STRATEGY (Linear Decay) ---
    current_round = getattr(config, 'CURRENT_ROUND', 0)
    global_rounds = getattr(config, 'GLOBAL_ROUNDS', 50)
    
    # Decays from 0.2 down to 0.0
    initial_alpha = 0.2
    decayed_alpha = initial_alpha * (1 - (current_round / global_rounds))
    decayed_alpha = max(0.0, decayed_alpha)

    mu_val = getattr(config, 'FEDPROX_MU', 0.01)
    
    criterion = DoubleRegLoss(
        mu=mu_val,
        alpha=decayed_alpha,
        temperature=3.0,
        confidence_threshold=0.0,
        kd_type='logits',
        device=device
    )
    
    # --- 3. LR Scheduler ---
    base_lr = config.LEARNING_RATE
    if current_round >= 30:
        base_lr = base_lr * 0.1
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=base_lr, 
                          momentum=config.MOMENTUM)

    # --- 4. Setup Teacher ---
    teacher_model = None
    # ALWAYS load teacher for this test, even if Alpha is 0, so we can measure accuracy
    if criterion.alpha > 0 or criterion.mu > 0 or True: 
        teacher_model = copy.deepcopy(model)
        teacher_model.eval() 
        teacher_model.to(device)
        for param in teacher_model.parameters():
            param.requires_grad = False
            
    global_params = []
    if criterion.mu > 0 and teacher_model:
        global_params = list(teacher_model.parameters())

    scaler = torch.cuda.amp.GradScaler() 
    epoch_times = []
    
    # Metrics for the "Truth Test"
    student_correct_total = 0
    teacher_correct_total = 0
    total_samples = 0
    
    # --- TRAINING LOOP ---
    for epoch in range(actual_epochs):
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Always run teacher to check its accuracy
                teacher_logits = None
                if teacher_model:
                    with torch.no_grad():
                        teacher_logits = teacher_model(images)

                student_logits = model(images)
                
                loss = criterion(
                    student_logits=student_logits, 
                    labels=labels, 
                    student_params=model.parameters(), 
                    global_params=global_params,
                    teacher_logits=teacher_logits, 
                    student_features=None, 
                    teacher_features=None
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # --- MEASURE ACCURACY (Student vs Teacher) ---
            total_samples += labels.size(0)
            
            # Student Accuracy
            _, pred_student = torch.max(student_logits.data, 1)
            student_correct_total += (pred_student == labels).sum().item()
            
            # Teacher Accuracy
            if teacher_logits is not None:
                _, pred_teacher = torch.max(teacher_logits.data, 1)
                teacher_correct_total += (pred_teacher == labels).sum().item()
            
        end_time = time.time()
        epoch_times.append(end_time - start_time)

    # --- FINAL REPORT ---
    student_acc = 100 * student_correct_total / total_samples
    teacher_acc = 100 * teacher_correct_total / total_samples
    diff = student_acc - teacher_acc
    
    # Print the "Truth" to the console
    print(f"[{client_id}] R{current_round} Results: Student {student_acc:.2f}% | Teacher {teacher_acc:.2f}% | Diff: {diff:+.2f}%")

    if diff > 0:
        print(f"[{client_id}] ✅ STUDENT IS WINNING (Alpha Decay is Correct)")
    else:
        print(f"[{client_id}] ❌ TEACHER IS SMARTER (We need higher Alpha)")
        
    avg_loss = 0.0 # Placeholder
    
    return {
        "loss": avg_loss,
        "accuracy": student_acc,
        "training_time_sec": sum(epoch_times)
    }