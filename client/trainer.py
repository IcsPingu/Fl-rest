import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from shared.models import get_model
def get_device(device_config):
    if device_config == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_config == "cpu":
        return torch.device("cpu")
    else:
        # "auto"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. A NOVA CLASSE DE LOSS HÍBRIDA (DoubleReg) ---
class DoubleRegLoss(nn.Module):
    def __init__(self, mu=0.01, alpha=0.5, temperature=3.0, device="cpu"):
        super(DoubleRegLoss, self).__init__()
        self.mu = mu          # Peso do FedProx
        self.alpha = alpha    # Peso do KD
        self.T = temperature  # Temperatura do KD
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, labels, student_params, global_params, teacher_logits):
        # A. Task Loss (Sua CrossEntropy padrão)
        loss_task = self.ce_loss(student_logits, labels)

        # B. FedProx Term (Distância L2 dos pesos)
        loss_prox = 0.0
        if self.mu > 0:
            # Calcula a norma L2 entre pesos locais e globais
            prox_term = 0.0
            for s_p, g_p in zip(student_params, global_params):
                prox_term += (s_p - g_p).norm(2)**2
            loss_prox = (self.mu / 2) * prox_term

        # C. KD Term (Divergência KL)
        loss_kd = 0.0
        if self.alpha > 0 and teacher_logits is not None:
            # Suaviza as distribuições com Temperatura T
            soft_student = F.log_softmax(student_logits / self.T, dim=1)
            soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
            
            # KL Divergence: O quanto o aluno difere do professor
            loss_kd = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T**2)

        # Soma final: Tarefa + Proximal + (Alpha * Destilação)
        return loss_task + loss_prox + (self.alpha * loss_kd)

# --- 2. FUNÇÃO DE TREINO ATUALIZADA ---
def train_model(client_id, model, train_loader, config):
    device = get_device(config.DEVICE)
    model.to(device)
    model.train()

    # Configuração dos Otimizadores
    criterion_params = {
        'mu': getattr(config, 'FEDPROX_MU', 0.0),      # Pega do config ou usa 0
        'alpha': getattr(config, 'KD_ALPHA', 0.0),     # Pega do config ou usa 0
        'temperature': getattr(config, 'KD_TEMPERATURE', 3.0),
        'device': device
    }
    
    # Instancia a Loss Híbrida
    criterion = DoubleRegLoss(**criterion_params)
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=config.LEARNING_RATE, 
                          momentum=config.MOMENTUM)

    # --- SETUP DO PROFESSOR (SCENARIO A/C) ---
    teacher_model = None
    global_params = []
    
    # Se KD ou FedProx estiverem ativos, precisamos de uma cópia do Global
    use_teacher = config.ENABLE_PARAMETER_BASED_KD or (config.FEDPROX_MU > 0)
    
    if use_teacher:
        # Cria uma cópia congelada do modelo atual (que acabou de chegar do servidor)
        # Isso serve tanto como referência para FedProx quanto como Professor para KD
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Guarda os parâmetros globais em uma lista para acesso rápido no loop
        global_params = list(teacher_model.parameters())

    print(f"[{client_id}] Iniciando Treino Híbrido. FedProx={config.FEDPROX_MU}, KD={config.ENABLE_PARAMETER_BASED_KD}")
    
    epoch_times = []
    
    for epoch in range(config.LOCAL_EPOCHS):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 1. Forward do Aluno
            student_logits = model(images)
            
            # 2. Forward do Professor (se necessário)
            teacher_logits = None
            if config.ENABLE_PARAMETER_BASED_KD and teacher_model:
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
            
            # 3. Cálculo da Loss Híbrida
            # Passamos os parâmetros atuais (model.parameters) e os globais (global_params)
            loss = criterion(
                student_logits=student_logits, 
                labels=labels, 
                student_params=model.parameters(), 
                global_params=global_params, 
                teacher_logits=teacher_logits
            )
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        end_time = time.time()
        epoch_times.append(end_time - start_time)
        
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "training_time_sec": sum(epoch_times) # Tempo total
    }