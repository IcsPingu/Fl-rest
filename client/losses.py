import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleRegLoss(nn.Module):
    def __init__(self, mu, alpha, temperature=3.0, confidence_threshold=0.0, kd_type='logits', device='cpu'):
        super(DoubleRegLoss, self).__init__()
        self.mu = mu               # Peso do FedProx
        self.alpha = alpha         # Peso da Destilação (KD)
        self.T = temperature       # Temperatura para KD
        self.device = device
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, labels, student_params, global_params, teacher_logits=None):
        # 1. Perda Padrão (Cross Entropy)
        loss = self.ce_criterion(student_logits, labels)

        # 2. Regularização Proximal (FedProx)
        # Penaliza se os pesos do aluno (student_params) se afastarem do global (global_params)
        if self.mu > 0 and global_params is not None:
            proximal_term = 0.0
            # Itera sobre os parâmetros do modelo (Student) e do Global (Lista de tensores)
            for w, w_t in zip(student_params, global_params):
                # w_t precisa estar no mesmo device que w
                w_t = w_t.to(self.device)
                proximal_term += (w - w_t).norm(2) ** 2
            
            loss += (self.mu / 2) * proximal_term

        # 3. Regularização Semântica (Knowledge Distillation) - ASTRA
        if self.alpha > 0 and teacher_logits is not None:
            # Softmax com temperatura
            student_soft = F.log_softmax(student_logits / self.T, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            
            # KL Divergence
            kd_loss = self.kl_div(student_soft, teacher_soft) * (self.T ** 2)
            
            loss += self.alpha * kd_loss

        return loss