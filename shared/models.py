import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 (3 channels) with Group Normalization.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.gn1 = nn.GroupNorm(2, 6) 
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(4, 16)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, return_features=False):
        # Block 1
        x = self.conv1(x)
        x = self.gn1(x) 
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1) 
        
        # FC 1
        x = F.relu(self.fc1(x))
        
        # FC 2
        x = F.relu(self.fc2(x))

        # --- UPDATE FOR MOON ---
        # Capture features HERE (vector of size 84), not earlier.
        features = x 

        # Final Classification
        logits = self.fc3(x)
        
        if return_features:
            return logits, features
        return logits

class SimpleMLP(nn.Module):
    """
    A simple MLP for flattened inputs.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # This was already correct!
        features = x

        logits = self.fc3(x)
        
        if return_features:
            return logits, features
        return logits

# --- Model Registry ---
def get_model(model_name):
    if model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "SimpleMLP":
        return SimpleMLP()
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")