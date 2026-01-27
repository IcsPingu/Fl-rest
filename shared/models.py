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
        # GroupNorm(num_groups, num_channels)
        # We split 6 channels into 2 groups (3 channels each)
        self.gn1 = nn.GroupNorm(2, 6) 
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        # We split 16 channels into 4 groups (4 channels each)
        self.gn2 = nn.GroupNorm(4, 16)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, return_features=False):
        # Block 1: Conv -> Norm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.gn1(x) 
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2: Conv -> Norm -> ReLU -> Pool
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # --- CAPTURE FEATURES ---
        features = x.clone()

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        if return_features:
            return x, features
        return x

class SimpleMLP(nn.Module):
    """
    A simple MLP for flattened inputs.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512) # CIFAR is 32x32x3
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        
        # Capture features after the second hidden layer
        x = F.relu(self.fc2(x))
        features = x.clone()

        x = self.fc3(x)
        
        if return_features:
            return x, features
        return x

# --- Model Registry ---
def get_model(model_name):
    """Factory function to instantiate models by name."""
    if model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "SimpleMLP":
        return SimpleMLP()
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")