import torch
import torch.nn as nn

class DeepClinicalNet(nn.Module):
    """
    Advanced Deep Learning Architecture for Tabular Clinical Data.
    Uses BatchNorm to stabilize training and Dropout to prevent overfitting.
    """
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Randomly drops 30% of neurons to prevent overfitting
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)