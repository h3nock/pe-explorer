import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # fully connected layers
        self.fc1 = nn.Linear(d_model, d_ff) 
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass of the MLP module."""
        x = self.fc1(x)
        x = F.gelu(x) # gelu used by gpt2, gpt3 
        x = self.dropout(x)
        x = self.fc2(x)
        return x
