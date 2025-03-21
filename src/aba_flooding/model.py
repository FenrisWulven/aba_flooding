from torch import nn
import torch
import copy
from pathlib import Path
import warnings

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class TemporalFusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Implement the Temporal Fusion Transformer here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement the forward pass here
        pass



if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")



