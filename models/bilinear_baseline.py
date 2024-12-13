# Gene2Conn/models/bilinear_baseline.py

from imports import *
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.metrics.eval import mse_cupy

class BilinearRegressionModel(nn.Module):
    """Basic bilinear model without activation function."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x1, x2):
        # Project both inputs to lower dimensional space
        out1 = self.linear(x1)   # Shape: [batch_size, output_size]
        out2 = self.linear2(x2)  # Shape: [batch_size, output_size]
        # Compute similarity between projections
        return torch.matmul(out1, out2.T)  # Shape: [batch_size, batch_size]

class BilinearSigmoidRegressionModel(nn.Module):
    """Bilinear model with sigmoid activation for bounded output."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Project and apply sigmoid to constrain outputs between 0 and 1
        out1 = self.sigmoid(self.linear(x1))
        out2 = self.sigmoid(self.linear2(x2))
        return torch.matmul(out1, out2.T)

class BilinearReLURegressionModel(nn.Module):
    """Bilinear model with ReLU activation for non-negative outputs."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        # Project and apply ReLU to ensure non-negative outputs
        out1 = self.relu(self.linear(x1))
        out2 = self.relu(self.linear2(x2))
        return torch.matmul(out1, out2.T)

class BilinearSoftplusModel(nn.Module):
    """Bilinear model with Softplus activation for smooth non-negative outputs."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, x1, x2):
        # Project and apply Softplus for smooth, non-negative outputs
        out1 = self.softplus(self.linear(x1))
        out2 = self.softplus(self.linear2(x2))
        return torch.matmul(out1, out2.T)