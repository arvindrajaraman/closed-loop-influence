import torch
import torch.nn as nn
import torch.nn.functional as F

from device import device
from env_setup import *

class ThetaEstimatorMLP(nn.Module):
    def __init__(self):
        super(ThetaEstimatorMLP, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, input):
        output = input
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.gelu(self.fc3(output)) + 0.001
        return output

class ThetaEstimatorLargeResidualMLP(nn.Module):
    def __init__(self):
        super(ThetaEstimatorLargeResidualMLP, self).__init__()
        self.fc1 = nn.Linear(3, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, input):
        output = input
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output)) + output
        output = F.relu(self.fc3(output)) + output
        output = F.gelu(self.fc4(output)) + 0.001
        return output

class ThetaEstimatorTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = nX + nX + nU
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=5, dropout=0.2, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        _, layers, _ = x.size()
        mask = torch.zeros(layers, layers)
        for i in range(layers):
            for j in range(i+1, layers):
                mask[i, j] = float('-inf')
        mask = mask.to(device).double()

        y = self.transformer_encoder(x, src_mask=mask)
        y = self.fc_layers(y)
        return y
