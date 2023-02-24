import torch.nn as nn
import torch.nn.functional as F

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
