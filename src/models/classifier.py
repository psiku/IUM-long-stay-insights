import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(BinaryClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        self.actv = F.relu

    def forward(self, x):
        x = self.actv(self.fc1(x))
        x = self.actv(self.fc2(x))
        x = self.fc3(x)

        return x
