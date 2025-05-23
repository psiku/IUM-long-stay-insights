import torch
import torch.nn as nn
import torch.nn.functional as F


# We assume that 0 will represent tha class that is more representative in the dataset
class NaiveClassifier(nn.Module):
    def __init__(self, input_size: int, output: int = 0):
        super(NaiveClassifier, self).__init__()
        self.input_size = input_size
        self.output = output


    def forward(self, x):
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, 2), dtype=torch.float32)
        logits[:, self.output] = 1.0
        return logits