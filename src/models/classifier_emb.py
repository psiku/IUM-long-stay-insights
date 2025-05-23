import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifierEmbeddings(nn.Module):
    def __init__(self, num_numeric: int, embedding_sizes: list, output_size: int):
        super(BinaryClassifierEmbeddings, self).__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim)
            for num_categories, emb_dim in embedding_sizes
        ])

        emb_dim_total = sum(emb_dim for _, emb_dim in embedding_sizes)
        self.input_size = num_numeric + emb_dim_total

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        self.actv = F.relu

    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        if x_num.dim() == 1:
            x_num = x_num.unsqueeze(1)

        x = torch.cat(embedded + [x_num], dim=1)

        x = self.actv(self.fc1(x))
        x = self.actv(self.fc2(x))
        x = self.fc3(x)
        return x
