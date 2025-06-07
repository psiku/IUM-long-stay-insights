import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifierEmbeddings(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        embedding_sizes: list,
        output_size: int,
        hidden_sizes: list[int] = [128, 64],
        dropout: float = 0.0
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in embedding_sizes
        ])
        emb_dim_total = sum(emb_dim for _, emb_dim in embedding_sizes)
        in_features = num_numeric + emb_dim_total

        # dynamiczna sieć wg hidden_sizes
        layers: list[nn.Module] = []
        prev = in_features
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev, hs))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hs
        layers.append(nn.Linear(prev, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embedded = torch.cat([
            emb(x_cat[:, i])
            for i, emb in enumerate(self.embeddings)
        ] + [x_num], dim=1)
        return self.net(embedded)
