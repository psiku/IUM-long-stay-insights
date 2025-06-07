from itertools import combinations
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def permutation_importance_dataloader_advanced(
    model,
    dataloader,
    emb_categorical_columns,
    numeric_columns,
    baseline_score,
    device,
    max_group_size=1
):
    model.eval()
    importances = []

    feature_info = [('cat', i, col) for i, col in enumerate(emb_categorical_columns)] + \
                   [('num', i, col) for i, col in enumerate(numeric_columns)]

    for k in range(1, max_group_size + 1):
        for group in combinations(feature_info, k):
            y_true = []
            y_pred = []
            group_names = [feat[2] for feat in group]

            for x_cat_batch, x_num_batch, y_batch in tqdm(dataloader, desc=f"Permuting {group_names}", leave=False):
                x_cat_batch = x_cat_batch.clone()
                x_num_batch = x_num_batch.clone()

                for feat_type, idx, _ in group:
                    if feat_type == 'cat':
                        perm = x_cat_batch[:, idx][torch.randperm(x_cat_batch.size(0))]
                        x_cat_batch[:, idx] = perm
                    else:
                        perm = x_num_batch[:, idx][torch.randperm(x_num_batch.size(0))]
                        x_num_batch[:, idx] = perm

                x_cat_batch = x_cat_batch.to(device)
                x_num_batch = x_num_batch.to(device)
                y_batch = y_batch.to(device)

                with torch.no_grad():
                    outputs = model(x_cat_batch, x_num_batch)
                    preds = torch.argmax(outputs, dim=1)

                y_true.append(y_batch.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            permuted_score = roc_auc_score(y_true, y_pred)
            importance = baseline_score - permuted_score

            importances.append((group_names, importance))

    return sorted(importances, key=lambda x: x[1], reverse=True)