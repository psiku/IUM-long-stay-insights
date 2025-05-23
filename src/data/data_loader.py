import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def train_test_split_data(
    df: pd.DataFrame,
    emb_categorical_columns: list = None,
    test_size: float = 0.2,
    target_column: str = 'target'
):
    df = df.copy()
    y = df[target_column].map({'short': 0, 'long': 1})

    if emb_categorical_columns:
        # Ensure proper category dtype
        for col in emb_categorical_columns:
            df[col] = df[col].astype('category')

        # Extract categorical features as codes
        x_cat = df[emb_categorical_columns].apply(lambda col: col.cat.codes)

        # Extract numerical features correctly
        numeric_cols = df.columns.difference(emb_categorical_columns + [target_column])
        x_num = df[numeric_cols].astype(np.float32)

        # Final split
        x_cat_train, x_cat_test, x_num_train, x_num_test, y_train, y_test = train_test_split(
            x_cat, x_num, y, test_size=test_size, random_state=42, stratify=y
        )

        return x_cat_train, x_cat_test, x_num_train, x_num_test, y_train, y_test

    else:
        X = df.drop(columns=[target_column]).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test


# x-> categorical features or all features if we are not using embeddings
def create_dataloader(
    x,
    y: pd.Series,
    x_num=None,
    batch_size: int = 32,
    shuffle: bool = True
):
    if x_num is not None:
        x_cat_tensor = torch.tensor(x.values, dtype=torch.long)
        x_num_tensor = torch.tensor(x_num.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        dataset = TensorDataset(x_cat_tensor, x_num_tensor, y_tensor)
    else:
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
