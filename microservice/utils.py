import pandas as pd
import torch
import json
from naive_classifier import NaiveClassifier


def get_rows_by_ids(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    return df[df['id'].isin(ids)]


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def get_data_for_prediction(path_to_csv: str, ids: list) -> pd.DataFrame:
    df = load_data(path_to_csv)
    df = get_rows_by_ids(df, ids)
    return df


def load_torch_model(model_path: str, input_size: int):
    model = NaiveClassifier(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_normalized_map(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        normalized_map = json.load(file)
    return normalized_map