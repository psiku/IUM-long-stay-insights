import pandas as pd
import torch
import json

from prediction_service.naive_classifier import NaiveClassifier


def load_torch_model(model_path: str, input_size: int):
    model = NaiveClassifier(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_normalized_map(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        normalized_map = json.load(file)
    return normalized_map