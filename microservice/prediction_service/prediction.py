import pandas as pd
import torch

from prediction_service.models import Prediction
from prediction_service.core.logger import get_logger


logger = get_logger(__name__)


def make_xgboost_predictions(model, processed_data: pd.DataFrame) -> list[Prediction]:
    preds = []
    for _, row in processed_data.iterrows():
        listing_id = row["id"]
        input_vector = row.drop("id").to_frame().T
        pred = model.predict(input_vector)
        preds.append(Prediction(listing_id=listing_id, prediction="short" if pred[0] == 0 else "long"))
    return preds


def make_naive_classifier_predictions(model, df: pd.DataFrame) -> list[Prediction]:
    preds = []
    model.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            listing_id = row.get("id", None)
            input_tensor = torch.tensor(row.drop("id").values, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            label = "short" if predicted_class == 0 else "long"
            preds.append(Prediction(listing_id=listing_id, prediction=label))

    return preds
