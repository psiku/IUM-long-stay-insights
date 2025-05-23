import pandas as pd
from joblib import load

from models import Prediction
from config import settings


def make_xgboost_predictions(df: pd.DataFrame) -> list[Prediction]:
    preds = []
    model = load(settings.XGBOOST_MODEL_PATH)
    for index, row in df.iterrows():
        id = row["id"]
        row = row.drop(["id"])
        pred = model.predict(row.values.reshape(1, -1))
        preds.append(Prediction(listing_id=id, prediction="short" if pred[0] == 0 else "long"))
    return preds
