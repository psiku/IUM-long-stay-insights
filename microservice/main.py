from fastapi import FastAPI, HTTPException
import numpy as np

from models import ListingInput, Prediction
from utils import get_data_for_prediction
from prediction import make_xgboost_predictions
from config import settings


app = FastAPI(prefix="/predict", tags=["predict"])


@app.post("/xgboost", response_model=list[Prediction])
async def predict_with_XGBoost(linsting_inputs: list[ListingInput]):
    ids = [listing_input.listing_id for listing_input in linsting_inputs]
    df = get_data_for_prediction(settings.XGBOOST_DATA_PATH, ids)
    predictions = make_xgboost_predictions(df) 
    return predictions


@app.post("/xgboost/health", response_model=str)
async def health_check():
    try:
        model = load(settings.XGBOOST_MODEL_PATH)
        return "Model is loaded and ready for predictions."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
