from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
import logging

from contextlib import asynccontextmanager

from models import ListingInput, Prediction
from utils import load_torch_model, load_normalized_map
from prediction import make_xgboost_predictions, make_naive_classifier_predictions
from config import settings
from proccesing import process_input_data



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.xgboost_model = load(settings.XGBOOST_MODEL_PATH)
        df = pd.read_csv(settings.XGBOOST_DATA_PATH, nrows=1)
        app.state.xgboost_required_columns = df.columns.tolist()
        app.state.naive_classifier = load_torch_model(
            settings.NAIVE_CLASSIFIER_MODEL_PATH, input_size=len(df.columns.to_list()) - 1
        )
        app.state.normalized_map = load_normalized_map(settings.NORMALIZED_MAP_PATH)
        logging.info("Models loaded successfully.")
        yield
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


app = FastAPI(lifespan=lifespan, prefix="/predict", tags=["predict"])


@app.post("/xgboost", response_model=list[Prediction])
async def predict_with_XGBoost(linsting_inputs: list[ListingInput]):
    processed_data = process_input_data(
        pd.DataFrame([listing.model_dump() for listing in linsting_inputs]),
        app.state.xgboost_required_columns,
        app.state.normalized_map
)
    predictions = make_xgboost_predictions(processed_data=processed_data, model=app.state.xgboost_model) 
    return predictions


@app.post("/base-model", response_model=list[Prediction])
async def predict_with_base_model(linsting_inputs: list[ListingInput]):
    processed_data = process_input_data(
        pd.DataFrame([listing.model_dump() for listing in linsting_inputs]),
        app.state.xgboost_required_columns,
        app.state.normalized_map
    )
    predictions = make_naive_classifier_predictions(
        model=app.state.naive_classifier, df=processed_data
    )
    return predictions

 