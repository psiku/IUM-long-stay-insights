from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    XGBOOST_MODEL_PATH: str = "microservice/trained_models/xgboost/xgboost_classifier.joblib"
    XGBOOST_DATA_PATH: str = "microservice/data/final_df_xgboost.csv"

settings = Settings()