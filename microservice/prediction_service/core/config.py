from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    XGBOOST_MODEL_PATH: str = "microservice/trained_models/xgboost/xgboost_classifier.joblib"
    XGBOOST_DATA_PATH: str = "microservice/data/final_df_xgboost.csv"

    NAIVE_CLASSIFIER_MODEL_PATH: str = "microservice/trained_models/torch/NaiveClassifier.pth"
    NORMALIZED_MAP_PATH: str = "microservice/data/amenities_normalized_map.json"
    LOGS_DIR: str = "microservice/AB_experiments/logs"

settings = Settings()