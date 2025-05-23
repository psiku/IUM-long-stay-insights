import pandas as pd


def get_rows_by_ids(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    return df[df['id'].isin(ids)]


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def get_data_for_prediction(path_to_csv: str, ids: list) -> pd.DataFrame:
    df = load_data(path_to_csv)
    df = get_rows_by_ids(df, ids)
    return df
