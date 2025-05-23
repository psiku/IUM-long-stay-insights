import pandas as pd


def get_rows_by_ids(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    """
    Load rows from a DataFrame by their IDs.

    Args:
        df (pd.DataFrame): The DataFrame to load rows from.
        ids (list): A list of IDs to load.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded rows.
    """
    return df[df['id'].isin(ids)]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def get_data_for_prediction(path_to_csv: str, ids: list) -> pd.DataFrame:
    """
    Load data for prediction from a CSV file and filter by IDs.

    Args:
        path_to_csv (str): The path to the CSV file.
        ids (list): A list of IDs to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered data.
    """
    df = load_data(path_to_csv)
    df = get_rows_by_ids(df, ids)
    return df