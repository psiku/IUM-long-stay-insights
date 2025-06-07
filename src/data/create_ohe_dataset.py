import pandas as pd

def create_ohe_dataset(
    df: pd.DataFrame,
    drop_cols: list = None,
    drop_post_ohe: list = None,
    target_col: str = 'target',
    fillna_value=0
) -> pd.DataFrame:

    df = df.copy()

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')


    non_num = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in non_num:
        non_num.remove(target_col)

    df = pd.get_dummies(df, columns=non_num, drop_first=False)


    if drop_post_ohe:
        df.drop(columns=drop_post_ohe, inplace=True, errors='ignore')

    df.fillna(fillna_value, inplace=True)
    return df
