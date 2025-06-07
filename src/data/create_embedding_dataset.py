import pandas as pd

def create_embeddings_dataset(
    df: pd.DataFrame,
    drop_cols: list = None,
    target_col: str = 'target',
    fillna_value: float = 0.0
):

    df_emb = df.copy()
    if drop_cols is None:
        drop_cols = [
            'availability_eoy',
            'number_of_reviews_ly',
            'id',
            'amenities',
            'standardized_amenities'
        ]
    df_emb = df_emb.drop(columns=drop_cols, errors='ignore')

    df_emb = df_emb.fillna(fillna_value)

    return df_emb

def get_categorical_and_numerical_columns(df: pd.DataFrame, target_col: str = 'target'):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    return cat_cols, num_cols


def get_embedding_sizes(embeddings: pd.DataFrame, emb_categorical_columns: list):
    embedding_sizes = [
    (len(embeddings[col].astype('category').cat.categories), min(50, (len(embeddings[col].astype('category').cat.categories) + 1) // 2))
    for col in emb_categorical_columns
    ]

    return embedding_sizes