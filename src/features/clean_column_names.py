def clean_column_names(df):
    df_cleaned = df.copy()

    new_columns = {}
    for col in df_cleaned.columns:
        if any(char in str(col) for char in ['[', ']', '<', '>']):
            new_col = str(col)
            if '[]' in new_col:
                new_col = new_col.replace('[]', '')
            else:
                new_col = new_col.replace('[', '_').replace(']', '_').replace("'", "").replace(', ', '_')

            while '__' in new_col:
                new_col = new_col.replace('__', '_')

            new_col = new_col.rstrip('_')

            new_columns[col] = new_col

    df_cleaned = df_cleaned.rename(columns=new_columns)
    return df_cleaned

