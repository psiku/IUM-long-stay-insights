import pandas as pd

import re
import ast
import unicodedata


def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = re.sub(r'\s+', ' ', text)
    return text if len(text) >= 2 else None


def process_amenities(df: pd.DataFrame, normalized_map: dict, top_amenities: list[str]) -> pd.DataFrame:
    df = df.copy()

    def standardize_amenities(raw: str):
        try:
            items = ast.literal_eval(raw)
            cleaned = [clean_text(i) for i in items if clean_text(i)]
            mapped = [normalized_map.get(i, "OTHER") for i in cleaned]
            return mapped
        except Exception:
            return []

    df["standardized_amenities"] = df["amenities"].apply(standardize_amenities)
    df["num_of_amenities"] = df["standardized_amenities"].apply(len)
    df["num_of_other_amenities"] = df["standardized_amenities"].apply(lambda x: sum(1 for i in x if i == "OTHER"))
    df["num_of_top_10_common_amenities"] = df["standardized_amenities"].apply(
        lambda x: sum(1 for i in x if i in top_amenities)
    )
    df["has_wifi"] = df["standardized_amenities"].apply(lambda x: int("Wifi" in x))
    df["has_air_conditioning"] = df["standardized_amenities"].apply(lambda x: int("Air conditioning" in x))

    return df


def process_input_data(listing_input: pd.DataFrame, required_cols: list[str], amenities_map: dict) -> pd.DataFrame:
    top_amenities = [
        "Wifi", "Air conditioning", "Coffe maker", "Heating", "Washer", "Refrigerator",
        "TV", "Drying rack for clothing", "Shampoo"
    ]

    listing_input = process_amenities(listing_input, amenities_map, top_amenities)

    listing_input = listing_input.drop(columns=["listing_id", "amenities", "standardized_amenities"], errors='ignore')
    
    non_numerical_columns = listing_input.select_dtypes(include=['object', 'category']).columns
    final_df = pd.get_dummies(listing_input, columns=non_numerical_columns)
    final_df = final_df.fillna(0)

    missing_cols = [col for col in required_cols if col not in final_df.columns]
    
    if missing_cols:
        zeros_df = pd.DataFrame(0, index=final_df.index, columns=missing_cols)
        final_df = pd.concat([final_df, zeros_df], axis=1)

    final_df = final_df[required_cols]

    final_df = final_df.astype({col: 'float' for col in final_df.columns})

    return final_df

    
