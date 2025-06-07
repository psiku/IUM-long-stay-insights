import pandas as pd
import numpy as np
from src.data.utils import create_dataframe_from_csv_zst


AMENITIES_PATH = "../data/processed/amenities_stats.csv.zst"
REVIEWS_STATISTICS_PATH = "../data/processed/merged_reviews_statistics.csv.zst"
LISTINGS_PATH = "../data/raw/v2/listings.csv.zst"
SESSIONS_STATISTICS_PATH = "../data/processed/listings_statistics.csv"


interesting_columns = [
    'id',
    'host_is_superhost',
    'host_verifications',
    'host_acceptance_rate',
    'neighbourhood_cleansed',
    'latitude',
    'longitude',
    'property_type',
    'room_type',
    'accommodates',
    'bathrooms',
    'bathrooms_text',
    'bedrooms',
    'beds',
    'price',
    'license',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'instant_bookable',
    'reviews_per_month',
    'availability_eoy',
    'number_of_reviews_ly'
]

amenities_df = create_dataframe_from_csv_zst(AMENITIES_PATH)
reviews_statistics_df = create_dataframe_from_csv_zst(REVIEWS_STATISTICS_PATH)
listings_df = create_dataframe_from_csv_zst(LISTINGS_PATH)

listings_statistics = pd.read_csv(SESSIONS_STATISTICS_PATH)


def create_final_dataset():

    listings_statistics['target'] = np.where(
        listings_statistics['num_of_short_stays'] > listings_statistics['num_of_long_stays'],
        'short',
        'long'
    )

    listing_statistics_final_df = listings_statistics[['listing_id','total_bookings', 'target']].copy()

    final_df = listings_df[interesting_columns].copy()

    final_df = final_df.merge(
        listing_statistics_final_df,
        left_on='id',
        right_on='listing_id',
        how='left'
    )

    final_df= final_df.merge(
        reviews_statistics_df,
        left_on='id',
        right_on='listing_id',
        how='left'
    )

    final_df = final_df.merge(
        amenities_df,
        left_on='id',
        right_on='id',
        how='left'
    )

    final_df = final_df.dropna(subset=['target'])

    final_df['price'] = final_df['price'].replace('[\$,]', '', regex=True).astype(float)

    final_df.drop(columns=['listing_id_x', 'listing_id_y', 'mean_embedding'], inplace=True)

    final_df['host_acceptance_rate'] = final_df['host_acceptance_rate'].replace('[\%,]', '', regex=True).astype(float)

    return final_df
