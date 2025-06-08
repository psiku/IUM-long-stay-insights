from pydantic import BaseModel
from typing import Optional


class ListingInput(BaseModel):
    id: int
    host_acceptance_rate: Optional[float]
    latitude: float
    longitude: float
    accommodates: int
    bathrooms: Optional[float]
    bedrooms: Optional[float]
    beds: Optional[float]
    price: float
    review_scores_rating: Optional[float]
    review_scores_accuracy: Optional[float]
    review_scores_cleanliness: Optional[float]
    review_scores_checkin: Optional[float]
    review_scores_communication: Optional[float]
    review_scores_location: Optional[float]
    review_scores_value: Optional[float]
    reviews_per_month: Optional[float]
    total_bookings: Optional[int]
    total_reviews: Optional[int]
    total_english_reviews: Optional[int]
    count_negative_english: Optional[int]
    count_positive_english: Optional[int]
    amenities: Optional[str]
    host_is_superhost: Optional[bool]
    host_verifications: Optional[str]
    property_type: Optional[str]
    room_type: Optional[str]
    bathrooms_text: Optional[str]
    instant_bookable: Optional[str]


class Prediction(BaseModel):
    listing_id: int
    prediction: str