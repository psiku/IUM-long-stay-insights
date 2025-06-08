import pytest
from fastapi.testclient import TestClient
from prediction_service.main import app


@pytest.fixture(scope="session")
def listing_input():
    return {
        "id": 27262,
        "host_acceptance_rate": 100.0,
        "latitude": 37.98924,
        "longitude": 23.765,
        "accommodates": 2,
        "bathrooms": 1.0,
        "bedrooms": 1.0,
        "beds": 1.0,
        "price": 131.0,
        "review_scores_rating": 4.86,
        "review_scores_accuracy": 4.89,
        "review_scores_cleanliness": 4.9,
        "review_scores_checkin": 4.86,
        "review_scores_communication": 4.97,
        "review_scores_location": 4.75,
        "review_scores_value": 4.71,
        "reviews_per_month": 0.19,
        "total_bookings": 29,
        "total_reviews": 29,
        "total_english_reviews": 17,
        "count_negative_english": 0,
        "count_positive_english": 17,
        "amenities": "Wifi",
        "host_is_superhost_f": False,
        "host_is_superhost_t": True,
        "host_verifications": "Email",
        "property_type": "Earthen home",
        "room_type": "Entire home/apt",
        "bathrooms_text": "1.5 baths",
        "instant_bookable": "f"
  }


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as client:
        yield client