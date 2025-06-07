import pytest
from httpx import AsyncClient
from main import app


sample_input = {
    "id": 1,
    "host_acceptance_rate": 100,
    "latitude": 37.7749,
    "longitude": -122.4194,
    "accommodates": 2,
    "bathrooms": 1,
    "bedrooms": 1,
    "beds": 1,
    "price": 120,
    "review_scores_rating": 95,
    "review_scores_accuracy": 9,
    "review_scores_cleanliness": 9,
    "review_scores_checkin": 10,
    "review_scores_communication": 10,
    "review_scores_location": 9,
    "review_scores_value": 8,
    "reviews_per_month": 0.3,
    "total_bookings": 5,
    "total_reviews": 10,
    "total_english_reviews": 10,
    "count_negative_english": 0,
    "count_positive_english": 10,
    "num_of_amenities": 5,
    "num_of_other_amenities": 1,
    "has_wifi": 1,
    "has_air_conditioning": 1,
    "num_of_top_10_common_amenities": 3,
    "host_is_superhost_f": 0,
    "host_is_superhost_t": 1,
    "amenities": "['Wifi', 'Air conditioning', 'TV']"
}


@pytest.mark.asyncio
async def test_predict_with_xgboost():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/xgboost", json=[sample_input])

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "listing_id" in response.json()[0]
    assert "prediction" in response.json()[0]


@pytest.mark.asyncio
async def test_predict_with_base_model():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/base-model", json=[sample_input])

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "listing_id" in response.json()[0]
    assert "prediction" in response.json()[0]
