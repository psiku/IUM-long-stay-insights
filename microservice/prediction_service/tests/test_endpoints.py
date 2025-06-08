from unittest.mock import patch
from fastapi.testclient import TestClient
from prediction_service.main import app


def test_predict_with_xgboost(client: TestClient, listing_input: dict):
    response = client.post("/xgboost", json=[listing_input])
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "listing_id" in response.json()[0]
    assert "prediction" in response.json()[0]
    assert response.json()[0]["listing_id"] == listing_input["id"]
    assert response.json()[0]["prediction"] in ["short", "long"]


def test_predict_with_base_model(client: TestClient, listing_input: dict):
    response = client.post("/base-model", json=[listing_input])
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "listing_id" in response.json()[0]
    assert "prediction" in response.json()[0]
    assert response.json()[0]["listing_id"] == listing_input["id"]
    assert response.json()[0]["prediction"] in ["short", "long"]


@patch("prediction_service.main.random.choice", return_value="xgboost")
def test_AB_xgboost(mock_choice, client: TestClient, listing_input: dict):
    response = client.post("/AB-test", json=[listing_input])

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert "listing_id" in result[0]
    assert "prediction" in result[0]
    assert result[0]["listing_id"] == listing_input["id"]
    assert result[0]["prediction"] in ["short", "long"]


@patch("prediction_service.main.random.choice", return_value="base")
def test_AB_base(mock_choice, client: TestClient, listing_input: dict):
    response = client.post("/AB-test", json=[listing_input])

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert "listing_id" in result[0]
    assert "prediction" in result[0]
    assert result[0]["listing_id"] == listing_input["id"]
    assert result[0]["prediction"] in ["short", "long"]