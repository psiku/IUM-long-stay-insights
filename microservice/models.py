from pydantic import BaseModel


class ListingInput(BaseModel):
    listing_id: int


class Prediction(BaseModel):
    listing_id: int
    prediction: str