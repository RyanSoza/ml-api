"""API contract"""

import pandas as pd
from pydantic import BaseModel, Field

# API field names
COLUMN_FOR = {
    "median_income": "MedInc",
    "house_age": "HouseAge",
    "avg_rooms": "AveRooms",
    "avg_bedrooms": "AveBedrms",
    "population": "Population",
    "avg_occupancy": "AveOccup",
    "latitude": "Latitude",
    "longitude": "Longitude",
}


class PredictRequest(BaseModel):
    median_income: float = Field(ge=0, le=20, description="Block median income, in $10,000s")
    house_age: float = Field(ge=0, le=100, description="Median house age in years")
    avg_rooms: float = Field(ge=0, le=150, description="Average number of rooms per household")
    avg_bedrooms: float = Field(ge=0, le=38, description="Average number of bedrooms per household")
    population: float = Field(ge=0, le=40000000, description="Block group population")
    avg_occupancy: float = Field(ge=0, le=1400, description="Average number of household members")
    latitude: float = Field(ge=30, le=43, description="Block group latitude")
    longitude: float = Field(ge=-126, le=-113, description="Block group longitude")

    def to_frame(self) -> pd.DataFrame:
        """One-row Dataframe with the columns names"""
        return pd.DataFrame([self.model_dump()]).rename(columns=COLUMN_FOR)


class PredictResponse(BaseModel):
    prediction: float = Field(description="Median house value, in $100,000s")
    version: str
