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
    median_income: float = Field(gt=0, le=20, description="Block median income, in $10,000s")
    house_age: float = Field(gt=0, le=100, description="Median house age in years")
    # TODO Add field for other columns

    def to_frame(self) -> pd.DataFrame:
        """One-row Dataframe with the columns names"""
        return pd.DataFrame([self.model_dump()]).rename(columns=COLUMN_FOR)


class PredictResponse(BaseModel):
    predictions: float = Field(description="Median house value, in $100,000s")
    version: str
