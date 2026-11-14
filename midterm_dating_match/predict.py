"""FastAPI inference service for two pre-date participant profiles."""

import os
import pickle
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from features import pair_features


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.bin")))
with MODEL_PATH.open("rb") as f:
    artifact = pickle.load(f)

app = FastAPI(title="Mixmosa mutual-interest prediction", version="1.0.0")


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100)
    height_inches: float | None = Field(default=None, gt=0)
    avg_physical_attractiveness_rating: float | None = Field(default=None, ge=1, le=10)
    agreeableness: float | None = None
    conscientiousness: float | None = None
    extraversion: float | None = None
    neuroticism: float | None = None
    openness: float | None = None
    behavioral_health_score: float | None = None
    get_along_well_with_family_score: float | None = None
    num_children_wanted_score: float | None = None


class PairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    male: Participant
    female: Participant


class Prediction(BaseModel):
    mutual_interest_probability: float
    mutual_interest_prediction: bool
    threshold: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(pair: PairRequest) -> Prediction:
    features = pair_features(pair.male.model_dump(), pair.female.model_dump())
    probability = float(artifact["model"].predict_proba(features)[0, 1])
    threshold = float(artifact["threshold"])
    return Prediction(
        mutual_interest_probability=probability,
        mutual_interest_prediction=probability >= threshold,
        threshold=threshold,
    )
