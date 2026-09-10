"""
Tiny demo API. Train first (`python src/train.py`) so artifacts/models.joblib
exists. Then: `uvicorn app:app --app-dir src --reload`

Interview line: scores come from the fitted booster; the importance table is
why a prediction is not a black box. This is not a plant-facing product.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from train import FEATURE_COLS

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Predictive Maintenance (demo)")


class SensorReading(BaseModel):
    hours_run: float = Field(examples=[4200])
    vibration_mm_s: float = Field(examples=[3.1])
    temperature_c: float = Field(examples=[88])
    pressure_psi: float = Field(examples=[118])
    current_amps: float = Field(examples=[18])


def load_bundle() -> dict:
    path = ARTIFACTS / "models.joblib"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Train first: python src/train.py (local demo, not deployed).",
        )
    return joblib.load(path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/metrics")
def metrics() -> dict:
    path = ARTIFACTS / "metrics.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Train first: python src/train.py (local demo, not deployed).",
        )
    rows = pd.read_csv(path).to_dict(orient="records")
    return {"models": rows}


@app.post("/api/predict")
def predict(reading: SensorReading) -> dict:
    bundle = load_bundle()
    frame = pd.DataFrame([reading.model_dump()], columns=FEATURE_COLS)
    booster = bundle["booster"]
    probability = float(booster.predict_proba(frame)[0, 1])
    return {
        "failure_probability": round(probability, 4),
        "flagged": probability >= 0.5,
        "model": bundle["booster_name"],
    }
