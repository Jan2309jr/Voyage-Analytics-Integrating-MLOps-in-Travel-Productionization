import os
import pandas as pd
from src.models.flights.train import FlightModelTrainer


def test_flight_model_training():
    df = pd.read_csv("artifacts/processed/flights_processed.csv")

    trainer = FlightModelTrainer()
    model, _, _ = trainer.train(df)

    assert model is not None
    assert os.path.exists("artifacts/models/flight_model_v1.pkl")
