import joblib
import pandas as pd


class FlightPricePredictor:
    def __init__(self):
        self.model = joblib.load(
            "artifacts/models/flight_model_v1.pkl"
        )

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        return self.model.predict(df)[0]
