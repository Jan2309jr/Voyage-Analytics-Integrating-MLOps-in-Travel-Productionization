import joblib
import pandas as pd


class HotelDemandPredictor:
    def __init__(self):
        self.model = joblib.load(
            "artifacts/models/hotel_model_v1.pkl"
        )

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        return int(self.model.predict(df)[0])
