import joblib
import pandas as pd


class GenderPredictor:
    def __init__(self):
        self.model = joblib.load("artifacts/models/gender_model.pkl")
        self.encoder = joblib.load("artifacts/models/gender_encoder.pkl")

    def predict(self, data: dict):
        df = pd.DataFrame([data])
        pred = self.model.predict(df)[0]
        return self.encoder.inverse_transform([pred])[0]
