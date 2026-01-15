import joblib
import pandas as pd


class UserSegmenter:
    def __init__(self):
        self.model = joblib.load(
            "artifacts/models/user_clusters_v1.pkl"
        )

    def assign_segment(self, input_data: dict):
        df = pd.DataFrame([input_data])
        return int(self.model.predict(df)[0])
