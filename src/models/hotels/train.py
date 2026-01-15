import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class HotelModelTrainer:
    def __init__(self):
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, df: pd.DataFrame):
        X = df.drop(columns=["high_demand"])
        y = df["high_demand"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        model.fit(X_train, y_train)

        model_path = os.path.join(self.model_dir, "hotel_model_v1.pkl")
        joblib.dump(model, model_path)

        return model, X_test, y_test
