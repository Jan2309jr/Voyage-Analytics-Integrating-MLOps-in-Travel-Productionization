import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class GenderModelTrainer:
    def __init__(self):
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, df: pd.DataFrame):
        if "gender" not in df.columns:
            raise ValueError("Target column 'gender' not found")

        X = df.drop(columns=["gender"])
        y = df["gender"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        model.fit(X_train, y_train)

        joblib.dump(model, os.path.join(self.model_dir, "gender_model.pkl"))
        joblib.dump(le, os.path.join(self.model_dir, "gender_encoder.pkl"))

        return model, le, X_test, y_test
