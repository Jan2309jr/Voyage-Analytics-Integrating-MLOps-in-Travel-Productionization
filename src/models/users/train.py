import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class UserClusteringTrainer:
    def __init__(self):
        self.model_dir = "artifacts/models"
        self.metrics_dir = "artifacts/metrics"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def train(self, df: pd.DataFrame, n_clusters: int = 3):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(df)

        score = silhouette_score(df, clusters)

        joblib.dump(
            model, os.path.join(self.model_dir, "user_clusters_v1.pkl")
        )

        with open(
            os.path.join(self.metrics_dir, "user_metrics.json"), "w"
        ) as f:
            f.write(f'{{"silhouette_score": {score}}}')

        df["cluster"] = clusters
        return model, df
