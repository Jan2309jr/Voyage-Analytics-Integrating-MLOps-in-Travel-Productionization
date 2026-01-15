import json
import os
from sklearn.metrics import accuracy_score, f1_score


class HotelModelEvaluator:
    def __init__(self):
        self.metrics_dir = "artifacts/metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
        }

        with open(
            os.path.join(self.metrics_dir, "hotel_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=4)

        return metrics
