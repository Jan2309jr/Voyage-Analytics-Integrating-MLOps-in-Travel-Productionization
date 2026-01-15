import json
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class FlightModelEvaluator:
    def __init__(self):
        self.metrics_dir = "artifacts/metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse,
        }

        with open(
            os.path.join(self.metrics_dir, "flight_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=4)

        return metrics
