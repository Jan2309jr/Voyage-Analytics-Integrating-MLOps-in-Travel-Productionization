import json
import os
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error


class FlightModelEvaluator:
    def __init__(self):
        self.metrics_dir = "artifacts/metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
        }

        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        with open(
            os.path.join(self.metrics_dir, "flight_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=4)

        return metrics
