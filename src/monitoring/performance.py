import json
import os
from datetime import datetime


class PerformanceMonitor:
    """
    Logs model performance metrics over time.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_dir = os.path.join("artifacts", "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.file_path = os.path.join(
            self.metrics_dir,
            f"{self.model_name}_performance_history.json",
        )

    def log_performance(self, metrics: dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }

        history = []
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                history = json.load(f)

        history.append(record)

        with open(self.file_path, "w") as f:
            json.dump(history, f, indent=4)
