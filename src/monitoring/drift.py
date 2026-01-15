import pandas as pd
import numpy as np


class DriftDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect_drift(
        self, reference_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> bool:
        drift_detected = False

        numeric_cols = reference_df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in numeric_cols:
            ref_mean = reference_df[col].mean()
            new_mean = new_df[col].mean()

            if ref_mean == 0:
                continue

            change = abs(new_mean - ref_mean) / abs(ref_mean)
            if change > self.threshold:
                drift_detected = True

        return drift_detected
