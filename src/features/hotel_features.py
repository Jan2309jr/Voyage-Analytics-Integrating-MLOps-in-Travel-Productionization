import pandas as pd


class HotelFeatureEngineer:
    """
    Builds derived features for hotel demand prediction.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Weekend stay indicator
        if "checkin_weekday" in df.columns:
            df["is_weekend"] = df["checkin_weekday"].apply(
                lambda x: 1 if x >= 5 else 0
            )

        # Seasonal demand indicator
        if "checkin_month" in df.columns:
            df["is_high_season"] = df["checkin_month"].apply(
                lambda x: 1 if x in [4, 5, 6, 10, 11, 12] else 0
            )

        # Price bucket (low, medium, high)
        if "price_per_night" in df.columns:
            df["price_bucket"] = pd.qcut(
                df["price_per_night"], q=3, labels=[0, 1, 2]
            )

        return df
