import pandas as pd


class FlightFeatureEngineer:
    """
    Builds derived features for flight price prediction.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Days before departure (if columns exist)
        if "booking_date" in df.columns and "departure_date" in df.columns:
            df["booking_date"] = pd.to_datetime(df["booking_date"])
            df["departure_date"] = pd.to_datetime(df["departure_date"])
            df["days_before_departure"] = (
                df["departure_date"] - df["booking_date"]
            ).dt.days

            df.drop(["booking_date", "departure_date"], axis=1, inplace=True)

        # Peak season flag (basic travel assumption)
        if "departure_month" in df.columns:
            df["is_peak_season"] = df["departure_month"].apply(
                lambda x: 1 if x in [4, 5, 6, 10, 11, 12] else 0
            )

        # Route frequency (demand proxy)
        if "route" in df.columns:
            route_freq = df["route"].value_counts()
            df["route_frequency"] = df["route"].map(route_freq)

        return df
