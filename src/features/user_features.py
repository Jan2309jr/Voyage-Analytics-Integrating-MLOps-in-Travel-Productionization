import pandas as pd


class UserFeatureEngineer:
    """
    Builds derived features for user segmentation.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Average spend per booking
        if "total_spend" in df.columns and "total_bookings" in df.columns:
            df["avg_spend_per_booking"] = (
                df["total_spend"] / df["total_bookings"].replace(0, 1)
            )

        # Travel frequency category
        if "total_bookings" in df.columns:
            df["travel_frequency"] = pd.cut(
                df["total_bookings"],
                bins=[-1, 2, 5, 10, float("inf")],
                labels=[0, 1, 2, 3],
            )

        # Loyalty flag
        if "membership_years" in df.columns:
            df["is_loyal_user"] = df["membership_years"].apply(
                lambda x: 1 if x >= 3 else 0
            )

        return df
