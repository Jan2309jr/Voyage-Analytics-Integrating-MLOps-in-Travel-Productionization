import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


class TravelRecommender:
    """
    Content-based recommender using user demographics
    and hotel attributes.
    """

    def __init__(self, users_df: pd.DataFrame, hotels_df: pd.DataFrame):
        self.users_df = users_df.copy()
        self.hotels_df = hotels_df.copy()
        self.scaler = MinMaxScaler()
        self._prepare_hotel_features()

    def _prepare_hotel_features(self):
        # Use numeric hotel attributes only
        numeric_cols = ["days", "price", "total"]
        self.hotel_features = self.hotels_df[numeric_cols]
        self.hotel_features_scaled = self.scaler.fit_transform(
            self.hotel_features
        )

    def _build_user_profile(self, user_code):
        # Users dataset uses 'code' as identifier
        if "code" not in self.users_df.columns:
            raise ValueError("Users dataset must contain 'code' column")

        user_row = self.users_df[self.users_df["code"] == user_code]

        if user_row.empty:
            raise ValueError("User not found")

        age = user_row["age"].iloc[0]

        # Simple, explainable heuristic:
        # Younger users → shorter & cheaper stays
        # Older users → longer & higher spend stays
        if age < 25:
            profile = [2, 2000, 4000]
        elif age < 40:
            profile = [4, 3500, 8000]
        else:
            profile = [6, 5000, 12000]

        return np.array(profile).reshape(1, -1)

    def recommend(self, user_code, top_n=5):
        user_vector = self._build_user_profile(user_code)
        user_vector_scaled = self.scaler.transform(user_vector)

        similarities = cosine_similarity(
            user_vector_scaled, self.hotel_features_scaled
        )[0]

        self.hotels_df["similarity_score"] = similarities

        recommendations = (
            self.hotels_df.sort_values(
                by="similarity_score", ascending=False
            )
            .head(top_n)
            .drop(columns=["similarity_score"])
        )

        return recommendations
