import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import joblib
import pandas as pd

from src.data.ingest import DataIngestion
from src.models.recommendation.recommender import TravelRecommender
from src.data.preprocess import DataPreprocessor
from src.features.flight_features import FlightFeatureEngineer

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Voyage Analytics", layout="wide")
st.title("✈️ Voyage Analytics – Travel ML Platform")

# -----------------------------------
# LOAD MODELS & DATA
# -----------------------------------
@st.cache_resource
def load_flight_model():
    return joblib.load("artifacts/models/flight_model_v1.pkl")

@st.cache_resource
def load_data():
    data = DataIngestion().load_all()
    return data["users"], data["hotels"]

flight_model = load_flight_model()
users_df, hotels_df = load_data()
recommender = TravelRecommender(users_df, hotels_df)

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------
menu = st.sidebar.radio(
    "Select Feature",
    ["Flight Price Prediction", "Hotel Recommendations"]
)

# -----------------------------------
# FLIGHT PRICE PREDICTION
# -----------------------------------
if menu == "Flight Price Prediction":
    st.header("Flight Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        duration = st.number_input("Flight Duration (hours)", 1.0, 20.0, 3.0)
        distance = st.number_input("Distance (scaled)", 0.0, 1.0, 0.3)
        route_frequency = st.number_input("Route Frequency", 0.0, 1.0, 0.5)

    with col2:
        is_peak_season = st.selectbox("Peak Season?", [0, 1])
        days_before_departure = st.slider("Days Before Departure", 1, 60, 15)

    if st.button("Predict Price"):
        raw_input = pd.DataFrame([{
            "agency": "DemoAgency",
            "flightType": "Economy",
            "date": pd.Timestamp.today(),
            "duration": duration,
            "distance": distance,
            "route_frequency": route_frequency,
            "is_peak_season": is_peak_season,
            "days_before_departure": days_before_departure
        }])

        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess(raw_input, "flights")


        feature_engineer = FlightFeatureEngineer()
        features = feature_engineer.build_features(processed)

        expected_cols = flight_model.feature_names_in_
        features = features.reindex(columns=expected_cols, fill_value=0)

        prediction = flight_model.predict(features)[0]
        st.success(f"Estimated Flight Price: ₹ {round(prediction, 2)}")

# -----------------------------------
# HOTEL RECOMMENDATIONS
# -----------------------------------
elif menu == "Hotel Recommendations":
    st.header("Hotel Recommendations")

    user_code = st.selectbox("Select User", users_df["code"].tolist())
    top_n = st.slider("Number of Recommendations", 1, 5, 3)

    if st.button("Get Recommendations"):
        recommendations = recommender.recommend(user_code=user_code, top_n=top_n)
        st.dataframe(recommendations.reset_index(drop=True))
