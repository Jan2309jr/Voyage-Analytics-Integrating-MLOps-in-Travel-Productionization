from src.models.flights.predict import FlightPricePredictor


def test_flight_prediction_output():
    predictor = FlightPricePredictor()

    dummy_input = {
        "duration": 0.1,
        "distance": 0.2,
        "route_frequency": 0.3,
        "is_peak_season": 1,
        "days_before_departure": 0.4,
    }

    prediction = predictor.predict(dummy_input)

    assert prediction is not None
    assert isinstance(prediction, float)
