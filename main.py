from src.pipelines.flight_pipeline import run_flight_pipeline
from src.pipelines.hotel_pipeline import run_hotel_pipeline
from src.pipelines.user_pipeline import run_user_pipeline


def main():
    run_flight_pipeline()
    run_hotel_pipeline()
    run_user_pipeline()


if __name__ == "__main__":
    main()
