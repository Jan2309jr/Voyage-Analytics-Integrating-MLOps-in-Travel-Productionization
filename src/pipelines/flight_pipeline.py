from src.data.ingest import DataIngestion
from src.data.validate import DataValidation
from src.data.preprocess import DataPreprocessor
from src.features.flight_features import FlightFeatureEngineer
from src.models.flights.train import FlightModelTrainer
from src.models.flights.evaluate import FlightModelEvaluator
from src.monitoring.performance import PerformanceMonitor
from src.utils.logger import get_logger


def run_flight_pipeline():
    logger = get_logger("flight_pipeline")
    logger.info("Starting flight pipeline")

    ingestion = DataIngestion()
    validator = DataValidation()
    preprocessor = DataPreprocessor()
    feature_engineer = FlightFeatureEngineer()

    datasets = ingestion.load_all()
    flights_df = datasets["flights"]

    validator.validate_flights(flights_df)

    processed_df = preprocessor.preprocess(flights_df, "flights")
    feature_df = feature_engineer.build_features(processed_df)

    trainer = FlightModelTrainer()
    model, X_test, y_test = trainer.train(feature_df)

    evaluator = FlightModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)

    monitor = PerformanceMonitor("flight")
    monitor.log_performance(metrics)

    logger.info("Flight pipeline completed successfully")


if __name__ == "__main__":
    run_flight_pipeline()
