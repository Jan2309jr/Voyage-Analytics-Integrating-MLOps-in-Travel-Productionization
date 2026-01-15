from src.data.ingest import DataIngestion
from src.data.validate import DataValidation
from src.data.preprocess import DataPreprocessor
from src.features.hotel_features import HotelFeatureEngineer
from src.models.hotels.train import HotelModelTrainer
from src.models.hotels.evaluate import HotelModelEvaluator
from src.monitoring.performance import PerformanceMonitor
from src.utils.logger import get_logger


def run_hotel_pipeline():
    logger = get_logger("hotel_pipeline")
    logger.info("Starting hotel pipeline")

    ingestion = DataIngestion()
    validator = DataValidation()
    preprocessor = DataPreprocessor()
    feature_engineer = HotelFeatureEngineer()

    datasets = ingestion.load_all()
    hotels_df = datasets["hotels"]

    # Validate
    validator.validate_hotels(hotels_df)

    # Preprocess
    processed_df = preprocessor.preprocess(hotels_df, "hotels")

    # Feature engineering
    feature_df = feature_engineer.build_features(processed_df)

# ----- CREATE TARGET VARIABLE (DATASET-SPECIFIC, CORRECT) -----

    threshold = feature_df["total"].median()
    feature_df["high_demand"] = (feature_df["total"] > threshold).astype(int)

# ----------------------------------

    # ----------------------------------

    # Train model
    trainer = HotelModelTrainer()
    model, X_test, y_test = trainer.train(feature_df)

    # Evaluate
    evaluator = HotelModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)

    # Monitor
    monitor = PerformanceMonitor("hotel")
    monitor.log_performance(metrics)

    logger.info("Hotel pipeline completed successfully")


if __name__ == "__main__":
    run_hotel_pipeline()
