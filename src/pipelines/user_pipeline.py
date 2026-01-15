from src.data.ingest import DataIngestion
from src.data.validate import DataValidation
from src.data.preprocess import DataPreprocessor
from src.features.user_features import UserFeatureEngineer
from src.models.users.train import UserClusteringTrainer
from src.monitoring.performance import PerformanceMonitor
from src.utils.logger import get_logger


def run_user_pipeline():
    logger = get_logger("user_pipeline")
    logger.info("Starting user pipeline")

    ingestion = DataIngestion()
    validator = DataValidation()
    preprocessor = DataPreprocessor()
    feature_engineer = UserFeatureEngineer()

    datasets = ingestion.load_all()
    users_df = datasets["users"]

    validator.validate_users(users_df)

    processed_df = preprocessor.preprocess(users_df, "users")
    feature_df = feature_engineer.build_features(processed_df)

    trainer = UserClusteringTrainer()
    model, clustered_df = trainer.train(feature_df)

    monitor = PerformanceMonitor("users")
    monitor.log_performance(
        {"num_clusters": clustered_df["cluster"].nunique()}
    )

    logger.info("User pipeline completed successfully")


if __name__ == "__main__":
    run_user_pipeline()
