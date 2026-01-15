import pytest
from src.data.ingest import DataIngestion
from src.data.validate import DataValidation


def test_data_validation_passes():
    ingestion = DataIngestion()
    datasets = ingestion.load_all()

    validator = DataValidation()

    # Should not raise any exception
    validator.validate_flights(datasets["flights"])
    validator.validate_hotels(datasets["hotels"])
    validator.validate_users(datasets["users"])
