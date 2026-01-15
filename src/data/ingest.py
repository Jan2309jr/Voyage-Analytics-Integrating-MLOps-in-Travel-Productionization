import os
import pandas as pd


class DataIngestion:
    """
    Handles ingestion of raw travel datasets (flights, hotels, users).
    Ensures data is loaded in a consistent and reproducible manner.
    """

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """
        Internal method to load a CSV file with basic validation.
        """
        file_path = os.path.join(self.data_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"Dataset is empty: {filename}")

        return df

    def load_flights_data(self) -> pd.DataFrame:
        """
        Loads flight dataset.
        """
        return self._load_csv("flights.csv")

    def load_hotels_data(self) -> pd.DataFrame:
        """
        Loads hotel dataset.
        """
        return self._load_csv("hotels.csv")

    def load_users_data(self) -> pd.DataFrame:
        """
        Loads users dataset.
        """
        return self._load_csv("users.csv")

    def load_all(self) -> dict:
        """
        Loads all datasets and returns them as a dictionary.
        """
        return {
            "flights": self.load_flights_data(),
            "hotels": self.load_hotels_data(),
            "users": self.load_users_data(),
        }


if __name__ == "__main__":
    # Simple sanity check for ingestion
    ingestion = DataIngestion()

    datasets = ingestion.load_all()
    for name, df in datasets.items():
        print(f"{name} dataset loaded with shape: {df.shape}")
