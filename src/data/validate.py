import pandas as pd


class DataValidation:
    """
    Validates travel datasets before preprocessing.
    Ensures schema consistency, data completeness, and value sanity.
    """

    def __init__(self):
        pass

    def _check_empty(self, df: pd.DataFrame, name: str):
        if df.empty:
            raise ValueError(f"{name} dataset is empty")

    def _check_duplicates(self, df: pd.DataFrame, name: str):
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            raise ValueError(
                f"{name} dataset contains {duplicates} duplicate rows"
            )

    def _check_missing_values(self, df: pd.DataFrame, name: str):
        missing = df.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0].to_dict()
            raise ValueError(
                f"{name} dataset has missing values: {missing_cols}"
            )

    def _check_negative_values(
        self, df: pd.DataFrame, columns: list, name: str
    ):
        for col in columns:
            if col in df.columns and (df[col] < 0).any():
                raise ValueError(
                    f"{name} dataset has negative values in column '{col}'"
                )

    def _check_unique_column(
        self, df: pd.DataFrame, column: str, name: str
    ):
        if column in df.columns and not df[column].is_unique:
            raise ValueError(
                f"{name} dataset has non-unique values in '{column}'"
            )

    def validate_flights(self, df: pd.DataFrame):
        name = "Flights"
        self._check_empty(df, name)
        self._check_duplicates(df, name)
        self._check_missing_values(df, name)
        self._check_negative_values(
            df, ["price", "duration", "distance"], name
        )

    def validate_hotels(self, df: pd.DataFrame):
        name = "Hotels"
        self._check_empty(df, name)
        self._check_duplicates(df, name)
        self._check_missing_values(df, name)
        self._check_negative_values(df, ["price_per_night"], name)

    def validate_users(self, df: pd.DataFrame):
        name = "Users"
        self._check_empty(df, name)
        self._check_duplicates(df, name)
        self._check_missing_values(df, name)
        self._check_unique_column(df, "user_id", name)
        self._check_negative_values(
            df, ["age", "total_bookings"], name
        )

    def validate_all(self, datasets: dict):
        """
        Validates all datasets in one call.
        """
        self.validate_flights(datasets.get("flights"))
        self.validate_hotels(datasets.get("hotels"))
        self.validate_users(datasets.get("users"))


if __name__ == "__main__":
    # Standalone validation test
    from ingest import DataIngestion

    ingestion = DataIngestion()
    datasets = ingestion.load_all()

    validator = DataValidation()
    validator.validate_all(datasets)

    print("All datasets passed validation checks.")
