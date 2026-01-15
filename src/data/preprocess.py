import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    Handles preprocessing of travel datasets:
    - Missing value handling
    - Categorical encoding
    - Numerical scaling
    - Date feature extraction
    """

    def __init__(self, output_dir: str = "artifacts/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values:
        - Numeric: median
        - Categorical: mode
        """
        for column in df.columns:
            if df[column].dtype in ["int64", "float64"]:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])
        return df

    def _encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encodes categorical columns.
        """
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        return df

    def _scale_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales numerical columns using StandardScaler.
        """
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts features from date columns if present.
        """
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_weekday"] = df[col].dt.weekday
                df.drop(columns=[col], inplace=True)
        return df

    def preprocess(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        """
        df = df.copy()

        df = self._handle_missing_values(df)
        df = self._extract_date_features(df)
        df = self._encode_categorical_columns(df)
        df = self._scale_numerical_columns(df)

        output_path = os.path.join(
            self.output_dir, f"{dataset_name}_processed.csv"
        )
        df.to_csv(output_path, index=False)

        return df


if __name__ == "__main__":
    # Example standalone run (for testing only)
    from ingest import DataIngestion

    ingestion = DataIngestion()
    datasets = ingestion.load_all()

    preprocessor = DataPreprocessor()

    for name, df in datasets.items():
        processed_df = preprocessor.preprocess(df, name)
        print(f"{name} processed data saved with shape: {processed_df.shape}")
