import os

BASE_DIR = os.getcwd()

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "artifacts", "models")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
LOG_DIR = os.path.join(BASE_DIR, "logs")

RANDOM_STATE = 42
