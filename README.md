# Voyage Analytics - Integrating MLOps in Travel

## Overview
Voyage Analytics is an end-to-end MLOps project that demonstrates how
machine learning systems in the travel domain can be productionized
using automated pipelines, monitoring, and versioning.

The project covers:
- Flight price prediction
- Hotel demand prediction
- User travel segmentation

## Datasets
- Flights: Pricing, routes, travel dates
- Hotels: Pricing, booking demand
- Users: Travel behavior and preferences

## Architecture
- Data Ingestion
- Data Validation
- Preprocessing
- Feature Engineering
- Model Training & Evaluation
- Monitoring & Drift Detection
- Automated Pipelines

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Pytest
- Joblib

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run pipelines:
python src/pipelines/flight_pipeline.py
python src/pipelines/hotel_pipeline.py
python src/pipelines/user_pipeline.py

3. Or run everything:
python main.py

4. Run tests:
pytest


## MLOps Features
- Automated pipelines
- Model versioning
- Performance monitoring
- Data drift detection
- Reproducible artifacts

## Ethical Note
The gender classification model included in this project is a
demonstration-only component using synthetic data. It is not intended
for real-world deployment. Gender inference from behavioral data raises
ethical and privacy concerns and must be handled with caution in
production systems.

## Future Enhancements
- CI/CD integration
- Real-time inference APIs
- Advanced drift detection
- Cloud deployment
