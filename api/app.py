from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.getcwd(), "artifacts", "models", "flight_model_v1.pkl"
)

# Load model at startup
model = joblib.load(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        return jsonify(
            {
                "predicted_flight_price": float(prediction)
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
