"""
api.py
Flask REST API that serves the federated fraud detection model.
Run: python api.py
"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL_PATH = os.path.join("results", "federated_model.h5")
DATA_DIR = "data"

FEATURES = [
    "amount",
    "hour",
    "day_of_week",
    "merchant_category",
    "distance_from_home",
    "num_transactions_last_hour",
    "is_foreign",
]

# ── Load model & fit scaler on training data ──────────────────────────────────

def load_model_and_scaler():
    """Load trained model and fit scaler from combined bank data."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train_federated.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Fit scaler on all available bank data (simulates a shared schema, not shared data)
    dfs = []
    for i in range(3):
        path = os.path.join(DATA_DIR, f"bank_{i}_transactions.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise FileNotFoundError("No bank data found. Run train_federated.py first.")

    combined = pd.concat(dfs, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(combined[FEATURES].values)

    return model, scaler


try:
    model, scaler = load_model_and_scaler()
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    model, scaler = None, None
    print(f"Warning: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    status = "ready" if model is not None else "model_not_loaded"
    return jsonify({"status": status})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud probability for a transaction.

    Expected JSON body:
    {
        "amount": 1500.0,
        "hour": 2,
        "day_of_week": 5,
        "merchant_category": 3,
        "distance_from_home": 450.0,
        "num_transactions_last_hour": 7,
        "is_foreign": 1
    }
    """
    if model is None or scaler is None:
        return jsonify({
            "error": "Model not loaded. Run train_federated.py first."
        }), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400

    # Validate required fields
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Preprocess
    try:
        row = np.array([[float(data[f]) for f in FEATURES]])
        row_scaled = scaler.transform(row)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400

    # Predict
    prob = float(model.predict(row_scaled, verbose=0)[0][0])
    is_fraud = prob >= 0.5

    # Risk level
    if prob >= 0.8:
        risk = "HIGH"
    elif prob >= 0.5:
        risk = "MEDIUM"
    elif prob >= 0.3:
        risk = "LOW"
    else:
        risk = "MINIMAL"

    return jsonify({
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud,
        "risk_level": risk,
        "transaction": data,
    })


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict fraud for multiple transactions at once.

    Expected JSON body: {"transactions": [ {...}, {...} ]}
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run train_federated.py first."}), 503

    data = request.get_json(force=True)
    transactions = data.get("transactions", [])
    if not transactions:
        return jsonify({"error": "No transactions provided."}), 400

    results = []
    for txn in transactions:
        missing = [f for f in FEATURES if f not in txn]
        if missing:
            results.append({"error": f"Missing fields: {missing}", "transaction": txn})
            continue
        try:
            row = np.array([[float(txn[f]) for f in FEATURES]])
            row_scaled = scaler.transform(row)
            prob = float(model.predict(row_scaled, verbose=0)[0][0])
            results.append({
                "fraud_probability": round(prob, 4),
                "is_fraud": prob >= 0.5,
                "risk_level": "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW" if prob >= 0.3 else "MINIMAL",
                "transaction": txn,
            })
        except Exception as e:
            results.append({"error": str(e), "transaction": txn})

    return jsonify({"results": results, "total": len(results)})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    print(f"\nFlask API running at http://{host}:{port}")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("  POST /batch_predict\n")

    app.run(host=host, port=port, debug=debug)
