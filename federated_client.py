"""
federated_client.py
Each bank is a client that trains locally and returns model weights.
Data never leaves the bank — only weights are shared.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import os

FEATURES = [
    "amount",
    "hour",
    "day_of_week",
    "merchant_category",
    "distance_from_home",
    "num_transactions_last_hour",
    "is_foreign",
]
LABEL = "is_fraud"


def build_model(input_dim: int) -> tf.keras.Model:
    """Build a simple fraud detection neural network."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


class FederatedClient:
    """Simulates a single bank participating in federated learning."""

    def __init__(self, bank_id: int, data_path: str):
        self.bank_id = bank_id
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.model = None
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """Load bank's local data and prepare for training."""
        df = pd.read_csv(self.data_path)
        X = df[FEATURES].values
        y = df[LABEL].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.y_train = y_train
        self.y_val = y_val

        self.input_dim = self.X_train.shape[1]
        print(f"  Bank {self.bank_id}: {len(self.X_train)} train | {len(self.X_val)} val samples")

    def set_weights(self, weights: List[np.ndarray]):
        """Receive global model weights from the server."""
        if self.model is None:
            self.model = build_model(self.input_dim)
        self.model.set_weights(weights)

    def train_local(self, epochs: int = 3, batch_size: int = 32) -> List[np.ndarray]:
        """Train locally and return updated weights (NOT the data)."""
        # Class weight to handle imbalance
        n_neg = np.sum(self.y_train == 0)
        n_pos = np.sum(self.y_train == 1)
        class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0,
        )
        return self.model.get_weights()

    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluate the model on local validation data."""
        results = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return {"loss": results[0], "accuracy": results[1], "auc": results[2]}

    def preprocess_single(self, transaction: dict) -> np.ndarray:
        """Preprocess a single transaction for inference."""
        row = [[transaction.get(f, 0) for f in FEATURES]]
        return self.scaler.transform(row)
