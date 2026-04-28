"""
federated_server.py
Central server that coordinates federated learning.
Aggregates model weights using FedAvg (Federated Averaging).
Never sees raw bank data — only model weights.
"""

import numpy as np
import tensorflow as tf
from typing import List
import os

from federated_client import build_model, FederatedClient

RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "federated_model.h5")


class FederatedServer:
    """Central server for Federated Averaging (FedAvg)."""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.global_model = build_model(input_dim)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("Server initialized with fresh global model.")

    def get_global_weights(self) -> List[np.ndarray]:
        """Return current global model weights."""
        return self.global_model.get_weights()

    def aggregate(self, client_weights: List[List[np.ndarray]], client_sizes: List[int]) -> List[np.ndarray]:
        """
        Federated Averaging: weighted average of client weights.
        Clients with more data contribute proportionally more.
        """
        total = sum(client_sizes)
        new_weights = []

        for layer_idx in range(len(client_weights[0])):
            weighted_layer = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
            for client_idx, weights in enumerate(client_weights):
                weight_factor = client_sizes[client_idx] / total
                weighted_layer += weight_factor * weights[layer_idx]
            new_weights.append(weighted_layer.astype(np.float32))

        self.global_model.set_weights(new_weights)
        return new_weights

    def save_model(self):
        """Save the global model to disk."""
        self.global_model.save(MODEL_PATH)
        print(f"Global model saved to '{MODEL_PATH}'")

    @staticmethod
    def load_model() -> tf.keras.Model:
        """Load the saved global model."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run train_federated.py first."
            )
        return tf.keras.models.load_model(MODEL_PATH)


def run_federated_training(
    data_dir: str = "data",
    num_rounds: int = 10,
    local_epochs: int = 3,
    num_banks: int = 3,
):
    """
    Main federated training loop.
    Simulates the complete FL process across all banks.
    """
    print("\n" + "="*60)
    print("  FEDERATED LEARNING - FRAUD DETECTION")
    print("="*60)

    # Initialize clients (banks)
    clients = []
    for bank_id in range(num_banks):
        path = os.path.join(data_dir, f"bank_{bank_id}_transactions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data for bank {bank_id} not found. Run data_generator.py first."
            )
        clients.append(FederatedClient(bank_id, path))

    # Initialize server
    input_dim = clients[0].input_dim
    server = FederatedServer(input_dim)

    client_sizes = [len(c.X_train) for c in clients]
    history = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")

        # Step 1: Broadcast global weights to all clients
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_weights(global_weights)

        # Step 2: Each client trains locally and returns weights
        client_weights = []
        for client in clients:
            weights = client.train_local(epochs=local_epochs)
            client_weights.append(weights)
            print(f"  Bank {client.bank_id}: local training done")

        # Step 3: Server aggregates weights (FedAvg)
        server.aggregate(client_weights, client_sizes)

        # Step 4: Evaluate on each client's validation set
        round_metrics = {}
        for client in clients:
            client.set_weights(server.get_global_weights())
            metrics = client.evaluate()
            round_metrics[f"bank_{client.bank_id}"] = metrics

        avg_auc = np.mean([m["auc"] for m in round_metrics.values()])
        avg_acc = np.mean([m["accuracy"] for m in round_metrics.values()])
        print(f"  Avg Accuracy: {avg_acc:.4f} | Avg AUC: {avg_auc:.4f}")
        history.append({"round": round_num, "avg_accuracy": avg_acc, "avg_auc": avg_auc})

    # Save final model
    server.save_model()

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  Final Avg AUC:      {history[-1]['avg_auc']:.4f}")
    print(f"  Final Avg Accuracy: {history[-1]['avg_accuracy']:.4f}")
    print("="*60 + "\n")

    return history


if __name__ == "__main__":
    run_federated_training()
