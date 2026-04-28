"""
train_federated.py
Run this script to:
  1. Generate bank data
  2. Train federated model
  3. Save model to results/
"""

from data_generator import generate_all_banks
from federated_server import run_federated_training

if __name__ == "__main__":
    print("Step 1: Generating bank data...")
    generate_all_banks()

    print("\nStep 2: Running federated training...")
    history = run_federated_training(
        data_dir="data",
        num_rounds=10,
        local_epochs=3,
        num_banks=3,
    )

    print("\nDone! You can now run:")
    print("  python api.py         → start the REST API")
    print("  streamlit run app.py  → start the Copilot UI")
