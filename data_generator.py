"""
data_generator.py
Simulates transaction data for 3 banks.
Data never leaves the bank (federated learning principle).
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_BANKS = 3
SAMPLES_PER_BANK = 2000
DATA_DIR = "data"


def generate_bank_data(bank_id: int, n_samples: int = SAMPLES_PER_BANK) -> pd.DataFrame:
    """
    Generate synthetic transaction data for a single bank.
    Features: amount, hour, day_of_week, merchant_category, distance_from_home,
              num_transactions_last_hour, is_foreign, is_fraud (label)
    """
    fraud_ratio = 0.08 + bank_id * 0.02  # slight variation per bank

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def legit_transactions(n):
        return {
            "amount": np.random.exponential(scale=150, size=n).clip(1, 5000),
            "hour": np.random.randint(6, 22, size=n),
            "day_of_week": np.random.randint(0, 7, size=n),
            "merchant_category": np.random.randint(0, 10, size=n),
            "distance_from_home": np.random.exponential(scale=20, size=n).clip(0, 200),
            "num_transactions_last_hour": np.random.poisson(lam=1.5, size=n).clip(0, 10),
            "is_foreign": np.random.binomial(1, 0.05, size=n),
            "is_fraud": np.zeros(n, dtype=int),
        }

    def fraud_transactions(n):
        return {
            "amount": np.random.exponential(scale=800, size=n).clip(100, 10000),
            "hour": np.random.choice([0, 1, 2, 3, 23], size=n),
            "day_of_week": np.random.randint(0, 7, size=n),
            "merchant_category": np.random.randint(0, 10, size=n),
            "distance_from_home": np.random.exponential(scale=300, size=n).clip(50, 2000),
            "num_transactions_last_hour": np.random.poisson(lam=6, size=n).clip(3, 20),
            "is_foreign": np.random.binomial(1, 0.6, size=n),
            "is_fraud": np.ones(n, dtype=int),
        }

    legit = pd.DataFrame(legit_transactions(n_legit))
    fraud = pd.DataFrame(fraud_transactions(n_fraud))

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=bank_id).reset_index(drop=True)
    df["bank_id"] = bank_id
    return df


def generate_all_banks():
    """Generate and save data for all banks."""
    os.makedirs(DATA_DIR, exist_ok=True)

    all_data = {}
    for bank_id in range(NUM_BANKS):
        df = generate_bank_data(bank_id)
        filepath = os.path.join(DATA_DIR, f"bank_{bank_id}_transactions.csv")
        df.to_csv(filepath, index=False)
        all_data[bank_id] = df
        fraud_count = df["is_fraud"].sum()
        print(f"Bank {bank_id}: {len(df)} transactions | {fraud_count} fraudulent ({fraud_count/len(df)*100:.1f}%)")

    print(f"\nData saved to '{DATA_DIR}/' directory.")
    return all_data


if __name__ == "__main__":
    generate_all_banks()
