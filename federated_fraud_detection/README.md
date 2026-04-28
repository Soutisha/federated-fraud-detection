# 🛡️ FraudShield — Federated Learning Fraud Detection + Copilot

A privacy-preserving fraud detection system where multiple banks collaboratively train
a model using **Federated Learning**, with a **Copilot interface** for natural language
explanations powered by an LLM.

---

## 🏗️ Architecture

```
Banks (3x local data)
      │
      ▼
FederatedClient  ──┐
FederatedClient  ──┼──► FederatedServer (FedAvg) ──► federated_model.h5
FederatedClient  ──┘
                              │
                              ▼
                         Flask API (/predict)
                              │
                              ▼
                    Streamlit UI + Copilot (LLM)
```

---

## 📁 Project Structure

```
federated_fraud_detection/
├── data_generator.py      # Simulate bank transaction data
├── federated_client.py    # Local bank training logic
├── federated_server.py    # FedAvg aggregation + training loop
├── train_federated.py     # One-command: generate data + train
├── api.py                 # Flask REST API
├── copilot.py             # LLM explanation engine
├── app.py                 # Streamlit Copilot UI
├── requirements.txt
├── .env.example
└── .vscode/
    └── launch.json        # VS Code run configs
```

---

## ⚡ Quick Start

### 1. Create virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (optional — fallback works without it)
```

### 4. Generate data + train model

```bash
python train_federated.py
```

### 5. Start the API (Terminal 1)

```bash
python api.py
```

### 6. Start the Copilot UI (Terminal 2)

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | /health          | Check if API and model are ready     |
| POST   | /predict         | Predict fraud for one transaction    |
| POST   | /batch_predict   | Predict fraud for multiple transactions |

### Example: Single Prediction

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 3500,
    "hour": 2,
    "day_of_week": 5,
    "merchant_category": 3,
    "distance_from_home": 650,
    "num_transactions_last_hour": 8,
    "is_foreign": 1
  }'
```

### Example Response

```json
{
  "fraud_probability": 0.9231,
  "is_fraud": true,
  "risk_level": "HIGH",
  "transaction": { ... }
}
```

---

## 🧠 How Federated Learning Works Here

1. **Data Generation** — 3 banks each get synthetic transaction data (never shared)
2. **Local Training** — Each bank trains on its own data for N epochs
3. **Weight Sharing** — Only model weights (not data) are sent to the server
4. **FedAvg** — Server computes weighted average of all client weights
5. **Repeat** — Process repeats for multiple rounds until convergence

---

## 🤖 Copilot

The Copilot translates raw model outputs into business-readable explanations:

- **With OpenAI API key** → GPT-3.5-turbo generates natural language analysis
- **Without API key** → Rule-based fallback engine works automatically

---

## ⚙️ VS Code Integration

Use the pre-configured launch configs in `.vscode/launch.json`:

- **1. Generate Data + Train** → runs `train_federated.py`
- **2. Start Flask API** → runs `api.py`
- **3. Start Streamlit UI** → runs `streamlit run app.py`

---

## 🔒 Limitations & Future Work

| Limitation           | Improvement                          |
|----------------------|--------------------------------------|
| Simulated data       | Real banking data (with consent)     |
| No encryption        | Differential privacy / Secure Aggregation |
| Local deployment     | AWS/Azure cloud deployment           |
| Basic model          | Transformer-based fraud detection    |

---

## 📊 Tech Stack

| Layer              | Technology                    |
|--------------------|-------------------------------|
| ML Framework       | TensorFlow / Keras            |
| Federated Learning | Custom FedAvg implementation  |
| API                | Flask                         |
| AI Copilot         | OpenAI GPT-3.5-turbo          |
| UI                 | Streamlit                     |
| Data Processing    | NumPy, Pandas, scikit-learn   |
