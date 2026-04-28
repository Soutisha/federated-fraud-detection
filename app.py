"""
app.py
Streamlit Copilot UI for the Fraud Detection System.
Run: streamlit run app.py
"""

import streamlit as st
import requests
import json
import os

from copilot import get_copilot_explanation

API_BASE = "http://127.0.0.1:5000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .risk-high   { background: #ff4d4d22; border-left: 4px solid #ff4d4d; padding: 1rem; border-radius: 4px; }
    .risk-medium { background: #ff990022; border-left: 4px solid #ff9900; padding: 1rem; border-radius: 4px; }
    .risk-low    { background: #00cc6622; border-left: 4px solid #00cc66; padding: 1rem; border-radius: 4px; }
    .risk-minimal{ background: #00aaff22; border-left: 4px solid #00aaff; padding: 1rem; border-radius: 4px; }
    .copilot-box { background: #f0f4ff; border: 1px solid #c0d0ff; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("FraudShield")
    st.caption("Federated Learning · Privacy Preserving")
    st.divider()

    st.subheader("ℹ️ About")
    st.markdown("""
    This system uses **Federated Learning** across 3 simulated banks.
    Bank data **never leaves** each institution — only model weights are shared.

    The **Copilot** interface explains predictions in plain language for business users.
    """)

    st.divider()
    st.subheader("🔌 API Status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.json().get("status") == "ready":
            st.success("API Connected ✅")
        else:
            st.warning("API running but model not loaded.")
    except requests.exceptions.ConnectionError:
        st.error("API offline. Run `python api.py` first.")

    st.divider()
    st.caption("Built with Federated Learning + OpenAI Copilot")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛡️ FraudShield Copilot</div>', unsafe_allow_html=True)
st.markdown("Enter transaction details to detect fraud and get an AI-powered explanation.")
st.divider()

tab1, tab2 = st.tabs(["🔍 Single Transaction", "📊 Batch Analysis"])

# ── TAB 1: Single Transaction ─────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=1.0, max_value=50000.0, value=250.0, step=10.0
        )
        hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday"][x],
            index=2,
        )
        merchant_category = st.selectbox(
            "Merchant Category",
            options=list(range(10)),
            format_func=lambda x: [
                "Groceries", "Electronics", "Travel", "Dining",
                "Healthcare", "Entertainment", "Utilities", "Retail",
                "Fuel", "Online Services"
            ][x],
            index=1,
        )
        distance_from_home = st.number_input(
            "Distance from Home (km)", min_value=0.0, max_value=5000.0, value=12.0
        )
        num_transactions_last_hour = st.slider(
            "Transactions in Last Hour", 0, 20, 1
        )
        is_foreign = st.checkbox("Foreign Merchant?", value=False)

    with col2:
        st.subheader("Prediction & Explanation")

        if st.button("🔎 Analyze Transaction", use_container_width=True, type="primary"):
            transaction = {
                "amount": amount,
                "hour": hour,
                "day_of_week": day_of_week,
                "merchant_category": merchant_category,
                "distance_from_home": distance_from_home,
                "num_transactions_last_hour": num_transactions_last_hour,
                "is_foreign": int(is_foreign),
            }

            try:
                with st.spinner("Analyzing..."):
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        json=transaction,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    prediction = resp.json()

                # Risk display
                risk = prediction.get("risk_level", "UNKNOWN")
                prob = prediction.get("fraud_probability", 0)
                risk_class = f"risk-{risk.lower()}"

                st.markdown(f"""
                <div class="{risk_class}">
                    <strong>Risk Level: {risk}</strong><br>
                    Fraud Probability: <strong>{prob*100:.1f}%</strong><br>
                    Flagged: <strong>{'⚠️ YES' if prediction.get('is_fraud') else '✅ NO'}</strong>
                </div>
                """, unsafe_allow_html=True)

                # Progress bar
                bar_color = "🔴" if prob >= 0.8 else "🟠" if prob >= 0.5 else "🟡" if prob >= 0.3 else "🟢"
                st.progress(prob, text=f"{bar_color} {prob*100:.1f}% fraud probability")

                # Copilot explanation
                st.markdown("#### 🤖 Copilot Explanation")
                with st.spinner("Generating explanation..."):
                    explanation = get_copilot_explanation(transaction, prediction)

                st.markdown(f'<div class="copilot-box">{explanation}</div>', unsafe_allow_html=True)

                # Raw JSON expander
                with st.expander("📄 Raw API Response"):
                    st.json(prediction)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure `python api.py` is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ── TAB 2: Batch Analysis ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Transaction Analysis")
    st.info("Paste a JSON array of transactions to analyze multiple at once.")

    sample_batch = json.dumps([
        {"amount": 3500, "hour": 2, "day_of_week": 5, "merchant_category": 2,
         "distance_from_home": 600, "num_transactions_last_hour": 8, "is_foreign": 1},
        {"amount": 85, "hour": 13, "day_of_week": 1, "merchant_category": 0,
         "distance_from_home": 3, "num_transactions_last_hour": 1, "is_foreign": 0},
    ], indent=2)

    batch_input = st.text_area(
        "Transactions JSON",
        value=sample_batch,
        height=250,
    )

    if st.button("🔎 Analyze Batch", use_container_width=True, type="primary"):
        try:
            transactions = json.loads(batch_input)
            if not isinstance(transactions, list):
                st.error("Input must be a JSON array.")
            else:
                with st.spinner(f"Analyzing {len(transactions)} transactions..."):
                    resp = requests.post(
                        f"{API_BASE}/batch_predict",
                        json={"transactions": transactions},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                results = data.get("results", [])
                flagged = sum(1 for r in results if r.get("is_fraud"))

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Analyzed", len(results))
                col_b.metric("Flagged as Fraud", flagged)
                col_c.metric("Clean Transactions", len(results) - flagged)

                for i, result in enumerate(results, 1):
                    if "error" in result:
                        st.error(f"Transaction {i}: {result['error']}")
                        continue

                    risk = result.get("risk_level", "UNKNOWN")
                    prob = result.get("fraud_probability", 0)
                    icon = "🔴" if result.get("is_fraud") else "✅"
                    with st.expander(f"{icon} Transaction {i} — {risk} ({prob*100:.1f}%)"):
                        st.json(result)

        except json.JSONDecodeError:
            st.error("Invalid JSON. Please check the format.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure `python api.py` is running.")
        except Exception as e:
            st.error(f"Error: {e}")
