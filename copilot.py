"""
copilot.py
LLM-powered Copilot that converts raw fraud predictions into
human-readable explanations for business users.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a fraud analyst AI assistant for a banking system.
You receive transaction data and a fraud detection model's output.
Your job is to:
1. Explain clearly whether the transaction is suspicious and why
2. Highlight the key risk factors from the transaction data
3. Suggest a concrete action (approve, flag for review, decline)
4. Keep the explanation professional but understandable for non-technical business users

Be concise (3-5 sentences max). Do not use technical ML jargon."""


def get_copilot_explanation(transaction: dict, prediction: dict) -> str:
    """
    Generate a natural language explanation for a fraud prediction.

    Args:
        transaction: Raw transaction fields
        prediction: API response with fraud_probability, is_fraud, risk_level

    Returns:
        A human-readable explanation string
    """
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key or api_key == "OPENAI_API_KEY=your_api_key_here":
        return _fallback_explanation(transaction, prediction)

    try:
        client = OpenAI(api_key=api_key)

        user_message = f"""Transaction details:
- Amount: ${transaction.get('amount', 'N/A')}
- Hour of day: {transaction.get('hour', 'N/A')}:00
- Day of week: {_day_name(transaction.get('day_of_week', 0))}
- Merchant category code: {transaction.get('merchant_category', 'N/A')}
- Distance from home: {transaction.get('distance_from_home', 'N/A')} km
- Transactions in last hour: {transaction.get('num_transactions_last_hour', 'N/A')}
- Foreign transaction: {'Yes' if transaction.get('is_foreign') else 'No'}

Model output:
- Fraud probability: {prediction.get('fraud_probability', 0) * 100:.1f}%
- Risk level: {prediction.get('risk_level', 'UNKNOWN')}
- Flagged as fraud: {'Yes' if prediction.get('is_fraud') else 'No'}

Please explain this to a bank manager."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=200,
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return _fallback_explanation(transaction, prediction, error=str(e))


def _fallback_explanation(transaction: dict, prediction: dict, error: str = "") -> str:
    """
    Rule-based fallback explanation when OpenAI API is unavailable.
    Ensures the system works end-to-end even without an API key.
    """
    prob = prediction.get("fraud_probability", 0)
    risk = prediction.get("risk_level", "UNKNOWN")
    is_fraud = prediction.get("is_fraud", False)

    reasons = []

    amount = float(transaction.get("amount", 0))
    if amount > 2000:
        reasons.append(f"unusually high transaction amount (${amount:.0f})")

    hour = int(transaction.get("hour", 12))
    if hour <= 4 or hour >= 23:
        reasons.append(f"transaction at an unusual hour ({hour}:00)")

    dist = float(transaction.get("distance_from_home", 0))
    if dist > 200:
        reasons.append(f"transaction {dist:.0f} km from home")

    txns = int(transaction.get("num_transactions_last_hour", 0))
    if txns >= 5:
        reasons.append(f"{txns} transactions within the last hour")

    if transaction.get("is_foreign"):
        reasons.append("foreign merchant")

    reason_text = (
        "Key risk factors: " + ", ".join(reasons) + "."
        if reasons
        else "No single dominant risk factor was detected."
    )

    if is_fraud:
        verdict = f"⚠️ This transaction is flagged as HIGH RISK with {prob*100:.1f}% fraud probability."
        action = "Recommended action: DECLINE and notify the cardholder."
    elif risk == "LOW":
        verdict = f"This transaction shows minor anomalies with {prob*100:.1f}% fraud probability."
        action = "Recommended action: APPROVE but log for monitoring."
    else:
        verdict = f"This transaction appears legitimate with only {prob*100:.1f}% fraud probability."
        action = "Recommended action: APPROVE."

    note = f" (Fallback mode — {error})" if error else ""
    return f"{verdict} {reason_text} {action}{note}"


def _day_name(day: int) -> str:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        return days[int(day) % 7]
    except (ValueError, TypeError):
        return str(day)
