def calculate_risk_score(
    fraud_probability,
    anomaly_score
):
    """
    Combine supervised fraud probability
    and anomaly score into one risk score.
    """

    # Convert Isolation Forest score into
    # an anomaly level.

    # For our trained model, scores generally
    # range approximately from -0.5 to +0.5.

    normalized_anomaly = (
        0.5 - anomaly_score
    )

    # Keep value between 0 and 1.

    normalized_anomaly = max(
        0,
        min(
            1,
            normalized_anomaly
        )
    )

    # Combine the two AI signals.

    risk_score = (
        0.70 * fraud_probability
        +
        0.30 * normalized_anomaly
    )

    return risk_score


def make_decision(risk_score):
    """
    Convert risk score into a transaction decision.
    """

    if risk_score < 0.30:

        return "APPROVE"

    elif risk_score < 0.70:

        return "FLAG_VERIFY"

    else:

        return "BLOCK"


def evaluate_transaction(
    fraud_probability,
    anomaly_score
):
    """
    Complete risk evaluation.
    """

    # Convert NumPy values to normal Python floats
    fraud_probability = float(fraud_probability)
    anomaly_score = float(anomaly_score)

    risk_score = calculate_risk_score(
        fraud_probability,
        anomaly_score
    )

    # Make sure risk score is also a Python float
    risk_score = float(risk_score)

    decision = make_decision(
        risk_score
    )

    return {
        "fraud_probability": round(
            fraud_probability, 4
        ),

        "anomaly_score": round(
            anomaly_score, 4
        ),

        "risk_score": round(
            risk_score, 4
        ),

        "decision": decision
    }