import joblib
import pandas as pd

from risk_engine import evaluate_transaction


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

print("Loading models...")

xgb_data = joblib.load(
    "models/xgboost_fraud_model.pkl"
)

anomaly_data = joblib.load(
    "models/isolation_forest_model.pkl"
)

xgb_model = xgb_data["model"]
preprocessor = xgb_data["preprocessor"]

anomaly_model = anomaly_data["model"]

print("Models loaded successfully!")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_transaction(transaction):

    """
    Predict fraud risk for a single transaction.
    """

    # Convert dictionary into DataFrame

    df = pd.DataFrame(
        [transaction]
    )

    # Remove identifiers if present

    df = df.drop(
        columns=[
            "customer_id",
            "transaction_id"
        ],
        errors="ignore"
    )

    # ======================================
    # PREPROCESS
    # ======================================

    processed_data = preprocessor.transform(
        df
    )

    # ======================================
    # XGBOOST PREDICTION
    # ======================================

    fraud_probability = xgb_model.predict_proba(
        processed_data
    )[0][1]

    # ======================================
    # ISOLATION FOREST
    # ======================================

    anomaly_score = anomaly_model.decision_function(
        processed_data
    )[0]

    # ======================================
    # RISK ENGINE
    # ======================================

    result = evaluate_transaction(
        fraud_probability,
        anomaly_score
    )

    return result