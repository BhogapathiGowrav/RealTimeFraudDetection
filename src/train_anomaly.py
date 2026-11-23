import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/credit_card_fraud.parquet"
MODEL_PATH = "models/isolation_forest_model.pkl"

SAMPLE_SIZE = 500_000


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded.")
print("Total rows:", len(df))


# ==========================================
# 3. CREATE SAMPLE
# ==========================================

print("\nCreating sample...")

# Use legitimate transactions to learn
# normal transaction behavior.

normal_df = df[df["is_fraud"] == 0]

normal_sample = normal_df.sample(
    n=SAMPLE_SIZE,
    random_state=42
)

# Also select fraud transactions for
# evaluating whether the anomaly model
# assigns different scores to them.

fraud_sample = df[df["is_fraud"] == 1].sample(
    n=50_000,
    random_state=42
)

evaluation_df = pd.concat(
    [normal_sample, fraud_sample]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 4. REMOVE IDENTIFIERS AND TARGET
# ==========================================

features_to_remove = [
    "customer_id",
    "transaction_id",
    "is_fraud"
]

X_normal = normal_sample.drop(
    columns=features_to_remove
)

X_evaluation = evaluation_df.drop(
    columns=features_to_remove
)

y_evaluation = evaluation_df["is_fraud"]


# ==========================================
# 5. FEATURE DEFINITIONS
# ==========================================

numeric_features = [
    "transaction_amount",
    "avg_amount_deviation_sigma",
    "geo_velocity_kmh",
    "geo_distance_km",
    "session_duration_sec",
    "cards_on_device_30d",
    "failed_attempts_before_success",
    "account_age_days",
    "chargeback_history_count"
]

categorical_features = [
    "transaction_amount_bin",
    "merchant_category",
    "merchant_risk_score",
    "merchant_category_vs_history",
    "geo_velocity_bin",
    "geo_distance_bin",
    "merchant_location",
    "country_consistency",
    "ip_address_type",
    "device_fingerprint_match",
    "session_duration_bin",
    "network_carrier_type",
    "cvv_match_status",
    "three_ds_auth_result",
    "tokenization_used",
    "transaction_velocity_1h",
    "time_of_transaction",
    "account_credential_change_recency",
    "card_present_cnp",
    "order_shipping_speed",
    "segment",
    "customer_type"
]


# ==========================================
# 6. LOAD XGBOOST PREPROCESSOR
# ==========================================

print("\nLoading preprocessing configuration...")

xgb_data = joblib.load(
    "models/xgboost_fraud_model.pkl"
)

preprocessor = xgb_data["preprocessor"]


# ==========================================
# 7. TRANSFORM DATA
# ==========================================

print("Transforming normal transactions...")

X_normal_processed = preprocessor.transform(
    X_normal
)

print("Transforming evaluation transactions...")

X_evaluation_processed = preprocessor.transform(
    X_evaluation
)

print(
    "Processed feature count:",
    X_normal_processed.shape[1]
)


# ==========================================
# 8. CREATE ISOLATION FOREST
# ==========================================

print("\nCreating Isolation Forest...")

anomaly_model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=4
)


# ==========================================
# 9. TRAIN
# ==========================================

print("\nTraining Isolation Forest...")
print("This may take some time.")

anomaly_model.fit(
    X_normal_processed
)

print("\nIsolation Forest training completed!")


# ==========================================
# 10. GENERATE ANOMALY SCORES
# ==========================================

print("\nCalculating anomaly scores...")

scores = anomaly_model.decision_function(
    X_evaluation_processed
)

predictions = anomaly_model.predict(
    X_evaluation_processed
)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

anomaly_prediction = (
    predictions == -1
).astype(int)


# ==========================================
# 11. EVALUATION
# ==========================================

print("\n==============================")
print("ANOMALY MODEL EVALUATION")
print("==============================")

print("\nClassification Report:")

print(
    classification_report(
        y_evaluation,
        anomaly_prediction,
        digits=4
    )
)


# ==========================================
# 12. COMPARE ANOMALY SCORES
# ==========================================

normal_scores = scores[
    y_evaluation.values == 0
]

fraud_scores = scores[
    y_evaluation.values == 1
]

print("\nAverage anomaly score:")
print(
    "Normal transactions:",
    round(normal_scores.mean(), 4)
)

print(
    "Fraud transactions:",
    round(fraud_scores.mean(), 4)
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    {
        "model": anomaly_model,
        "preprocessor": preprocessor
    },
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)