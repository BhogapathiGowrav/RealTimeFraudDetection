import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/credit_card_fraud.parquet"
MODEL_PATH = "models/xgboost_fraud_model.pkl"

TARGET = "is_fraud"

SAMPLE_SIZE = 1_000_000


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded.")
print("Total rows:", len(df))


# ==========================================
# 3. SAMPLE DATA
# ==========================================

print("\nCreating training sample...")

# Stratified sampling keeps the fraud ratio
# approximately the same as the full dataset.

fraud_df = df[df[TARGET] == 1]
normal_df = df[df[TARGET] == 0]

fraud_sample_size = int(
    SAMPLE_SIZE * len(fraud_df) / len(df)
)

normal_sample_size = SAMPLE_SIZE - fraud_sample_size

fraud_sample = fraud_df.sample(
    n=fraud_sample_size,
    random_state=42
)

normal_sample = normal_df.sample(
    n=normal_sample_size,
    random_state=42
)

sample_df = pd.concat(
    [fraud_sample, normal_sample]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("Sample size:", len(sample_df))
print("\nSample fraud distribution:")
print(sample_df[TARGET].value_counts())


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

y = sample_df[TARGET]

X = sample_df.drop(
    columns=[
        "customer_id",
        "transaction_id",
        TARGET
    ]
)


# ==========================================
# 5. FEATURE GROUPS
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
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# 70% training, 15% validation, 15% final testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))


# ==========================================
# 7. PREPROCESSING
# ==========================================

print("\nEncoding categorical features...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("Processed feature count:", X_train_processed.shape[1])


# ==========================================
# 8. CLASS IMBALANCE
# ==========================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nscale_pos_weight:", round(scale_pos_weight, 4))


# ==========================================
# 9. REGULARIZED XGBOOST MODEL
# ==========================================

print("\nCreating regularized XGBoost model...")

model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=30,
    tree_method="hist",
    random_state=42,
    n_jobs=4
)


# ==========================================
# 10. TRAIN WITH EARLY STOPPING
# ==========================================

print("\nTraining XGBoost...")

model.fit(
    X_train_processed,
    y_train,
    eval_set=[
        (X_val_processed, y_val)
    ],
    verbose=False
)

print("Training completed!")
print("Best iteration:", model.best_iteration)


# ==========================================
# 11. EVALUATE TRAIN / VALIDATION / TEST
# ==========================================

def evaluate_split(name, X_processed, y_true):
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc = roc_auc_score(y_true, probabilities)
    pr = average_precision_score(y_true, probabilities)

    print(f"\n===== {name} =====")
    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))
    print(classification_report(
        y_true, predictions, digits=4, zero_division=0
    ))

    return roc, pr


train_roc, train_pr = evaluate_split(
    "TRAINING DATA",
    X_train_processed,
    y_train
)

val_roc, val_pr = evaluate_split(
    "VALIDATION DATA",
    X_val_processed,
    y_val
)

test_roc, test_pr = evaluate_split(
    "FINAL TEST DATA",
    X_test_processed,
    y_test
)

print("\n===== GENERALIZATION GAP =====")
print("Train-Test ROC-AUC gap:", round(train_roc - test_roc, 4))
print("Train-Test PR-AUC gap:", round(train_pr - test_pr, 4))


# ==========================================
# 12. SAVE MODEL AND PREPROCESSOR
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    },
    MODEL_PATH
)

print("\nModel saved successfully:", MODEL_PATH)
