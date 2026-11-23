
import os
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier

DATA_PATH = "data/credit_card_fraud.parquet"
MODEL_PATH = "models/xgboost_grouped_experiment.pkl"
SAMPLE_SIZE = 1_000_000
TARGET = "is_fraud"

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

fraud = df[df[TARGET] == 1]
normal = df[df[TARGET] == 0]

fraud_n = round(SAMPLE_SIZE * len(fraud) / len(df))
normal_n = SAMPLE_SIZE - fraud_n

sample = pd.concat([
    fraud.sample(n=fraud_n, random_state=42),
    normal.sample(n=normal_n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

y = sample[TARGET]
groups = sample["customer_id"]
X = sample.drop(
    columns=["customer_id", "transaction_id", TARGET]
)

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

# 70% training, 15% validation, 15% testing,
# with customers separated between every split.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, temp_idx = next(
    split1.split(X, y, groups=groups)
)

temp_groups = groups.iloc[temp_idx]
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=43
)
val_rel, test_rel = next(
    split2.split(
        X.iloc[temp_idx],
        y.iloc[temp_idx],
        groups=temp_groups
    )
)

val_idx = temp_idx[val_rel]
test_idx = temp_idx[test_rel]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse=False),
            categorical_features
        )
    ],
    remainder="passthrough"
)

print("Fitting preprocessor...")
X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

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

print("Training grouped model...")
model.fit(
    X_train_p,
    y_train,
    eval_set=[(X_val_p, y_val)],
    verbose=False
)

print("Best iteration:", model.best_iteration)

def evaluate(name, features, labels):
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n===== {name} =====")
    print("ROC-AUC:", round(roc_auc_score(labels, probabilities), 4))
    print("PR-AUC:", round(average_precision_score(labels, probabilities), 4))
    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))
    print(classification_report(
        labels, predictions, digits=4, zero_division=0
    ))

evaluate("TRAINING", X_train_p, y_train)
evaluate("VALIDATION", X_val_p, y_val)
evaluate("FINAL TEST", X_test_p, y_test)

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

print("\nGrouped experiment saved to:", MODEL_PATH)
print("The existing production/demo model was not overwritten.")
