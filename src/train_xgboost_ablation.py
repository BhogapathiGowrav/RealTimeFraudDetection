
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
DATA_PATH = "data/credit_card_fraud.parquet"
OUTPUT_PATH = "models/xgboost_ablation_experiment.pkl"

TARGET = "is_fraud"
GROUP = "customer_id"
SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42

# Diagnostic features to exclude
REMOVE_FEATURES = [
    "merchant_category_vs_history",
    "geo_velocity_bin",
    "geo_distance_bin",
    "session_duration_bin",
    "cards_on_device_30d"
]

# --------------------------------------------------
# 2. Load data and create the reproducible sample
# --------------------------------------------------
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)

# Match the stratified sample used in the grouped experiment
sample = (
    df.groupby(TARGET, group_keys=False)
      .apply(
          lambda group: group.sample(
              n=round(SAMPLE_SIZE * len(group) / len(df)),
              random_state=RANDOM_STATE
          )
      )
      .sample(frac=1, random_state=RANDOM_STATE)
      .reset_index(drop=True)
)

print("Sample shape:", sample.shape)
print("Fraud rate:", round(sample[TARGET].mean(), 5))

# --------------------------------------------------
# 3. Select features
# --------------------------------------------------
exclude = {TARGET, GROUP, "transaction_id"}
exclude.update(REMOVE_FEATURES)

feature_cols = [
    col for col in sample.columns
    if col not in exclude
]

X = sample[feature_cols]
y = sample[TARGET].astype(int)
groups = sample[GROUP]

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

print("\nRemoved features:", REMOVE_FEATURES)
print("Remaining features:", len(feature_cols))
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# --------------------------------------------------
# 4. Split by customer: 70% / 15% / 15%
# --------------------------------------------------
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, temp_idx = next(
    split1.split(X, y, groups=groups)
)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
groups_temp = groups.iloc[temp_idx]

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=43
)
val_rel, test_rel = next(
    split2.split(X_temp, y_temp, groups=groups_temp)
)

X_val, y_val = X_temp.iloc[val_rel], y_temp.iloc[val_rel]
X_test, y_test = X_temp.iloc[test_rel], y_temp.iloc[test_rel]

print("\nTrain rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# Confirm no customer appears in multiple splits
train_customers = set(groups.iloc[train_idx])
val_customers = set(groups_temp.iloc[val_rel])
test_customers = set(groups_temp.iloc[test_rel])

assert train_customers.isdisjoint(val_customers)
assert train_customers.isdisjoint(test_customers)
assert val_customers.isdisjoint(test_customers)

print("Customer overlap check: PASSED")

# --------------------------------------------------
# 5. Preprocess using training data only
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse=False),
            categorical_features
        )
    ],
    remainder="drop"
)

print("\nFitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# --------------------------------------------------
# 6. Train regularized XGBoost
# --------------------------------------------------
negatives = int((y_train == 0).sum())
positives = int((y_train == 1).sum())
scale_weight = negatives / max(positives, 1)

model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5,
    scale_pos_weight=scale_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
    early_stopping_rounds=30
)

print("\nTraining ablation model...")
model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    verbose=50
)

# --------------------------------------------------
# 7. Evaluate on train, validation and test
# --------------------------------------------------
def evaluate(name, X_data, y_data):
    probabilities = model.predict_proba(X_data)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n--- {name} RESULTS ---")
    print("ROC-AUC:", round(roc_auc_score(y_data, probabilities), 5))
    print(
        "PR-AUC:",
        round(average_precision_score(y_data, probabilities), 5)
    )
    print("Confusion matrix:\n", confusion_matrix(y_data, predictions))
    print(
        classification_report(
            y_data, predictions, digits=4, zero_division=0
        )
    )

evaluate("TRAIN", X_train_processed, y_train)
evaluate("VALIDATION", X_val_processed, y_val)
evaluate("TEST", X_test_processed, y_test)

# --------------------------------------------------
# 8. Save separately — preserve the API model
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_cols,
        "removed_features": REMOVE_FEATURES
    },
    OUTPUT_PATH
)

print("\nAblation experiment saved to:", OUTPUT_PATH)
print("Existing API model was NOT overwritten.")
