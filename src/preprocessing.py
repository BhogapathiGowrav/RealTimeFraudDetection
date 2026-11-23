import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==============================
# 1. LOAD DATA
# ==============================

DATA_PATH = "data/credit_card_fraud.parquet"

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded successfully")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# ==============================
# 2. DEFINE TARGET
# ==============================

TARGET = "is_fraud"

y = df[TARGET]


# ==============================
# 3. REMOVE IDENTIFIERS
# ==============================

df = df.drop(columns=[
    "customer_id",
    "transaction_id",
    TARGET
])


# ==============================
# 4. DEFINE NUMERICAL FEATURES
# ==============================

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


# ==============================
# 5. DEFINE CATEGORICAL FEATURES
# ==============================

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


# ==============================
# 6. CREATE PREPROCESSOR
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==============================
# 7. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFraud distribution in training:")
print(y_train.value_counts())

print("\nFraud distribution in testing:")
print(y_test.value_counts())


# ==============================
# 8. FIT PREPROCESSOR
# ==============================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nPreprocessing completed!")

print("Original features:", X_train.shape[1])
print("Processed features:", X_train_processed.shape[1])