
import os
import joblib
import pandas as pd

MODEL_PATH = "models/xgboost_ablation_experiment.pkl"
OUTPUT_PATH = "models/ablation_feature_importance.csv"

print("Loading ablation model...")
saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
preprocessor = saved_data["preprocessor"]

numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# Get the names created by the fitted one-hot encoder
encoder = preprocessor.named_transformers_["cat"]
categorical_names = encoder.get_feature_names_out(
    categorical_features
).tolist()

all_feature_names = list(numeric_features) + categorical_names

# Get feature importance from XGBoost
importances = model.feature_importances_

if len(all_feature_names) != len(importances):
    raise ValueError(
        "Feature-name count does not match model importance count: "
        f"{len(all_feature_names)} vs {len(importances)}"
    )

importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance", ascending=False
).reset_index(drop=True)

importance_df["importance_percent"] = (
    importance_df["importance"] /
    importance_df["importance"].sum() * 100
)

print("\nTOP 25 ENCODED FEATURES")
print(importance_df.head(25).to_string(index=False))

# Aggregate one-hot encoded categories back to original columns
original_features = (
    list(numeric_features) + list(categorical_features)
)

def get_original_feature(encoded_name):
    # Numeric features retain their original names
    if encoded_name in numeric_features:
        return encoded_name

    # OneHotEncoder names usually start with the source column
    # followed by an underscore and the category value.
    matches = [
        col for col in categorical_features
        if encoded_name.startswith(str(col) + "_")
    ]

    if matches:
        return max(matches, key=len)

    return encoded_name

importance_df["original_feature"] = importance_df[
    "feature"
].apply(get_original_feature)

grouped = (
    importance_df.groupby("original_feature", as_index=False)
    ["importance"].sum()
    .sort_values("importance", ascending=False)
)

grouped["importance_percent"] = (
    grouped["importance"] /
    grouped["importance"].sum() * 100
)

print("\nTOP ORIGINAL FEATURES")
print(grouped.head(25).to_string(index=False))

os.makedirs("models", exist_ok=True)
importance_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved encoded feature importances to:", OUTPUT_PATH)
