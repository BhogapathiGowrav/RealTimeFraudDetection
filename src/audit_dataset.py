
import pandas as pd

DATA_PATH = "data/credit_card_fraud.parquet"
TARGET = "is_fraud"

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sort_values(ascending=False).head(15))

print("\n===== TARGET DISTRIBUTION =====")
print(df[TARGET].value_counts(dropna=False))
print("\nFraud rate:", round(df[TARGET].mean() * 100, 4), "%")

print("\n===== CUSTOMER OVERLAP RISK =====")
if "customer_id" in df.columns:
    print("Unique customers:", df["customer_id"].nunique())
    print(
        "Rows per customer:\n",
        df.groupby("customer_id").size().describe()
    )

print("\n===== FEATURE AUDIT SAMPLE =====")

# Use a reproducible sample to keep this audit manageable.
sample = df.sample(
    n=min(300_000, len(df)),
    random_state=42
)

for col in sample.columns:
    if col in [TARGET, "customer_id", "transaction_id"]:
        continue

    grouped = sample.groupby(col, dropna=False)[TARGET].agg(
        ["count", "mean"]
    )

    if len(grouped) <= 30:
        grouped = grouped.sort_values("mean", ascending=False)

        print(f"\nFeature: {col}")
        print("Distinct values:", sample[col].nunique(dropna=False))
        print("Highest observed fraud rates:")
        print(grouped.head(10).to_string())

    else:
        print(f"\nFeature: {col}")
        print("Distinct values:", sample[col].nunique(dropna=False))
        print(
            sample.groupby(TARGET)[col]
            .describe()
            .to_string()
        )

print("\nAudit completed.")
