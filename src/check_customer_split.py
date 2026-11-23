
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "data/credit_card_fraud.parquet"
SAMPLE_SIZE = 1_000_000

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

# Reproduce a manageable, representative sample.
fraud = df[df["is_fraud"] == 1]
normal = df[df["is_fraud"] == 0]

fraud_n = round(SAMPLE_SIZE * len(fraud) / len(df))
normal_n = SAMPLE_SIZE - fraud_n

sample = pd.concat([
    fraud.sample(n=fraud_n, random_state=42),
    normal.sample(n=normal_n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

X = sample.drop(
    columns=["is_fraud", "customer_id", "transaction_id"]
)
y = sample["is_fraud"]
groups = sample["customer_id"]

# Split by customer, not by individual transaction.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, temp_idx = next(
    split1.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
g_train = groups.iloc[train_idx]

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]
g_temp = groups.iloc[temp_idx]

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=43
)
val_rel, test_rel = next(
    split2.split(X_temp, y_temp, groups=g_temp)
)

g_val = g_temp.iloc[val_rel]
g_test = g_temp.iloc[test_rel]

print("\n===== CUSTOMER-SEPARATED SPLIT =====")
print("Training rows:", len(train_idx))
print("Validation rows:", len(val_rel))
print("Testing rows:", len(test_rel))

print("\nFraud rates:")
print("Train:", round(y_train.mean() * 100, 3), "%")
print("Validation:", round(y_temp.iloc[val_rel].mean() * 100, 3), "%")
print("Test:", round(y_temp.iloc[test_rel].mean() * 100, 3), "%")

train_customers = set(g_train)
val_customers = set(g_val)
test_customers = set(g_test)

print("\nCustomer overlap:")
print("Train / validation:", len(train_customers & val_customers))
print("Train / test:", len(train_customers & test_customers))
print("Validation / test:", len(val_customers & test_customers))

print("\nCheck completed.")
