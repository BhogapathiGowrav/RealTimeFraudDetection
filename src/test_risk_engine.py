from risk_engine import evaluate_transaction


# ==========================================
# TEST 1 — LOW RISK
# ==========================================

result = evaluate_transaction(
    fraud_probability=0.05,
    anomaly_score=0.20
)

print("\nLOW RISK TRANSACTION")
print(result)


# ==========================================
# TEST 2 — MEDIUM RISK
# ==========================================

result = evaluate_transaction(
    fraud_probability=0.50,
    anomaly_score=-0.05
)

print("\nMEDIUM RISK TRANSACTION")
print(result)


# ==========================================
# TEST 3 — HIGH RISK
# ==========================================

result = evaluate_transaction(
    fraud_probability=0.95,
    anomaly_score=-0.30
)

print("\nHIGH RISK TRANSACTION")
print(result)