# Real-Time AI Fraud Detection in Banking

An academic prototype developed at **Presidency University, Bengaluru** (B.Tech CSE, 2026–27) for real-time banking transaction risk assessment. The system combines supervised fraud classification (**XGBoost**) with unsupervised anomaly detection (**Isolation Forest**), unifies them via a configurable **Risk and Decision Engine**, and exposes real-time inference through a **FastAPI** REST service (`POST /predict`).

---

## System Architecture

```text
Transaction (POST /predict)
   └── FastAPI Input & Validation
         └── Preprocessing (ColumnTransformer)
               ├── XGBoost Fraud Classifier ──────────► Fraud Probability (P)
               └── Isolation Forest Anomaly Detector ─► Anomaly Score (A)
                     └── Risk Engine: N = clamp(0.5 - A, 0, 1) | R = 0.70P + 0.30N
                           └── Decision Engine ──► APPROVE | FLAG_VERIFY | BLOCK
