from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os

# Allow importing modules from src
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

from prediction import predict_transaction


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Real-Time AI Fraud Detection API",
    description="AI-powered banking transaction fraud detection system",
    version="1.0"
)


# ==========================================
# REQUEST MODEL
# ==========================================

class Transaction(BaseModel):

    transaction_amount: float
    transaction_amount_bin: str
    avg_amount_deviation_sigma: float

    merchant_category: str
    merchant_risk_score: str
    merchant_category_vs_history: str

    geo_velocity_kmh: float
    geo_velocity_bin: str

    geo_distance_km: float
    geo_distance_bin: str

    merchant_location: str
    country_consistency: str

    ip_address_type: str
    device_fingerprint_match: str

    session_duration_sec: float
    session_duration_bin: str

    network_carrier_type: str

    cards_on_device_30d: int

    cvv_match_status: str
    three_ds_auth_result: str
    tokenization_used: str

    failed_attempts_before_success: int

    transaction_velocity_1h: str

    time_of_transaction: str

    account_credential_change_recency: str

    card_present_cnp: str

    order_shipping_speed: str

    account_age_days: int

    chargeback_history_count: int

    segment: str

    customer_type: str


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "system": "Real-Time AI Fraud Detection",
        "status": "running",
        "message": "Fraud detection API is ready"
    }


# ==========================================
# FRAUD PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(transaction: Transaction):

    transaction_data = transaction.dict()

    result = predict_transaction(
        transaction_data
    )

    return result