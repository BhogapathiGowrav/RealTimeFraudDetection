from prediction import predict_transaction


transaction = {
    "transaction_amount": 250.00,
    "transaction_amount_bin": "medium",
    "avg_amount_deviation_sigma": 1.2,

    "merchant_category": "grocery",
    "merchant_risk_score": "low_risk",
    "merchant_category_vs_history": "frequent_category",

    "geo_velocity_kmh": 5.0,
    "geo_velocity_bin": "normal",

    "geo_distance_km": 2.0,
    "geo_distance_bin": "same_location",

    "merchant_location": "home_city",
    "country_consistency": "same_country",

    "ip_address_type": "residential",
    "device_fingerprint_match": "known_device",

    "session_duration_sec": 120.0,
    "session_duration_bin": "normal",

    "network_carrier_type": "wifi",

    "cards_on_device_30d": 1,

    "cvv_match_status": "cvv2_match",
    "three_ds_auth_result": "challenge_pass",
    "tokenization_used": "issuer_token",

    "failed_attempts_before_success": 0,

    "transaction_velocity_1h": "within_normal",

    "time_of_transaction": "afternoon_12_18",

    "account_credential_change_recency":
        "no_recent_change",

    "card_present_cnp": "card_present_chip",

    "order_shipping_speed": "standard",

    "account_age_days": 1200,

    "chargeback_history_count": 0,

    "segment": "A_established",

    "customer_type": "customer_based"
}


result = predict_transaction(
    transaction
)

print("\n==============================")
print("FRAUD DETECTION RESULT")
print("==============================")

print(
    "Fraud Probability:",
    result["fraud_probability"]
)

print(
    "Anomaly Score:",
    result["anomaly_score"]
)

print(
    "Risk Score:",
    result["risk_score"]
)

print(
    "Decision:",
    result["decision"]
)