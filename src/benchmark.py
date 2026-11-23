import time
import statistics
import requests


URL = "http://127.0.0.1:8000/predict"


transaction = {
    "transaction_amount": 250.0,
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
    "account_credential_change_recency": "no_recent_change",
    "card_present_cnp": "card_present_chip",
    "order_shipping_speed": "standard",
    "account_age_days": 1200,
    "chargeback_history_count": 0,
    "segment": "A_established",
    "customer_type": "customer_based"
}


# Warm-up requests
for _ in range(5):
    requests.post(URL, json=transaction)


times = []

print("Running latency benchmark...\n")


for i in range(50):

    start = time.perf_counter()

    response = requests.post(
        URL,
        json=transaction
    )

    end = time.perf_counter()

    if response.status_code == 200:

        latency_ms = (
            end - start
        ) * 1000

        times.append(latency_ms)

        print(
            f"Request {i + 1}: "
            f"{latency_ms:.2f} ms"
        )


print("\n==============================")
print("LATENCY RESULTS")
print("==============================")

print(
    "Minimum:",
    f"{min(times):.2f} ms"
)

print(
    "Maximum:",
    f"{max(times):.2f} ms"
)

print(
    "Average:",
    f"{statistics.mean(times):.2f} ms"
)

print(
    "Median:",
    f"{statistics.median(times):.2f} ms"
)

print(
    "95th percentile:",
    f"{statistics.quantiles(times, n=20)[18]:.2f} ms"
)