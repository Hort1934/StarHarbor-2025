#!/usr/bin/env python3
"""
Quick test of single prediction with real Kepler data
"""
import requests
import time

# Wait for server to start
time.sleep(3)

API_BASE = "http://localhost:8000"

# Test with one real Kepler exoplanet
test_data = {
    "kepid": 10797460,
    "koi_name": "K00752.01", 
    "planet_name": "Kepler-227 b",
    "period_days": 9.488,
    "duration_hours": 2.96,
    "depth_ppm": 615.8,
    "snr": 35.8,
    "impact": 0.146,
    "rp_rearth": 2.26,
    "eq_temp_k": 793,
    "insolation_earth": 93.6,
    "stellar_teff_k": 5455,
    "stellar_logg_cgs": 4.467,
    "stellar_radius_rsun": 0.927,
    "mag_kepler": 15.347,
    "flag_not_transit_like": 0,
    "flag_eclipse": 0,
    "flag_centroid": 0, 
    "flag_ephemeris_match": 0,
    "label_raw": "CONFIRMED"
}

payload = {
    "rows": [test_data],
    "mission": "kepler",
    "return_labels": True
}

try:
    print("🧪 Testing with real Kepler confirmed planet...")
    response = requests.post(
        f"{API_BASE}/inference/predict",
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Classes: {result['classes']}")
        proba = result['proba'][0]
        print(f"   Probabilities: {[f'{p:.3f}' for p in proba]}")
        
        # Find predicted class
        max_idx = proba.index(max(proba))
        pred_class = result['classes'][max_idx]
        confidence = max(proba)
        print(f"   Prediction: {pred_class} ({confidence:.3f})")
        print(f"   True label: CONFIRMED")
        
        if pred_class != "fp":
            print("🎉 Model is working! Not everything is 'fp'!")
        else:
            print("⚠️  Still predicting 'fp' - model may need retraining")
            
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")