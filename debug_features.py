#!/usr/bin/env python3
"""
Debug feature alignment and model inputs
"""
import requests
import json
import pandas as pd
from api.utils.io import read_and_normalize
from api.services.pipeline import _align_feature_frame, _lazy_boot_tabular

def debug_features():
    print("🔍 Debugging feature alignment...")
    
    # Load one of the NASA files
    df = read_and_normalize("data/sources/kepler.csv", mission="kepler")
    print(f"Raw normalized data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)[:20]}...")
    
    # Try to align features
    try:
        _lazy_boot_tabular()
        aligned_df = _align_feature_frame(df.head(5))
        print(f"Aligned features shape: {aligned_df.shape}")
        print(f"Feature columns: {list(aligned_df.columns)}")
        
        # Check for missing or NaN values
        missing_cols = aligned_df.columns[aligned_df.isna().all()].tolist()
        partial_missing = aligned_df.columns[aligned_df.isna().any()].tolist()
        
        print(f"Completely missing columns: {missing_cols}")
        print(f"Partially missing columns: {partial_missing}")
        
        # Show sample data
        print(f"\nSample feature values:")
        for col in aligned_df.columns[:10]:
            sample_vals = aligned_df[col].dropna().head(3).tolist()
            print(f"  {col}: {sample_vals}")
            
        # Check if all values are NaN or 0
        non_zero_cols = []
        for col in aligned_df.columns:
            if aligned_df[col].fillna(0).abs().sum() > 0:
                non_zero_cols.append(col)
        
        print(f"\nColumns with non-zero values: {len(non_zero_cols)}/{len(aligned_df.columns)}")
        print(f"Non-zero columns: {non_zero_cols[:10]}...")
        
    except Exception as e:
        print(f"Error in feature alignment: {e}")
        import traceback
        traceback.print_exc()

def test_raw_prediction():
    """Test with manually crafted realistic data"""
    print("\n🧪 Testing with manually crafted data...")
    
    # Create realistic exoplanet data
    test_data = {
        "planet_name": "Test Planet",
        "period_days": 3.52,  # Earth-like
        "rp_rearth": 1.1,     # Slightly bigger than Earth
        "stellar_teff_k": 5800,  # Sun-like star
        "stellar_radius_rsun": 1.0,
        "depth_ppm": 84,      # Transit depth for Earth-size planet
        "duration_hours": 2.5,
        "insolation_earth": 1.2,
        "eq_temp_k": 290,
        "stellar_logg_cgs": 4.44,
        "snr": 15.0,
        "impact": 0.3
    }
    
    payload = {
        "rows": [test_data],
        "mission": "kepler",
        "return_labels": True
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/inference/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Manual prediction successful:")
            print(f"   Classes: {result['classes']}")
            print(f"   Probabilities: {result['proba'][0]}")
            
            # Find predicted class
            proba = result['proba'][0]
            max_idx = proba.index(max(proba))
            pred_class = result['classes'][max_idx]
            confidence = max(proba)
            print(f"   Prediction: {pred_class} ({confidence:.3f})")
        else:
            print(f"❌ Manual prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error in manual prediction: {e}")

if __name__ == "__main__":
    debug_features()
    test_raw_prediction()