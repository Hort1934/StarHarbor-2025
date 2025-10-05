#!/usr/bin/env python3
"""
Debug model output format
"""
from api.services.pipeline import _lazy_boot_tabular, _TAB_MODEL, _FEATURES, _TARGET_MAP
import pandas as pd
import numpy as np

print("🔍 Debugging model output...")

# Load model
_lazy_boot_tabular()

print(f"Model type: {type(_TAB_MODEL)}")
print(f"Model classes: {getattr(_TAB_MODEL, 'classes_', 'Not available')}")
print(f"Target map: {_TARGET_MAP}")
print(f"Number of features: {len(_FEATURES)}")

# Create test data with all features
test_data = pd.DataFrame({col: [1.0] for col in _FEATURES})
print(f"Test data shape: {test_data.shape}")

# Test prediction
try:
    if hasattr(_TAB_MODEL, "predict_proba"):
        proba = _TAB_MODEL.predict_proba(test_data)
        print(f"Predict_proba shape: {proba.shape}")
        print(f"Predict_proba output: {proba}")
        print(f"Proba sum: {proba.sum(axis=1)}")
    
    if hasattr(_TAB_MODEL, "predict"):
        pred = _TAB_MODEL.predict(test_data)
        print(f"Predict output: {pred}")
        
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()