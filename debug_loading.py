#!/usr/bin/env python3
"""
Debug model loading
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from api.utils.constants import (
    PREPROCESSOR_PATH, FEATURE_LIST_PATH, TARGET_MAP_PATH, TAB_MODEL_PATH,
    assert_artifacts_available, log_artifact_paths
)

print("🔍 Debugging model loading...")

# Check if files exist
print(f"Preprocessor: {PREPROCESSOR_PATH.exists()} - {PREPROCESSOR_PATH}")
print(f"Features: {FEATURE_LIST_PATH.exists()} - {FEATURE_LIST_PATH}")
print(f"Target map: {TARGET_MAP_PATH.exists()} - {TARGET_MAP_PATH}")
print(f"Model: {TAB_MODEL_PATH.exists()} - {TAB_MODEL_PATH}")

# Try to load manually
try:
    assert_artifacts_available()
    print("✅ All artifacts available")
except Exception as e:
    print(f"❌ Missing artifacts: {e}")

try:
    import json
    with open(FEATURE_LIST_PATH) as f:
        features = json.load(f)
    print(f"✅ Features loaded: {len(features)} features")
    print(f"   First 10: {features[:10]}")
except Exception as e:
    print(f"❌ Failed to load features: {e}")

try:
    with open(TARGET_MAP_PATH) as f:
        target_map = json.load(f)
    print(f"✅ Target map loaded: {target_map}")
except Exception as e:
    print(f"❌ Failed to load target map: {e}")

try:
    import joblib
    model = joblib.load(TAB_MODEL_PATH)
    print(f"✅ Model loaded: {type(model)}")
    print(f"   Classes: {getattr(model, 'classes_', 'Not available')}")
    if hasattr(model, 'n_classes_'):
        print(f"   Number of classes: {model.n_classes_}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()