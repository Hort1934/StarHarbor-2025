#!/usr/bin/env python3
"""
Check training data distribution
"""
import pandas as pd

print("🔍 Checking training data distribution...")

try:
    y_train = pd.read_parquet("models/y_train.parquet")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Label distribution:")
    print(y_train.value_counts())
    print(f"Unique values: {sorted(y_train.iloc[:, 0].unique())}")
    
    y_test = pd.read_parquet("models/y_test.parquet")
    print(f"\nTest labels shape: {y_test.shape}")
    print(f"Test distribution:")
    print(y_test.value_counts())
    
    y_val = pd.read_parquet("models/y_val.parquet")
    print(f"\nValidation labels shape: {y_val.shape}")
    print(f"Validation distribution:")
    print(y_val.value_counts())
    
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()