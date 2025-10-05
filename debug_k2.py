#!/usr/bin/env python3
"""Quick test of K2 file parsing"""

import pandas as pd
from api.utils.io import read_table, normalize_schema

# Test the K2 file directly
print("Testing K2 file parsing...")

# Read the file
df = read_table("data/samples/sample_k2_simple.csv")
print(f"Raw CSV rows: {len(df)}")
print(f"Raw columns: {list(df.columns)}")

# Normalize schema
df_norm = normalize_schema(df, "k2")
print(f"Normalized rows: {len(df_norm)}")
print(f"Normalized columns: {list(df_norm.columns)[:15]}...")

# Check for any string values in numeric columns that could cause issues
print("\nChecking for problematic values...")
feature_cols = ['period_days', 'rp_rearth', 'stellar_teff_k', 'ra_deg', 'dec_deg']
for col in feature_cols:
    if col in df_norm.columns:
        non_numeric = df_norm[col].apply(lambda x: isinstance(x, str) and x.strip() != '')
        if non_numeric.any():
            print(f"  {col}: {df_norm.loc[non_numeric, col].tolist()}")
        else:
            print(f"  {col}: OK")

print(f"\nFirst few rows of key columns:")
key_cols = ['planet_name', 'period_days', 'rp_rearth', 'disposition', 'mission']
available_cols = [c for c in key_cols if c in df_norm.columns]
print(df_norm[available_cols].head())