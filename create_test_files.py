#!/usr/bin/env python3
"""
Create diverse test files for StarHarbor exoplanet classification
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_test_files():
    """Create various test CSV files with different exoplanet characteristics"""
    
    test_dir = Path("data/test_samples")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Earth-like planets (should be CONFIRMED)
    earth_like = {
        'planet_name': ['Test-Earth-1', 'Test-Earth-2', 'Test-Earth-3'],
        'koi_name': ['TEST-001.01', 'TEST-002.01', 'TEST-003.01'],
        'koi_disposition': ['CANDIDATE', 'CANDIDATE', 'CANDIDATE'],
        'koi_period': [365.25, 350.0, 380.5],  # Earth-like periods
        'koi_duration': [6.5, 6.2, 6.8],      # Transit duration hours
        'koi_depth': [84, 87, 81],             # Earth-size transit depth (ppm)
        'koi_prad': [1.0, 1.1, 0.95],         # Earth-size radius
        'koi_teq': [288, 295, 280],            # Earth-like temperature
        'koi_insol': [1.0, 0.9, 1.1],         # Earth-like insolation
        'koi_steff': [5778, 5800, 5750],      # Sun-like stars
        'koi_slogg': [4.44, 4.42, 4.46],      # Sun-like gravity
        'koi_srad': [1.0, 1.05, 0.98],        # Sun-like radius
        'koi_model_snr': [15.2, 18.5, 14.8],  # Good signal-to-noise
        'koi_impact': [0.3, 0.25, 0.35],      # Impact parameter
        'koi_fpflag_nt': [0, 0, 0],           # No false positive flags
        'koi_fpflag_ss': [0, 0, 0],
        'koi_fpflag_co': [0, 0, 0],
        'koi_fpflag_ec': [0, 0, 0],
        'ra': [290.5, 315.2, 45.8],
        'dec': [42.3, -15.7, 28.9],
        'koi_kepmag': [12.5, 13.1, 11.8]
    }
    
    df_earth = pd.DataFrame(earth_like)
    df_earth.to_csv(test_dir / "earth_like_planets.csv", index=False)
    print(f"✅ Created earth_like_planets.csv: {len(df_earth)} Earth-like candidates")
    
    # Test 2: Hot Jupiters (should be CONFIRMED)
    hot_jupiters = {
        'planet_name': ['Test-HotJup-1', 'Test-HotJup-2', 'Test-HotJup-3'],
        'koi_name': ['TEST-004.01', 'TEST-005.01', 'TEST-006.01'],
        'koi_disposition': ['CANDIDATE', 'CANDIDATE', 'CANDIDATE'],
        'koi_period': [3.2, 2.8, 4.1],         # Very short periods
        'koi_duration': [3.5, 3.2, 3.8],       # Short transit duration
        'koi_depth': [12000, 15000, 10500],    # Large transit depth (ppm)
        'koi_prad': [11.2, 13.5, 9.8],        # Jupiter-size radius
        'koi_teq': [1500, 1800, 1300],        # Very hot
        'koi_insol': [2000, 3500, 1500],      # High insolation
        'koi_steff': [6200, 6500, 5900],      # Hot stars
        'koi_slogg': [4.3, 4.25, 4.35],       # Stellar gravity
        'koi_srad': [1.2, 1.4, 1.1],          # Slightly larger stars
        'koi_model_snr': [45.2, 52.1, 38.9],  # Very high SNR
        'koi_impact': [0.1, 0.15, 0.08],      # Low impact
        'koi_fpflag_nt': [0, 0, 0],
        'koi_fpflag_ss': [0, 0, 0],
        'koi_fpflag_co': [0, 0, 0],
        'koi_fpflag_ec': [0, 0, 0],
        'ra': [180.3, 220.7, 95.4],
        'dec': [35.6, -22.4, 48.1],
        'koi_kepmag': [10.2, 9.8, 11.5]
    }
    
    df_hotjup = pd.DataFrame(hot_jupiters)
    df_hotjup.to_csv(test_dir / "hot_jupiters.csv", index=False)
    print(f"✅ Created hot_jupiters.csv: {len(df_hotjup)} Hot Jupiter candidates")
    
    # Test 3: False Positives - Eclipsing Binaries
    false_positives = {
        'planet_name': ['Test-EB-1', 'Test-EB-2', 'Test-EB-3', 'Test-EB-4'],
        'koi_name': ['TEST-007.01', 'TEST-008.01', 'TEST-009.01', 'TEST-010.01'],
        'koi_disposition': ['FALSE POSITIVE', 'FALSE POSITIVE', 'FALSE POSITIVE', 'FALSE POSITIVE'],
        'koi_period': [15.6, 8.3, 25.1, 12.9],   # Binary periods
        'koi_duration': [8.5, 12.2, 15.3, 9.7],  # Long duration (secondary eclipse)
        'koi_depth': [25000, 45000, 18000, 32000], # Very deep (stellar eclipse)
        'koi_prad': [25.5, 35.2, 20.1, 28.8],    # Unrealistically large "radius"
        'koi_teq': [2200, 2800, 1900, 2500],     # Very hot (close binary)
        'koi_insol': [5000, 8000, 3500, 6200],   # Extreme insolation
        'koi_steff': [7200, 8500, 6800, 7600],   # Hot binary stars
        'koi_slogg': [4.1, 3.9, 4.15, 4.05],    # Lower gravity
        'koi_srad': [1.8, 2.2, 1.6, 2.0],       # Larger stars
        'koi_model_snr': [85.3, 120.5, 68.9, 95.2], # Very high SNR
        'koi_impact': [0.05, 0.02, 0.08, 0.04],  # Central transit
        'koi_fpflag_nt': [0, 0, 0, 0],
        'koi_fpflag_ss': [1, 1, 1, 1],           # Stellar eclipse flag
        'koi_fpflag_co': [0, 1, 0, 1],           # Some centroid offset
        'koi_fpflag_ec': [0, 0, 1, 0],           # Some ephemeris contamination
        'ra': [78.2, 156.8, 245.3, 302.1],
        'dec': [-8.4, 52.7, -35.2, 18.6],
        'koi_kepmag': [9.5, 8.2, 10.8, 9.1]
    }
    
    df_fp = pd.DataFrame(false_positives)
    df_fp.to_csv(test_dir / "false_positives.csv", index=False)
    print(f"✅ Created false_positives.csv: {len(df_fp)} False positive cases")
    
    # Test 4: Marginal Candidates (should be mixed results)
    marginal = {
        'planet_name': ['Test-Marg-1', 'Test-Marg-2', 'Test-Marg-3', 'Test-Marg-4', 'Test-Marg-5'],
        'koi_name': ['TEST-011.01', 'TEST-012.01', 'TEST-013.01', 'TEST-014.01', 'TEST-015.01'],
        'koi_disposition': ['CANDIDATE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE'],
        'koi_period': [45.2, 125.8, 8.9, 200.5, 67.3],  # Various periods
        'koi_duration': [4.2, 5.8, 2.1, 7.5, 4.9],      # Various durations
        'koi_depth': [150, 320, 45, 680, 220],           # Small transit depths
        'koi_prad': [1.8, 2.5, 0.8, 3.2, 2.1],          # Super-Earth to mini-Neptune
        'koi_teq': [450, 320, 850, 250, 380],            # Various temperatures
        'koi_insol': [25, 5.2, 180, 2.1, 12.8],          # Various insolation
        'koi_steff': [5200, 4800, 6100, 4600, 5500],     # Various stellar types
        'koi_slogg': [4.5, 4.6, 4.2, 4.7, 4.4],         # Various gravity
        'koi_srad': [0.8, 0.7, 1.3, 0.65, 1.0],         # Various stellar sizes
        'koi_model_snr': [8.5, 12.2, 6.8, 15.1, 9.9],   # Lower SNR
        'koi_impact': [0.6, 0.8, 0.45, 0.9, 0.55],      # Higher impact parameters
        'koi_fpflag_nt': [0, 1, 0, 0, 1],               # Some not-transit-like
        'koi_fpflag_ss': [0, 0, 0, 0, 0],
        'koi_fpflag_co': [1, 0, 1, 0, 0],               # Some centroid offset
        'koi_fpflag_ec': [0, 0, 0, 1, 0],               # Some ephemeris issues
        'ra': [25.7, 145.2, 195.8, 275.4, 335.1],
        'dec': [15.8, -45.3, 65.2, -12.7, 38.4],
        'koi_kepmag': [14.2, 15.8, 13.5, 16.1, 14.7]
    }
    
    df_marg = pd.DataFrame(marginal)
    df_marg.to_csv(test_dir / "marginal_candidates.csv", index=False)
    print(f"✅ Created marginal_candidates.csv: {len(df_marg)} Marginal cases")
    
    # Test 5: Mixed quality dataset
    mixed_data = []
    
    # Add some of each type
    mixed_data.extend([
        # One good Earth-like
        ['Mixed-Earth', 'MIX-001.01', 'CANDIDATE', 372, 6.3, 85, 1.05, 290, 0.95, 5780, 4.43, 1.02, 16.8, 0.28, 0, 0, 0, 0, 145.6, 22.1, 12.8],
        # One hot Jupiter  
        ['Mixed-HotJup', 'MIX-002.01', 'CANDIDATE', 3.5, 3.4, 11500, 10.8, 1600, 2200, 6300, 4.28, 1.25, 42.1, 0.12, 0, 0, 0, 0, 234.2, -18.5, 10.5],
        # One false positive
        ['Mixed-FP', 'MIX-003.01', 'FALSE POSITIVE', 18.2, 10.5, 28000, 24.2, 2300, 4800, 7800, 4.08, 1.9, 95.5, 0.03, 0, 1, 1, 0, 88.7, 45.3, 9.2],
        # Two marginal cases
        ['Mixed-Marg1', 'MIX-004.01', 'CANDIDATE', 89.5, 5.1, 180, 2.2, 410, 18.5, 5400, 4.48, 0.92, 10.2, 0.65, 1, 0, 0, 0, 312.4, -8.9, 15.2],
        ['Mixed-Marg2', 'MIX-005.01', 'CANDIDATE', 156.3, 6.8, 95, 1.5, 320, 7.8, 4900, 4.55, 0.78, 9.1, 0.75, 0, 0, 1, 1, 67.8, 35.7, 15.9]
    ])
    
    df_mixed = pd.DataFrame(mixed_data, columns=[
        'planet_name', 'koi_name', 'koi_disposition', 'koi_period', 'koi_duration', 
        'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 
        'koi_srad', 'koi_model_snr', 'koi_impact', 'koi_fpflag_nt', 'koi_fpflag_ss', 
        'koi_fpflag_co', 'koi_fpflag_ec', 'ra', 'dec', 'koi_kepmag'
    ])
    
    df_mixed.to_csv(test_dir / "mixed_quality.csv", index=False)
    print(f"✅ Created mixed_quality.csv: {len(df_mixed)} Mixed quality objects")
    
    # Test 6: Extreme cases (should challenge the model)
    extreme_cases = {
        'planet_name': ['Extreme-Tiny', 'Extreme-Giant', 'Extreme-Hot', 'Extreme-Cold', 'Extreme-LongP'],
        'koi_name': ['EXT-001.01', 'EXT-002.01', 'EXT-003.01', 'EXT-004.01', 'EXT-005.01'],
        'koi_disposition': ['CANDIDATE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE', 'CANDIDATE'],
        'koi_period': [0.8, 15.5, 1.2, 500.0, 800.0],    # Extreme periods
        'koi_duration': [0.5, 8.2, 0.8, 12.5, 18.0],     # Extreme durations
        'koi_depth': [15, 35000, 8000, 25, 180],          # Extreme depths
        'koi_prad': [0.5, 18.5, 8.2, 0.7, 2.8],          # Extreme radii
        'koi_teq': [200, 2500, 3000, 150, 180],           # Extreme temperatures
        'koi_insol': [0.1, 8000, 15000, 0.05, 0.8],       # Extreme insolation
        'koi_steff': [3500, 8000, 9500, 3200, 4000],      # Extreme stellar temps
        'koi_slogg': [4.8, 3.5, 3.2, 4.9, 4.7],          # Extreme gravity
        'koi_srad': [0.4, 2.8, 3.5, 0.35, 0.6],          # Extreme stellar radii
        'koi_model_snr': [5.2, 150.0, 89.5, 4.8, 6.1],   # Extreme SNR
        'koi_impact': [0.95, 0.01, 0.05, 0.98, 0.85],    # Extreme impact
        'koi_fpflag_nt': [1, 0, 1, 1, 0],
        'koi_fpflag_ss': [0, 1, 0, 0, 0],
        'koi_fpflag_co': [1, 1, 1, 1, 1],                 # All have centroid issues
        'koi_fpflag_ec': [0, 0, 1, 0, 1],
        'ra': [359.9, 0.1, 180.0, 90.0, 270.0],
        'dec': [89.5, -89.5, 0.0, 45.0, -45.0],
        'koi_kepmag': [18.5, 6.8, 7.2, 19.2, 17.8]
    }
    
    df_extreme = pd.DataFrame(extreme_cases)
    df_extreme.to_csv(test_dir / "extreme_cases.csv", index=False)
    print(f"✅ Created extreme_cases.csv: {len(df_extreme)} Extreme test cases")
    
    print(f"\n🎯 Created {6} test files in {test_dir}/")
    print("Ready for diverse testing!")

if __name__ == "__main__":
    create_test_files()