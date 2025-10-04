#!/usr/bin/env python3
"""
Final validation test for StarHarbor exoplanet vetting system
Tests all major functionality after model retraining
"""

import requests
import json
import os
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_api_health():
    """Test basic API health"""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print(f"   ✅ API Health: {response.json()}")
            return True
        else:
            print(f"   ❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False

def test_original_nasa_data():
    """Test with original NASA training data"""
    print("\n📡 Testing Original NASA Data...")
    
    # Test with a Kepler file
    kepler_file = "data/sources/kepler.csv"
    if os.path.exists(kepler_file):
        try:
            with open(kepler_file, 'rb') as f:
                files = {'file': f}
                data = {'mission': 'kepler'}
                response = requests.post(f"{API_BASE}/upload-dataset", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Kepler upload successful: {result['rows_processed']} rows")
                
                # Check class distribution
                classes = result.get('class_distribution', {})
                print(f"   📊 Class distribution: {classes}")
                
                # Expect diverse classes now
                unique_classes = len([k for k, v in classes.items() if v > 0])
                if unique_classes >= 2:
                    print(f"   ✅ Good diversity: {unique_classes} different classes predicted")
                    return True
                else:
                    print(f"   ⚠️  Limited diversity: only {unique_classes} class(es)")
                    return False
            else:
                print(f"   ❌ Kepler upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Kepler test error: {e}")
            return False
    else:
        print(f"   ⚠️  Kepler file not found: {kepler_file}")
        return False

def test_synthetic_diversity():
    """Test synthetic data diversity"""
    print("\n🔬 Testing Synthetic Data Diversity...")
    
    test_files = [
        ("earth_like_planets.csv", "candidate"),
        ("false_positives.csv", "fp"),
        ("mixed_quality.csv", "mixed")
    ]
    
    all_passed = True
    total_diversity = {}
    
    for filename, expected_type in test_files:
        filepath = f"data/test_samples/{filename}"
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    files = {'file': f}
                    data = {'mission': 'kepler'}
                    response = requests.post(f"{API_BASE}/upload-dataset", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    classes = result.get('class_distribution', {})
                    print(f"   ✅ {filename}: {classes}")
                    
                    # Accumulate diversity stats
                    for cls, count in classes.items():
                        total_diversity[cls] = total_diversity.get(cls, 0) + count
                        
                else:
                    print(f"   ❌ {filename} failed: {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {filename} error: {e}")
                all_passed = False
        else:
            print(f"   ⚠️  File not found: {filepath}")
            all_passed = False
    
    print(f"\n   📊 Total diversity across synthetic tests: {total_diversity}")
    unique_classes = len([k for k, v in total_diversity.items() if v > 0])
    print(f"   🎯 Classes represented: {unique_classes}/3")
    
    return all_passed and unique_classes >= 2

def test_model_endpoints():
    """Test model information endpoints"""
    print("\n🤖 Testing Model Endpoints...")
    
    endpoints = [
        "/model/info",
        "/model/features",
        "/model/classes"
    ]
    
    all_passed = True
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE}{endpoint}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {endpoint}: {len(str(data))} chars response")
            else:
                print(f"   ❌ {endpoint} failed: {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {endpoint} error: {e}")
            all_passed = False
    
    return all_passed

def test_metrics_endpoint():
    """Test metrics endpoint"""
    print("\n📈 Testing Metrics...")
    
    try:
        response = requests.get(f"{API_BASE}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print(f"   ✅ Metrics available: {len(metrics)} metric types")
            
            # Check for key metrics
            key_metrics = ['accuracy', 'precision', 'recall', 'f1']
            found_metrics = [m for m in key_metrics if any(m in str(metrics).lower() for m in [m])]
            print(f"   📊 Key metrics found: {found_metrics}")
            return True
        else:
            print(f"   ❌ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Metrics error: {e}")
        return False

def main():
    """Run full validation suite"""
    print("🌟 StarHarbor Final Validation Suite")
    print("=" * 60)
    
    tests = [
        ("API Health", test_api_health),
        ("Original NASA Data", test_original_nasa_data),
        ("Synthetic Diversity", test_synthetic_diversity),
        ("Model Endpoints", test_model_endpoints),
        ("Metrics", test_metrics_endpoint)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
        
    print("\n" + "="*60)
    print("🏆 FINAL VALIDATION RESULTS")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! StarHarbor system is fully operational!")
        print("🚀 Ready for NASA Space Apps Challenge 2025!")
    elif passed >= total * 0.8:
        print("✅ System mostly functional - minor issues detected")
    else:
        print("⚠️  System has significant issues - needs attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)