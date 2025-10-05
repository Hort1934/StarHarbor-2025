#!/usr/bin/env python3
"""
Test script for StarHarbor Exoplanet Vetting API
"""
import requests
import json
import os
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/inference/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_file_upload(file_path, mission=None):
    """Test file upload and parsing"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
        
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            params = {'mission': mission} if mission else {}
            
            response = requests.post(
                f"{API_BASE}/inference/upload",
                files=files,
                params=params
            )
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful:")
            print(f"   Filename: {result['filename']}")
            print(f"   Rows: {result['count']}/{result['total_count']}")
            print(f"   Columns: {len(result['columns'])}")
            print(f"   Truncated: {result['truncated']}")
            return result
        else:
            print(f"❌ Upload failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def test_prediction(data_rows, mission=None):
    """Test prediction endpoint"""
    try:
        payload = {
            "rows": data_rows,
            "mission": mission,
            "return_labels": True
        }
        
        # Debug: check data structure
        if data_rows:
            print(f"   Sample row keys: {list(data_rows[0].keys())[:10]}...")
            # Check for problematic values
            for i, row in enumerate(data_rows[:2]):
                numeric_issues = []
                for key, value in row.items():
                    if isinstance(value, str) and value.strip() and key not in ['mission', 'object_id', 'planet_name', 'disposition', 'discoverymethod']:
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            if not any(x in key.lower() for x in ['name', 'ref', 'str', 'flag', 'type', 'method', 'facility']):
                                numeric_issues.append(f"{key}={value}")
                if numeric_issues:
                    print(f"   Row {i} potential issues: {numeric_issues[:3]}...")
        
        response = requests.post(
            f"{API_BASE}/inference/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Processed: {result['n']} rows")
            print(f"   Classes: {result['classes']}")
            
            if result['proba']:
                print(f"   Predictions:")
                for i, proba in enumerate(result['proba'][:5]):  # Show first 5
                    max_idx = proba.index(max(proba))
                    max_class = result['classes'][max_idx] if result['classes'] else f"Class {max_idx}"
                    print(f"     Row {i}: {max_class} ({max(proba):.3f})")
                    
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def main():
    print("🌟 StarHarbor API Test Suite")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("❌ API server not available. Make sure it's running on localhost:8000")
        return
    
    # Test samples directory
    samples_dir = Path(__file__).parent / "data" / "samples"
    
    # Test Kepler sample
    kepler_file = samples_dir / "sample_exoplanets.csv"
    if kepler_file.exists():
        print("\n📊 Testing Kepler sample file...")
        upload_result = test_file_upload(str(kepler_file), mission="kepler")
        
        if upload_result and upload_result['rows']:
            print("\n🔮 Testing prediction...")
            test_prediction(upload_result['rows'][:10], mission="kepler")
    
    # Test K2 sample
    k2_file = samples_dir / "sample_k2_simple.csv"
    if k2_file.exists():
        print("\n📊 Testing K2 sample file...")
        upload_result = test_file_upload(str(k2_file), mission="k2")
        
        if upload_result and upload_result['rows']:
            print("\n🔮 Testing prediction...")
            test_prediction(upload_result['rows'][:10], mission="k2")
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()