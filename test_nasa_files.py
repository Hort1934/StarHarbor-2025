#!/usr/bin/env python3
"""
Test script for NASA source files from the sources directory
"""
import requests
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

def test_nasa_file(file_path, mission=None):
    """Test NASA file upload and prediction"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
        
    print(f"\n📊 Testing NASA file: {os.path.basename(file_path)}")
    print(f"   Mission: {mission}")
    print(f"   Size: {os.path.getsize(file_path)} bytes")
    
    try:
        # Upload file
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            params = {'mission': mission} if mission else {}
            
            response = requests.post(
                f"{API_BASE}/inference/upload",
                files=files,
                params=params
            )
            
        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ Upload successful:")
            print(f"   Filename: {upload_result['filename']}")
            print(f"   Rows: {upload_result['count']}/{upload_result['total_count']}")
            print(f"   Columns: {len(upload_result['columns'])}")
            print(f"   Column names: {upload_result['columns'][:10]}...")
            
            # Test prediction
            if upload_result['rows']:
                print(f"\n🔮 Testing prediction with {len(upload_result['rows'])} rows...")
                
                payload = {
                    "rows": upload_result['rows'][:50],  # Test with first 50 rows
                    "mission": mission,
                    "return_labels": True
                }
                
                pred_response = requests.post(
                    f"{API_BASE}/inference/predict",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if pred_response.status_code == 200:
                    pred_result = pred_response.json()
                    print(f"✅ Prediction successful:")
                    print(f"   Processed: {pred_result['n']} rows")
                    print(f"   Classes: {pred_result['classes']}")
                    
                    if pred_result['proba']:
                        # Count predictions by class
                        import numpy as np
                        proba_array = np.array(pred_result['proba'])
                        predictions = np.argmax(proba_array, axis=1)
                        
                        class_counts = {}
                        if pred_result['classes']:
                            for i, class_name in enumerate(pred_result['classes']):
                                count = np.sum(predictions == i)
                                class_counts[class_name] = count
                        
                        print(f"   Class distribution: {class_counts}")
                        
                        # Show first few predictions
                        print(f"   Sample predictions:")
                        for i, (proba_row, row) in enumerate(zip(pred_result['proba'][:5], upload_result['rows'][:5])):
                            max_idx = proba_row.index(max(proba_row))
                            pred_class = pred_result['classes'][max_idx] if pred_result['classes'] else f"Class_{max_idx}"
                            confidence = max(proba_row)
                            name = row.get('pl_name', row.get('Planet Name', row.get('kepoi_name', f'Row_{i}')))
                            print(f"     {name}: {pred_class} ({confidence:.3f})")
                    
                    return pred_result
                else:
                    print(f"❌ Prediction failed: {pred_response.status_code}")
                    try:
                        error = pred_response.json()
                        print(f"   Error: {error}")
                    except:
                        print(f"   Response: {pred_response.text}")
            else:
                print("❌ No data rows to predict")
            
            return upload_result
        else:
            print(f"❌ Upload failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🌟 StarHarbor NASA Files Test")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("❌ API server not available. Make sure it's running on localhost:8000")
        return
    
    # Test NASA source files
    sources_dir = Path(__file__).parent / "data" / "sources"
    
    nasa_files = [
        ("kepler.csv", "kepler"),
        ("k2.csv", "k2"),
        ("tess.csv", "tess")
    ]
    
    for filename, mission in nasa_files:
        file_path = sources_dir / filename
        if file_path.exists():
            test_nasa_file(str(file_path), mission)
        else:
            print(f"❌ File not found: {file_path}")
    
    print("\n✨ NASA files test completed!")

if __name__ == "__main__":
    main()