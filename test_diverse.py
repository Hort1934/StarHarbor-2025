#!/usr/bin/env python3
"""
Test StarHarbor model with diverse test files
"""
import requests
import os
from pathlib import Path
import pandas as pd

API_BASE = "http://localhost:8000"

def test_file_comprehensive(file_path, expected_results=None):
    """Test a file and analyze results in detail"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    filename = os.path.basename(file_path)
    print(f"\n📊 Testing: {filename}")
    print(f"   Path: {file_path}")
    print(f"   Size: {os.path.getsize(file_path)} bytes")
    
    # Read the CSV to see what we're testing
    try:
        df = pd.read_csv(file_path)
        print(f"   Rows: {len(df)}")
        
        # Show expected labels if available
        if 'koi_disposition' in df.columns:
            expected_labels = df['koi_disposition'].value_counts()
            print(f"   Expected labels: {dict(expected_labels)}")
    except Exception as e:
        print(f"   ⚠️ Could not read CSV: {e}")
    
    try:
        # Upload file
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            params = {'mission': 'kepler'}  # Use Kepler format
            
            response = requests.post(
                f"{API_BASE}/inference/upload",
                files=files,
                params=params
            )
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            return None
        
        upload_result = response.json()
        print(f"✅ Upload successful: {upload_result['count']} rows processed")
        
        if not upload_result['rows']:
            print("❌ No data rows to predict")
            return None
        
        # Make predictions
        payload = {
            "rows": upload_result['rows'],
            "mission": "kepler",
            "return_labels": True
        }
        
        pred_response = requests.post(
            f"{API_BASE}/inference/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if pred_response.status_code != 200:
            print(f"❌ Prediction failed: {pred_response.status_code}")
            return None
        
        pred_result = pred_response.json()
        
        # Analyze results
        print(f"🔮 Prediction Results:")
        print(f"   Processed: {pred_result['n']} rows")
        print(f"   Classes: {pred_result['classes']}")
        
        if pred_result['proba']:
            # Count predictions by class
            import numpy as np
            proba_array = np.array(pred_result['proba'])
            predictions = np.argmax(proba_array, axis=1)
            
            class_counts = {}
            confidences = {}
            for i, class_name in enumerate(pred_result['classes']):
                count = np.sum(predictions == i)
                class_counts[class_name] = count
                
                # Average confidence for this class
                if count > 0:
                    class_probs = proba_array[predictions == i, i]
                    confidences[class_name] = np.mean(class_probs)
            
            print(f"   Predicted distribution: {class_counts}")
            print(f"   Average confidence: {dict(confidences)}")
            
            # Calculate metrics if expected results provided
            if expected_results:
                accuracy_info = []
                for expected_class, expected_count in expected_results.items():
                    predicted_count = class_counts.get(expected_class, 0)
                    if expected_count > 0:
                        accuracy = predicted_count / expected_count * 100
                        accuracy_info.append(f"{expected_class}: {accuracy:.1f}%")
                
                if accuracy_info:
                    print(f"   Expected accuracy: {', '.join(accuracy_info)}")
            
            # Show detailed predictions
            print(f"   Sample predictions:")
            for i, (proba_row, row) in enumerate(zip(pred_result['proba'][:5], upload_result['rows'][:5])):
                max_idx = proba_row.index(max(proba_row))
                pred_class = pred_result['classes'][max_idx]
                confidence = max(proba_row)
                name = row.get('planet_name', row.get('koi_name', f'Row_{i}'))
                expected = row.get('koi_disposition', 'Unknown')
                
                status = "✅" if pred_class.upper() in expected.upper() else "❌"
                print(f"     {status} {name}: {pred_class} ({confidence:.3f}) | Expected: {expected}")
        
        return pred_result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🌟 StarHarbor Diverse Testing Suite")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/inference/health")
        if response.status_code != 200:
            print("❌ API server not available. Make sure it's running on localhost:8000")
            return
        print(f"✅ API Health: {response.json()}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return
    
    # Test all our diverse files
    test_dir = Path("data/test_samples")
    
    test_cases = [
        {
            'file': 'earth_like_planets.csv',
            'description': 'Earth-like planets (should be mostly CONFIRMED)',
            'expected': {'confirmed': 3}
        },
        {
            'file': 'hot_jupiters.csv', 
            'description': 'Hot Jupiters (should be mostly CONFIRMED)',
            'expected': {'confirmed': 3}
        },
        {
            'file': 'false_positives.csv',
            'description': 'Eclipsing binaries (should be mostly FALSE POSITIVE)',
            'expected': {'fp': 4}
        },
        {
            'file': 'marginal_candidates.csv',
            'description': 'Marginal cases (should be mixed CANDIDATE/FP)',
            'expected': {'candidate': 3, 'fp': 2}
        },
        {
            'file': 'mixed_quality.csv',
            'description': 'Mixed quality objects (diverse results expected)', 
            'expected': {'confirmed': 2, 'fp': 1, 'candidate': 2}
        },
        {
            'file': 'extreme_cases.csv',
            'description': 'Extreme parameter cases (challenging for model)',
            'expected': {'fp': 3, 'candidate': 2}
        }
    ]
    
    overall_results = {}
    
    for test_case in test_cases:
        file_path = test_dir / test_case['file']
        
        print(f"\n" + "="*60)
        print(f"🧪 {test_case['description']}")
        
        if file_path.exists():
            result = test_file_comprehensive(str(file_path), test_case.get('expected'))
            if result:
                overall_results[test_case['file']] = result
        else:
            print(f"❌ File not found: {file_path}")
    
    # Summary
    print(f"\n" + "="*60)
    print("📈 OVERALL TESTING SUMMARY")
    print("="*60)
    
    for filename, result in overall_results.items():
        print(f"\n📊 {filename}:")
        if result['proba']:
            import numpy as np
            proba_array = np.array(result['proba'])
            predictions = np.argmax(proba_array, axis=1)
            
            for i, class_name in enumerate(result['classes']):
                count = np.sum(predictions == i)
                if count > 0:
                    avg_conf = np.mean(proba_array[predictions == i, i])
                    print(f"   {class_name}: {count} predictions (avg confidence: {avg_conf:.3f})")
    
    print(f"\n✨ Diverse testing completed!")
    print("🎯 Model performance varies across different object types - this is expected and healthy!")

if __name__ == "__main__":
    main()