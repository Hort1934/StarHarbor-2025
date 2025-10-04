#!/usr/bin/env python3
"""
Test API response with a real mixed quality file
"""
import requests
import json

API_BASE = "http://localhost:8000"

def test_mixed_quality():
    """Test the exact same file that frontend is using"""
    
    # Upload the file first
    print("📤 Testing file upload...")
    with open("data/test_samples/mixed_quality.csv", 'rb') as f:
        files = {'file': f}
        params = {'mission': 'kepler'}
        upload_response = requests.post(f"{API_BASE}/inference/upload", files=files, params=params)
    
    if upload_response.status_code != 200:
        print(f"❌ Upload failed: {upload_response.status_code}")
        print(upload_response.text)
        return
    
    upload_data = upload_response.json()
    print(f"✅ Upload successful: {upload_data['count']} rows")
    print(f"   Filename: {upload_data['filename']}")
    print(f"   Columns: {len(upload_data['columns'])}")
    
    # Now test prediction
    print("\n🔮 Testing prediction...")
    predict_payload = {
        "rows": upload_data["rows"],
        "mission": "kepler",
        "return_labels": True
    }
    
    predict_response = requests.post(
        f"{API_BASE}/inference/predict",
        json=predict_payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if predict_response.status_code != 200:
        print(f"❌ Prediction failed: {predict_response.status_code}")
        print(predict_response.text)
        return
    
    predict_data = predict_response.json()
    print(f"✅ Prediction successful!")
    print(f"   Classes: {predict_data.get('classes', [])}")
    print(f"   Predictions count: {len(predict_data.get('proba', []))}")
    
    # Analyze predictions
    if 'proba' in predict_data and 'classes' in predict_data:
        proba_list = predict_data['proba']
        classes = predict_data['classes']
        
        # Count predictions same way as frontend
        confirmed = 0
        candidates = 0
        false_positives = 0
        
        for proba in proba_list:
            if proba and len(proba) >= 3:
                max_index = proba.index(max(proba))
                predicted_class = classes[max_index]
                confidence = max(proba)
                
                print(f"   Prediction: {predicted_class} ({confidence:.3f})")
                
                if predicted_class == 'confirmed':
                    confirmed += 1
                elif predicted_class == 'candidate':
                    candidates += 1
                elif predicted_class == 'fp':
                    false_positives += 1
        
        print(f"\n📊 Final Statistics:")
        print(f"   Confirmed: {confirmed}")
        print(f"   Candidates: {candidates}")
        print(f"   False Positives: {false_positives}")
        print(f"   Total: {confirmed + candidates + false_positives}")
        
        detection_rate = (confirmed + candidates) / len(proba_list) * 100 if proba_list else 0
        print(f"   Detection Rate: {detection_rate:.1f}%")
        
        # Check if this matches our previous test results
        if false_positives > 0 and candidates > 0:
            print("✅ Results look diverse - model is working correctly!")
        elif confirmed == len(proba_list):
            print("⚠️  All predictions are 'confirmed' - frontend bug confirmed!")
        else:
            print("🤔 Mixed results - let's see...")
    
    return predict_data

if __name__ == "__main__":
    test_mixed_quality()