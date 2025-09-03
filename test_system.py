#!/usr/bin/env python3

import subprocess
import time
import requests
import json

def run_command(cmd):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_api_locally():
    """Test the FastAPI service locally"""
    print("Testing API locally...")
    
    # Start the API in background
    print("Starting API server...")
    api_process = subprocess.Popen(
        ["python", "api_service.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code}")
        
        # Test prediction with sample log
        sample_log = """
        2023-10-27T12:00:00.000Z [INFO] Starting CI pipeline
        2023-10-27T12:00:01.000Z [INFO] Checkout successful
        2023-10-27T12:00:02.000Z [ERROR] pytest failed
        2023-10-27T12:00:03.000Z [ERROR] Test test_divide failed: ZeroDivisionError
        """
        
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "log_content": sample_log,
                "include_explanation": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction test successful:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Prediction test failed: {response.status_code}")
    
    except requests.RequestException as e:
        print(f"API test failed: {e}")
    
    finally:
        # Stop the API server
        api_process.terminate()
        api_process.wait()

def main():
    print("🚀 Starting CI/CD Anomaly Detection System Test")
    print("=" * 50)
    
    # Test 1: Check if model files exist
    print("1. Checking for model files...")
    try:
        import joblib
        model = joblib.load('anomaly_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("✅ Model files found and loaded")
    except FileNotFoundError:
        print("❌ Model files not found. Run training first:")
        print("   python preprocess_logs.py")
        return
    
    # Test 2: Test the model directly
    print("\n2. Testing model directly...")
    from preprocess_logs import AnomalyDetectionModel
    
    model = AnomalyDetectionModel()
    
    # Test with normal log
    normal_log = "INFO: All tests passed successfully"
    result = model.predict(normal_log)
    print(f"Normal log prediction: {result}")
    
    # Test with anomalous log
    anomalous_log = "ERROR: Test failed with ZeroDivisionError: division by zero"
    result = model.predict(anomalous_log)
    print(f"Anomalous log prediction: {result}")
    
    # Test 3: Test API
    print("\n3. Testing API service...")
    test_api_locally()
    
    print("\n🎉 System test completed!")
    print("\nNext steps:")
    print("1. Deploy your API to a cloud service")
    print("2. Set MODEL_API_URL secret in your GitHub repo")
    print("3. Push some code changes to trigger the workflows")

if __name__ == "__main__":
    main()