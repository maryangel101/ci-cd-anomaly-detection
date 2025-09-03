import os
import requests
import sys
import json
from urllib.parse import urljoin

def download_workflow_logs(run_id, token):
    """Download logs from a specific workflow run"""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Get repository info from environment
    repo_full_name = os.environ.get('GITHUB_REPOSITORY')
    
    # Download logs
    url = f'https://api.github.com/repos/{repo_full_name}/actions/runs/{run_id}/logs'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download logs: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def analyze_logs_with_model(log_content, model_url):
    """Send logs to the model API for analysis with fallback to mock response"""
    try:
        # If no model URL provided, use mock response
        if not model_url or model_url == "https://httpbin.org":
            print("🤖 Using mock response (no valid API URL provided)")
            return generate_mock_response(log_content)
            
        # Ensure the URL has a scheme
        if not model_url.startswith(('http://', 'https://')):
            print(f"❌ MODEL_API_URL is missing scheme: {model_url}")
            print("💡 Using mock response instead")
            return generate_mock_response(log_content)
        
        # Construct the full prediction endpoint URL
        if model_url.endswith('/predict'):
            prediction_url = model_url
        else:
            prediction_url = urljoin(model_url.rstrip('/') + '/', 'predict')
        
        print(f"🔗 Calling model API: {prediction_url}")
        
        response = requests.post(
            prediction_url,
            json={
                "log_content": log_content,
                "include_explanation": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Model API error: {response.status_code}")
            print(f"Response: {response.text}")
            print("💡 Falling back to mock response")
            return generate_mock_response(log_content)
    
    except requests.exceptions.Timeout:
        print("❌ Model API request timed out (30 seconds)")
        print("💡 Falling back to mock response")
        return generate_mock_response(log_content)
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Model API - check the URL")
        print("💡 Falling back to mock response")
        return generate_mock_response(log_content)
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to call model API: {e}")
        print("💡 Falling back to mock response")
        return generate_mock_response(log_content)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Falling back to mock response")
        return generate_mock_response(log_content)

def generate_mock_response(log_content):
    """Generate a mock response for testing when API is unavailable"""
    # Simple anomaly detection based on log content
    error_keywords = ['error', 'fail', 'failed', 'exception', 'traceback', 'timeout']
    warning_keywords = ['warning', 'slow', 'retry', 'timeout']
    
    error_count = sum(1 for keyword in error_keywords if keyword in log_content.lower())
    warning_count = sum(1 for keyword in warning_keywords if keyword in log_content.lower())
    
    # Determine if anomaly based on error count
    is_anomaly = error_count > 2 or (error_count > 0 and warning_count > 3)
    
    # Calculate confidence based on keyword matches
    confidence = min(0.95, 0.7 + (error_count * 0.1) + (warning_count * 0.05))
    
    return {
        'is_anomaly': is_anomaly,
        'confidence': round(confidence, 2),
        'anomaly_probability': 0.8 if is_anomaly else 0.2,
        'explanation': {
            'detection_method': 'mock_analysis',
            'error_keywords_found': [kw for kw in error_keywords if kw in log_content.lower()],
            'warning_keywords_found': [kw for kw in warning_keywords if kw in log_content.lower()],
            'error_count': error_count,
            'warning_count': warning_count,
            'message': 'This analysis used mock detection. Set up a real model API for accurate results.'
        }
    }

def main():
    # Get environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    workflow_run_id = os.environ.get('WORKFLOW_RUN_ID')
    model_api_url = os.environ.get('MODEL_API_URL')
    
    print("🔍 Environment variables check:")
    print(f"   GITHUB_TOKEN: {'✅' if github_token else '❌'}")
    print(f"   WORKFLOW_RUN_ID: {'✅' if workflow_run_id else '❌'} -> {workflow_run_id}")
    print(f"   MODEL_API_URL: {'✅' if model_api_url else '❌'} -> {model_api_url}")
    
    # Only require GITHUB_TOKEN and WORKFLOW_RUN_ID, MODEL_API_URL is optional
    if not github_token or not workflow_run_id:
        print("❌ Missing required environment variables: GITHUB_TOKEN or WORKFLOW_RUN_ID")
        sys.exit(0)  # Exit gracefully, don't fail the workflow
    
    print(f"📊 Analyzing workflow run: {workflow_run_id}")
    
    # Download logs
    print("⬇️ Downloading workflow logs...")
    logs = download_workflow_logs(workflow_run_id, github_token)
    if not logs:
        print("❌ Failed to download logs")
        sys.exit(0)  # Exit gracefully, don't fail the workflow
    
    print(f"📄 Logs downloaded ({len(logs)} characters)")
    
    # Analyze with model (or mock)
    result = analyze_logs_with_model(logs, model_api_url)
    if not result:
        print("❌ Failed to analyze logs")
        sys.exit(0)  # Exit gracefully, don't fail the workflow
    
    print(f"📋 Analysis result: {json.dumps(result, indent=2)}")
    
    # Check if anomaly detected
    if result.get('is_anomaly', False):
        print("🚨 ANOMALY DETECTED!")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Anomaly Probability: {result.get('anomaly_probability', 0):.2%}")
        
        if result.get('explanation'):
            print("   Explanation:", json.dumps(result['explanation'], indent=2))
        
        # Exit with error code to mark step as failed
        sys.exit(1)
    else:
        print("✅ No anomalies detected")
        sys.exit(0)

if __name__ == "__main__":
    main()