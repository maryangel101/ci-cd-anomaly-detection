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
    """Send logs to the model API for analysis"""
    try:
        # Validate and construct the full URL
        if not model_url:
            print("❌ MODEL_API_URL is empty")
            return None
            
        # Ensure the URL has a scheme
        if not model_url.startswith(('http://', 'https://')):
            print(f"❌ MODEL_API_URL is missing scheme: {model_url}")
            print("💡 Please set MODEL_API_URL to a valid URL starting with http:// or https://")
            return None
        
        # Construct the full prediction endpoint URL
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
            return None
    
    except requests.exceptions.Timeout:
        print("❌ Model API request timed out (30 seconds)")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Model API - check the URL")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to call model API: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    # Get environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    workflow_run_id = os.environ.get('WORKFLOW_RUN_ID')
    model_api_url = os.environ.get('MODEL_API_URL')
    
    print("🔍 Environment variables check:")
    print(f"   GITHUB_TOKEN: {'✅' if github_token else '❌'}")
    print(f"   WORKFLOW_RUN_ID: {'✅' if workflow_run_id else '❌'} -> {workflow_run_id}")
    print(f"   MODEL_API_URL: {'✅' if model_api_url else '❌'} -> {model_api_url}")
    
    if not all([github_token, workflow_run_id, model_api_url]):
        print("❌ Missing required environment variables")
        sys.exit(1)
    
    print(f"📊 Analyzing workflow run: {workflow_run_id}")
    
    # Download logs
    print("⬇️ Downloading workflow logs...")
    logs = download_workflow_logs(workflow_run_id, github_token)
    if not logs:
        print("❌ Failed to download logs")
        sys.exit(1)
    
    print(f"📄 Logs downloaded ({len(logs)} characters)")
    
    # Analyze with model
    print("🤖 Sending logs to model API...")
    result = analyze_logs_with_model(logs, model_api_url)
    if not result:
        print("❌ Failed to analyze logs with model")
        sys.exit(1)
    
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