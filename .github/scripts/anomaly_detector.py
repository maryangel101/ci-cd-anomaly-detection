import os
import requests
import sys
import json

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
        return None

def analyze_logs_with_model(log_content, model_url):
    """Send logs to the model API for analysis"""
    try:
        response = requests.post(
            f"{model_url}/predict",
            json={
                "log_content": log_content,
                "include_explanation": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Model API error: {response.status_code}")
            return None
    
    except requests.RequestException as e:
        print(f"Failed to call model API: {e}")
        return None

def main():
    # Get environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    workflow_run_id = os.environ.get('WORKFLOW_RUN_ID')
    model_api_url = os.environ.get('MODEL_API_URL')
    
    if not all([github_token, workflow_run_id, model_api_url]):
        print("Missing required environment variables")
        sys.exit(1)
    
    print(f"Analyzing workflow run: {workflow_run_id}")
    
    # Download logs
    logs = download_workflow_logs(workflow_run_id, github_token)
    if not logs:
        print("Failed to download logs")
        sys.exit(1)
    
    # Analyze with model
    result = analyze_logs_with_model(logs, model_api_url)
    if not result:
        print("Failed to analyze logs with model")
        sys.exit(1)
    
    print(f"Analysis result: {json.dumps(result, indent=2)}")
    
    # Check if anomaly detected
    if result.get('is_anomaly', False):
        print("🚨 ANOMALY DETECTED!")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Anomaly Probability: {result.get('anomaly_probability', 0):.2%}")
        
        if result.get('explanation'):
            print("Explanation:", result['explanation'])
        
        # Exit with error code to mark step as failed
        sys.exit(1)
    else:
        print("✅ No anomalies detected")
        sys.exit(0)

if __name__ == "__main__":
    main()