import os
import requests
import json
import zipfile
import io
from datetime import datetime

class GitHubLogCollector:
    def __init__(self, repo_owner, repo_name, token):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
    def get_workflow_runs(self, workflow_id='main.yml'):
        url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow_id}/runs'
        response = requests.get(url, headers=self.headers)
        return response.json()
    
    def download_log(self, run_id):
        url = f'https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}/logs'
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            # Extract zip content
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                log_content = ""
                for file_name in zip_file.namelist():
                    with zip_file.open(file_name) as file:
                        log_content += file.read().decode('utf-8') + "\n"
                return log_content
        return None
    
    def collect_all_logs(self):
        os.makedirs('data/normal', exist_ok=True)
        os.makedirs('data/anomalous', exist_ok=True)
        
        runs = self.get_workflow_runs()
        
        for run in runs['workflow_runs']:
            run_id = run['id']
            conclusion = run['conclusion']
            created_at = run['created_at']
            
            log_content = self.download_log(run_id)
            if log_content:
                # Clean filename
                timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y%m%d_%H%M%S')
                
                if conclusion == 'success':
                    filename = f'data/normal/run_{run_id}_{timestamp}.log'
                else:
                    filename = f'data/anomalous/run_{run_id}_{timestamp}.log'
                
                with open(filename, 'w') as f:
                    f.write(log_content)
                
                print(f"Saved: {filename} (Status: {conclusion})")

# Usage
if __name__ == "__main__":
    # Replace with your values
    REPO_OWNER = "your-username"
    REPO_NAME = "your-repo-name"
    GITHUB_TOKEN = "your-github-token"  # Create at https://github.com/settings/tokens
    
    collector = GitHubLogCollector(REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
    collector.collect_all_logs()