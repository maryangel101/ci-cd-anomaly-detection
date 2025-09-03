import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

class LogPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    
    def clean_log(self, log_content):
        # Remove timestamps
        log_content = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', '[TIMESTAMP]', log_content)
        
        # Remove specific IDs and numbers
        log_content = re.sub(r'run_\d+', 'run_ID', log_content)
        log_content = re.sub(r'job_\d+', 'job_ID', log_content)
        log_content = re.sub(r'\b\d{8,}\b', '[ID]', log_content)
        
        # Remove paths that might be environment-specific
        log_content = re.sub(r'/home/[^/\s]+', '/home/USER', log_content)
        log_content = re.sub(r'/tmp/[^/\s]+', '/tmp/TEMP', log_content)
        
        # Normalize whitespace
        log_content = re.sub(r'\s+', ' ', log_content)
        
        return log_content.strip()
    
    def load_logs(self, data_dir):
        logs = []
        labels = []
        
        # Load normal logs
        normal_dir = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_dir):
            for filename in os.listdir(normal_dir):
                filepath = os.path.join(normal_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    logs.append(self.clean_log(content))
                    labels.append(0)  # 0 for normal
        
        # Load anomalous logs
        anomalous_dir = os.path.join(data_dir, 'anomalous')
        if os.path.exists(anomalous_dir):
            for filename in os.listdir(anomalous_dir):
                filepath = os.path.join(anomalous_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    logs.append(self.clean_log(content))
                    labels.append(1)  # 1 for anomalous
        
        return logs, labels

class AnomalyDetectionModel:
    def __init__(self):
        self.preprocessor = LogPreprocessor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False
    
    def train(self, data_dir='data'):
        print("Loading logs...")
        logs, labels = self.preprocessor.load_logs(data_dir)
        
        if len(logs) == 0:
            raise ValueError("No logs found. Please run data collection first.")
        
        print(f"Loaded {len(logs)} logs ({labels.count(0)} normal, {labels.count(1)} anomalous)")
        
        # Vectorize logs
        print("Vectorizing logs...")
        X = self.preprocessor.vectorizer.fit_transform(logs)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomalous']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        
        # Save model and vectorizer
        joblib.dump(self.model, 'anomaly_model.pkl')
        joblib.dump(self.preprocessor.vectorizer, 'vectorizer.pkl')
        print("\nModel saved as 'anomaly_model.pkl' and 'vectorizer.pkl'")
    
    def predict(self, log_content):
        if not self.is_trained:
            # Load saved model
            try:
                self.model = joblib.load('anomaly_model.pkl')
                self.preprocessor.vectorizer = joblib.load('vectorizer.pkl')
                self.is_trained = True
            except FileNotFoundError:
                raise ValueError("No trained model found. Please train first.")
        
        # Preprocess and predict
        cleaned_log = self.preprocessor.clean_log(log_content)
        log_vector = self.preprocessor.vectorizer.transform([cleaned_log])
        
        prediction = self.model.predict(log_vector)[0]
        probability = self.model.predict_proba(log_vector)[0]
        
        return {
            'is_anomaly': bool(prediction),
            'confidence': float(max(probability)),
            'anomaly_probability': float(probability[1]) if len(probability) > 1 else 0.0
        }

if __name__ == "__main__":
    model = AnomalyDetectionModel()
    model.train()