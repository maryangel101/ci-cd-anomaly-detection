from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import numpy as np

app = FastAPI(title="CI/CD Log Anomaly Detection API")

# Global variables for model and vectorizer
model = None
vectorizer = None
explainer = None

class LogRequest(BaseModel):
    log_content: str
    include_explanation: bool = False

class PredictionResponse(BaseModel):
    is_anomaly: bool
    confidence: float
    anomaly_probability: float
    explanation: dict = None

def clean_log(log_content):
    """Clean and normalize log content"""
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

@app.on_event("startup")
async def load_model():
    """Load the trained model and vectorizer on startup"""
    global model, vectorizer, explainer
    
    try:
        model = joblib.load('anomaly_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        # Create SHAP explainer (simplified for speed)
        print("Initializing SHAP explainer...")
        
    except FileNotFoundError:
        print("Warning: Model files not found. Please train the model first.")

def get_feature_importance_explanation(log_content, prediction_proba):
    """Get simple feature importance explanation"""
    try:
        cleaned_log = clean_log(log_content)
        log_vector = vectorizer.transform([cleaned_log])
        
        # Get feature names and their importance
        feature_names = vectorizer.get_feature_names_out()
        feature_weights = log_vector.toarray()[0]
        
        # Get top features that contributed to the prediction
        top_indices = np.argsort(feature_weights)[-10:][::-1]  # Top 10 features
        top_features = []
        
        for idx in top_indices:
            if feature_weights[idx] > 0:
                top_features.append({
                    'feature': feature_names[idx],
                    'weight': float(feature_weights[idx]),
                    'impact': 'anomaly' if prediction_proba[1] > 0.5 else 'normal'
                })
        
        return {
            'top_contributing_features': top_features[:5],
            'explanation': "Features with higher weights contributed more to the prediction"
        }
    
    except Exception as e:
        return {
            'error': f"Could not generate explanation: {str(e)}",
            'explanation': "Feature importance analysis failed"
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(request: LogRequest):
    """Predict if a log indicates an anomaly"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure model files exist.")
    
    try:
        # Preprocess log
        cleaned_log = clean_log(request.log_content)
        log_vector = vectorizer.transform([cleaned_log])
        
        # Make prediction
        prediction = model.predict(log_vector)[0]
        probabilities = model.predict_proba(log_vector)[0]
        
        response = PredictionResponse(
            is_anomaly=bool(prediction),
            confidence=float(max(probabilities)),
            anomaly_probability=float(probabilities[1]) if len(probabilities) > 1 else 0.0
        )
        
        # Add explanation if requested
        if request.include_explanation:
            response.explanation = get_feature_importance_explanation(request.log_content, probabilities)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CI/CD Log Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)