from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="CI/CD Log Anomaly Detection API")

# Mock model setup
def setup_model():
    texts = ["error fail exception", "success passed", "warning slow"]
    labels = [1, 0, 1]
    
    vectorizer = TfidfVectorizer(max_features=50)
    X = vectorizer.fit_transform(texts)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, labels)
    
    return model, vectorizer

model, vectorizer = setup_model()

@app.post("/predict")
async def predict(log_content: str):
    try:
        # Simple prediction
        text_clean = re.sub(r'\s+', ' ', log_content.lower())
        X = vectorizer.transform([text_clean])
        
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        return {
            "is_anomaly": bool(prediction),
            "confidence": float(max(proba)),
            "anomaly_probability": float(proba[1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)