from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
from loguru import logger

logger.add("../logs/model_serving_api.log")

# Initialize the app
app = FastAPI()

# Load the model
model_path = "models/sales_model.pkl"
try:
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Define request and response structure
class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: float
    timestamp: str

# API Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        logger.error("Model is not available")
        raise HTTPException(status_code=500, detail="Model is not available")

    try:
        logger.info("Received prediction request")
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        response = PredictionResponse(
            prediction=prediction,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info(f"Prediction response: {response}")
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)