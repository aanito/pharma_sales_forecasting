from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load
import argparse
from loguru import logger

logger.add("../logs/model_serving_api.log")

app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # Add more features as needed

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        logger.info("Loading model")
        model = load("../models/sales_prediction_model.pkl")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        logger.info("Received prediction request")
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        logger.info("Prediction successful")
        return {"prediction": prediction[0]}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")