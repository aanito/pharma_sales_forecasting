import argparse
import time
from joblib import dump
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

logger.add("../logs/serialize_model.log")

def serialize_model(model, output_dir):
    try:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        model_filename = f"{output_dir}/sales_model_{timestamp}.pkl"
        logger.info(f"Saving model to {model_filename}")
        dump(model, model_filename)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Serialization Script")
    parser.add_argument('--output_dir', required=True, help="Path to output directory for serialized model")
    args = parser.parse_args()

    # Assuming a pre-trained pipeline for demonstration
    model = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    serialize_model(model, args.output_dir)