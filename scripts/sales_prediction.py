import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from loguru import logger

logger.add("../logs/sales_prediction.log")

def train_model(input_file, output_model):
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        X = df.drop(['Sales'], axis=1)
        y = df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        logger.info("Training model")
        pipeline.fit(X_train, y_train)

        logger.info(f"Saving model to {output_model}")
        dump(pipeline, output_model)
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Prediction Model Training Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_model', required=True, help="Path to output model file")
    args = parser.parse_args()

    train_model(args.input, args.output_model)