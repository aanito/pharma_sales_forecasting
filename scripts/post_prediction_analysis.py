import pandas as pd
import numpy as np
import argparse
from joblib import load
from sklearn.utils import resample
from loguru import logger

logger.add("../logs/post_prediction_analysis.log")

def analyze_feature_importance(model_path, input_file, output_dir):
    try:
        logger.info(f"Loading model from {model_path}")
        model = load(model_path)
        logger.info("Loading data")
        df = pd.read_csv(input_file)
        
        # Feature importance
        importance = model.named_steps['model'].feature_importances_
        feature_names = df.drop(['Sales'], axis=1).columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        logger.info("Saving feature importance")
        feature_importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        
        # Confidence interval estimation
        predictions = model.predict(df.drop(['Sales'], axis=1))
        intervals = bootstrap_confidence_intervals(df.drop(['Sales'], axis=1), model)
        logger.info("Saving confidence intervals")
        pd.DataFrame({
            'Prediction': predictions,
            'Lower Bound': intervals[:, 0],
            'Upper Bound': intervals[:, 1]
        }).to_csv(f"{output_dir}/confidence_intervals.csv", index=False)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def bootstrap_confidence_intervals(X, model, n_bootstrap=1000):
    predictions = []
    for _ in range(n_bootstrap):
        sample = resample(X)
        pred = model.predict(sample)
        predictions.append(pred)
    
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    return np.vstack((lower_bound, upper_bound)).T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post Prediction Analysis")
    parser.add_argument('--model', required=True, help="Path to the trained model")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Path to output directory")
    args = parser.parse_args()

    analyze_feature_importance(args.model, args.input, args.output_dir)