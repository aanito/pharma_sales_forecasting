import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
from loguru import logger

logger.add("../logs/prediction_model.log")

def train_model(input_file, output_file):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Example features and target split
    X = df.drop(columns=['Sales', 'Date'])
    y = df['Sales']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training model...")
    pipeline = Pipeline([('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    logger.info(f"Validation MSE: {mse}")

    # Save model
    logger.info(f"Saving model to {output_file}")
    pd.to_pickle(pipeline, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Model Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output', required=True, help="Path to output model file")
    args = parser.parse_args()

    train_model(args.input, args.output)

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from loguru import logger

# logger.add("../logs/prediction.log")

# @logger.catch
# def build_pipeline():
#     logger.info("Building model pipeline...")
#     pipeline = Pipeline([
#         ('model', RandomForestRegressor(n_estimators=100, random_state=42))
#     ])
#     return pipeline

# @logger.catch
# def train_model(train):
#     logger.info("Training model...")
#     X = train.drop(columns=['Sales', 'Date'])
#     y = train['Sales']
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#     pipeline = build_pipeline()
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_valid)
#     mse = mean_squared_error(y_valid, y_pred)
#     logger.info(f"Validation MSE: {mse}")
#     return pipeline