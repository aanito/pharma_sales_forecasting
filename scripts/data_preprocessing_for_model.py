import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import argparse
from loguru import logger

logger.add("../logs/data_preprocessing.log")

def preprocess_data(input_file, output_file):
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        # Handling datetime columns
        df['Date'] = pd.to_datetime(df['Date'])
        df['Weekday'] = df['Date'].dt.weekday
        df['Weekend'] = df['Date'].dt.weekday >= 5
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['IsMonthStart'] = df['Date'].dt.is_month_start
        df['IsMonthEnd'] = df['Date'].dt.is_month_end

        # Handling missing values
        df.fillna(0, inplace=True)

        # Scaling numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        logger.info(f"Saving preprocessed data to {output_file}")
        df.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output', required=True, help="Path to output CSV file")
    args = parser.parse_args()

    preprocess_data(args.input, args.output)