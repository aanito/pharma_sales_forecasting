import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import argparse
from loguru import logger
from logger import setup_logger

# Set up the logger
setup_logger()

# logger.add("../logs/data_preprocessing.log")

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        X = X.copy()
        for col in self.columns:
            if np.issubdtype(X[col].dtype, np.number):
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X.loc[(X[col] < lower_bound) | (X[col] > upper_bound), col] = np.nan
        return X

def preprocess_data(input_file, output_file):
    try:
        # Load the data with low_memory=False to handle mixed types
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, low_memory=False)

        # Identify columns to handle outliers and missing data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Build preprocessing pipeline
        logger.info("Building preprocessing pipeline...")
        pipeline = Pipeline([
            ('outlier_handler', OutlierHandler(columns=numeric_cols)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        logger.info("Fitting and transforming data...")
        df[numeric_cols] = pipeline.fit_transform(df[numeric_cols])

        # Save processed data
        logger.info(f"Saving processed data to {output_file}")
        df.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output', required=True, help="Path to output CSV file")
    args = parser.parse_args()

    preprocess_data(args.input, args.output)


# import pandas as pd
# import numpy as np
# import argparse
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.base import BaseEstimator, TransformerMixin
# from loguru import logger

# logger.add("../logs/data_preprocessing.log")

# class OutlierHandler(BaseEstimator, TransformerMixin):
#     def __init__(self, method='IQR', factor=1.5):
#         self.method = method
#         self.factor = factor

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         if self.method == 'IQR':
#             Q1 = X.quantile(0.25)
#             Q3 = X.quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - (self.factor * IQR)
#             upper_bound = Q3 + (self.factor * IQR)
#             return X.apply(lambda x: np.where((x < lower_bound) | (x > upper_bound), np.nan, x), axis=0)
#         else:
#             return X

# def preprocess_data(input_file, output_file):
#     try:
#         # Load the data
#         logger.info(f"Loading data from {input_file}")
#         df = pd.read_csv(input_file)

#         # Define the preprocessing pipeline
#         pipeline = Pipeline([
#             ('imputer', SimpleImputer(strategy='median')),
#             ('outlier_handler', OutlierHandler()),
#             ('scaler', StandardScaler())
#         ])

#         # Select numeric columns for the pipeline
#         numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#         df[numeric_cols] = pipeline.fit_transform(df[numeric_cols])

#         # Save processed data
#         logger.info(f"Saving processed data to {output_file}")
#         df.to_csv(output_file, index=False)
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Data Preprocessing Script")
#     parser.add_argument('--input', required=True, help="Path to input CSV file")
#     parser.add_argument('--output', required=True, help="Path to output CSV file")
#     args = parser.parse_args()

#     preprocess_data(args.input, args.output)
