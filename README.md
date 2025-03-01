Rossmann Pharmaceuticals Sales Forecasting Project
Report

1. Setting Up the Python Environment
   Step 1.1: Create the Project Directory Structure
   RossmannSalesForecasting/
   |-- data/
   | |-- raw/
   | | |-- store.csv
   | | |-- test.csv
   | | |-- sample_submission.csv
   | | |-- train.csv
   | |-- processed/
   |-- notebooks/
   | |-- exploratory_data_analysis.ipynb
   |-- scripts/
   | |-- **init**.py
   | |-- data_preprocessing.py
   | |-- exploratory_analysis.py
   | |-- prediction_model.py
   | |-- logger.py
   |-- logs/
   | |-- eda.log
   | |-- prediction.log
   |-- requirements.txt
   |-- README.md

Step 1.2: Install Required Libraries
Create a requirements.txt file with the following content:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
keras
tensorflow
jupyterlab
loguru

Install the required libraries using the following command:
pip install -r requirements.txt

2. Data Preprocessing
   Step 2.1: Create the Data Preprocessing Script
   In scripts/data_preprocessing.py:
   import pandas as pd
   import argparse
   from loguru import logger

logger.add("../logs/data_preprocessing.log")

def preprocess_data(input_file, output_file):
try: # Load the data
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

        # Preprocessing steps
        # Example: Filling NaN values
        logger.info("Preprocessing data...")
        df.fillna(0, inplace=True)

        # Save processed data
        logger.info(f"Saving processed data to {output_file}")
        df.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Data Preprocessing Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output', required=True, help="Path to output CSV file")
args = parser.parse_args()

    preprocess_data(args.input, args.output)

3. Exploratory Data Analysis (EDA)
   Step 3.1: Create the Exploratory Analysis Script
   In scripts/exploratory_analysis.py:
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/exploratory_analysis.log")

def plot_distributions(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Plotting distributions...")
    sns.histplot(df['Sales'], kde=True)
    plt.title('Sales Distribution')
    plt.savefig(f"{output_dir}/sales_distribution.png")
    plt.close()

    sns.countplot(x='Promo', data=df)
    plt.title('Promo Distribution')
    plt.savefig(f"{output_dir}/promo_distribution.png")
    plt.close()

    logger.info("Distributions plotted and saved.")

def holiday_sales_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing holiday sales...")
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales During Holidays')
    plt.savefig(f"{output_dir}/holiday_sales_analysis.png")
    plt.close()

def correlation_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing correlation between sales and customers...")
    correlation = df[['Sales', 'Customers']].corr()
    sns.heatmap(correlation, annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

def seasonal_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week

    logger.info("Analyzing seasonal trends...")
    sns.lineplot(data=df, x='Week', y='Sales', hue='Year')
    plt.title('Seasonal Sales Trends')
    plt.savefig(f"{output_dir}/seasonal_sales_trends.png")
    plt.close()

def promo_effect_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing promo effect on sales...")
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Promo Effect on Sales')
    plt.savefig(f"{output_dir}/promo_effect.png")
    plt.close()

def store_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing store attributes...")
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales by Store Type')
    plt.savefig(f"{output_dir}/store_type_sales.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Exploratory Analysis Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    plot_distributions(args.input, args.output_dir)
    holiday_sales_analysis(args.input, args.output_dir)
    correlation_analysis(args.input, args.output_dir)
    seasonal_analysis(args.input, args.output_dir)
    promo_effect_analysis(args.input, args.output_dir)
    store_analysis(args.input, args.output_dir)

Step 3.2: Create the EDA Notebook
In notebooks/exploratory_data_analysis.ipynb:
import pandas as pd
from scripts.data_preprocessing import preprocess_data
from scripts.exploratory_analysis import (plot_distributions, holiday_sales_analysis,
correlation_analysis, seasonal_analysis,
promo_effect_analysis, store_analysis)

# Define input and output paths

input_file = '../data/raw/train.csv'
processed_file = '../data/processed/train_processed.csv'
output_dir = '../data/processed'

# Preprocess data

preprocess_data(input_file, processed_file)

# Perform EDA

plot_distributions(processed_file, output_dir)
holiday_sales_analysis(processed_file, output_dir)
correlation_analysis(processed_file, output_dir)
seasonal_analysis(processed_file, output_dir)
promo_effect_analysis(processed_file, output_dir)
store_analysis(processed_file, output_dir)

4. Logging
   Step 4.1: Create a Logger Script
   In scripts/logger.py:
   from loguru import logger

def setup_logger():
logger.add("../logs/general.log", rotation="500 MB")
logger.info("Logger is set up.")

5. Prediction of Store Sales
   Step 5.1: Create the Prediction Model Script
   In scripts/prediction_model.py:
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

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Prediction Model Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output', required=True, help="Path to output model file")
args = parser.parse_args()

    train_model(args.input, args.output)

Additional Analysis
Step-by-Step Guidance for Exploratory Analysis

1. Checking Distribution in Training and Test Sets
   Script: scripts/distribution_check.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/distribution_check.log")

def check_distribution(train_file, test_file, output_dir):
logger.info(f"Loading data from {train_file} and {test_file}")
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

    logger.info("Checking promotion distribution in training and test sets...")
    sns.histplot(train_df['Promo'], label='Train', color='blue', kde=False)
    sns.histplot(test_df['Promo'], label='Test', color='orange', kde=False)
    plt.legend()
    plt.title('Promotion Distribution in Train vs Test')
    plt.savefig(f"{output_dir}/promo_distribution_comparison.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Check Distribution Script")
parser.add_argument('--train', required=True, help="Path to training CSV file")
parser.add_argument('--test', required=True, help="Path to test CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    check_distribution(args.train, args.test, args.output_dir)

2. Sales Behavior Around Holidays
   Script: scripts/holiday_sales_behavior.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/holiday_sales_behavior.log")

def holiday_sales_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing sales behavior around holidays...")
    sns.lineplot(data=df, x='Date', y='Sales', hue='StateHoliday')
    plt.title('Sales Behavior Around Holidays')
    plt.savefig(f"{output_dir}/holiday_sales_behavior.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Holiday Sales Behavior Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    holiday_sales_analysis(args.input, args.output_dir)

3. Seasonal Purchase Behavior
   Script: scripts/seasonal_behavior.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/seasonal_behavior.log")

def seasonal_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

    logger.info("Analyzing seasonal purchase behavior...")
    sns.lineplot(data=df, x='Month', y='Sales', hue='Year')
    plt.title('Seasonal Purchase Behavior')
    plt.savefig(f"{output_dir}/seasonal_behavior.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Seasonal Behavior Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    seasonal_analysis(args.input, args.output_dir)

4. Correlation Between Sales and Number of Customers
   Script: scripts/sales_customer_correlation.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/sales_customer_correlation.log")

def correlation_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing correlation between sales and number of customers...")
    correlation = df[['Sales', 'Customers']].corr()
    sns.heatmap(correlation, annot=True)
    plt.title('Correlation Between Sales and Customers')
    plt.savefig(f"{output_dir}/sales_customer_correlation.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Sales-Customer Correlation Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    correlation_analysis(args.input, args.output_dir)

5. Effect of Promotions on Sales
   Script: scripts/promo_effect_on_sales.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/promo_effect_on_sales.log")

def promo_effect_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing promo effect on sales...")
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Effect of Promotions on Sales')
    plt.savefig(f"{output_dir}/promo_effect_on_sales.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Promo Effect on Sales Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    promo_effect_analysis(args.input, args.output_dir)

6. Store Behavior Analysis
   Script: scripts/store_behavior_analysis.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/store_behavior_analysis.log")

def store_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing store attributes...")
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales by Store Type')
    plt.savefig(f"{output_dir}/store_type_sales.png")
    plt.close()

    sns.boxplot(x='Assortment', y='Sales', data=df)
    plt.title('Sales by Assortment Type')
    plt.savefig(f"{output_dir}/assortment_type_sales.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Store Behavior Analysis Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    store_analysis(args.input, args.output_dir)

7. Competitor Distance Effect on Sales
   Script: scripts/competitor_distance_effect.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   import argparse
   from loguru import logger

logger.add("../logs/competitor_distance_effect.log")

def competitor_distance_analysis(input_file, output_dir):
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

    logger.info("Analyzing effect of competitor distance on sales...")
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title('Effect of Competitor Distance on Sales')
    plt.savefig(f"{output_dir}/competitor_distance_effect.png")
    plt.close()

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Competitor Distance Effect Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
args = parser.parse_args()

    competitor_distance_analysis(args.input, args.output_dir)

Task 2 - Prediction of Store Sales: Script Development
Script: Data Preprocessing (Task 2.1)
Script Name: data_preprocessing.py
Purpose: Preprocess the data by converting non-numeric columns to numeric, handling missing values, generating new features, and scaling the data.
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

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Data Preprocessing Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output', required=True, help="Path to output CSV file")
args = parser.parse_args()

    preprocess_data(args.input, args.output)

Command to Run:
python scripts/data_preprocessing.py --input data/raw/train.csv --output data/processed/train_preprocessed.csv

Script: Model Training with Sklearn Pipelines (Task 2.2)
Script Name: model_training.py
Purpose: Build a Random Forest Regressor model using sklearn pipelines and save the trained model.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump
import argparse
from loguru import logger

logger.add("../logs/model_training.log")

def train_model(input_file, output_model):
try:
logger.info(f"Loading data from {input_file}")
df = pd.read_csv(input_file)

        X = df.drop(['Sales'], axis=1)
        y = df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        logger.info("Training model")
        pipeline.fit(X_train, y_train)

        logger.info(f"Saving model to {output_model}")
        dump(pipeline, output_model)
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_model', required=True, help="Path to output model file")
args = parser.parse_args()

    train_model(args.input, args.output_model)

Command to Run:
python scripts/model_training.py --input data/processed/train_preprocessed.csv --output_model models/sales_model.pkl

Script: Loss Function Selection (Task 2.3)
Script Name: loss_function_selection.py
Purpose: Select and justify the use of the Mean Squared Error (MSE) loss function for the regression task.
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
from loguru import logger

logger.add("../logs/loss_function_selection.log")

def calculate_mse(y_true, y_pred):
try:
mse = mean_squared_error(y_true, y_pred)
logger.info(f"Calculated MSE: {mse}")
return mse
except Exception as e:
logger.error(f"An error occurred: {e}")

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Loss Function Selection Script")
parser.add_argument('--true', required=True, nargs='+', type=float, help="True values")
parser.add_argument('--pred', required=True, nargs='+', type=float, help="Predicted values")
args = parser.parse_args()

    calculate_mse(np.array(args.true), np.array(args.pred))

Command to Run:
python scripts/loss_function_selection.py --true 200 250 300 --pred 210 245 290

2.4 Post Prediction Analysis
Objective: Explore the feature importance from our modeling and estimate the confidence interval of predictions.
Steps:
Feature Importance:
Utilize the feature importance attribute from the Random Forest model to determine which features contributed most to the predictions.
Confidence Interval Estimation:
Use bootstrapping to estimate the confidence intervals for predictions.
Generate multiple samples from the training data, train models on each sample, and calculate the prediction intervals.
Script: post_prediction_analysis.py
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

def bootstrap*confidence_intervals(X, model, n_bootstrap=1000):
predictions = []
for * in range(n_bootstrap):
sample = resample(X)
pred = model.predict(sample)
predictions.append(pred)

    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    return np.vstack((lower_bound, upper_bound)).T

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Post Prediction Analysis")
parser.add_argument('--model', required=True, help="Path to the trained model")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_dir', required=True, help="Path to output directory")
args = parser.parse_args()

    analyze_feature_importance(args.model, args.input, args.output_dir)

Command to Run:
python scripts/post_prediction_analysis.py --model models/sales_prediction_model.pkl --input data/processed/train_cleaned.csv --output_dir reports/post_prediction_analysis

2.5 Serialize Models
Objective: Serialize the models with a timestamp to enable tracking of predictions from various models.
Steps:
Model Serialization:
Save the trained model with a timestamp in the filename.
Script: serialize_model.py
import argparse
import time
from joblib import dump
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

logger.add("../logs/serialize_model.log")

def serialize*model(model, output_dir):
try:
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
model_filename = f"{output_dir}/sales_model*{timestamp}.pkl"
logger.info(f"Saving model to {model_filename}")
dump(model, model_filename)
except Exception as e:
logger.error(f"An error occurred: {e}")

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Model Serialization Script")
parser.add_argument('--output_dir', required=True, help="Path to output directory for serialized model")
args = parser.parse_args()

    # Assuming a pre-trained pipeline for demonstration
    model = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    serialize_model(model, args.output_dir)

Command to Run:
python scripts/serialize_model.py --output_dir models

2.6 Building Model with Deep Learning
Objective: Build a deep learning model using Long Short Term Memory (LSTM) to predict future sales.
Steps:
Data Preparation:

Isolate the dataset into time series data.
Check for stationarity and perform differencing if necessary.
Analyze autocorrelation and partial autocorrelation.
Transform the data into supervised learning format using a sliding window approach.
Scale the data in the range (-1, 1).
LSTM Model Construction:

Build a two-layer LSTM model using TensorFlow or PyTorch.
Train the model on the prepared time series data.
Script: lstm_sales_prediction.py
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

logger.add("../logs/lstm_sales_prediction.log")

def prepare_data(input_file):
df = pd.read_csv(input_file)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
sales_data = df[['Sales']].astype(float)
return sales_data

def create_supervised_data(data, look_back=1):
X, y = [], []
for i in range(len(data) - look_back):
X.append(data[i:(i + look_back), 0])
y.append(data[i + look_back, 0])
return np.array(X), np.array(y)

def build_lstm_model(input_shape):
model = tf.keras.Sequential([
tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
return model

if **name** == "**main**":
parser = argparse.ArgumentParser(description="LSTM Sales Prediction")
parser.add_argument('--input', required=True, help="Path to input CSV file")
parser.add_argument('--output_model', required=True, help="Path to output model file")
args = parser.parse_args()

    sales_data = prepare_data(args.input)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sales_scaled = scaler.fit_transform(sales_data)

    look_back = 10
    X, y = create_supervised_data(sales_scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_lstm_model((look_back, 1))
    logger.info("Training LSTM model")
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    logger.info(f"Saving model to {args.output_model}")
    model.save(args.output_model)

Command to Run:
python scripts/lstm_sales_prediction.py --input data/processed/train_cleaned.csv --output_model models/lstm_sales_model.h5
