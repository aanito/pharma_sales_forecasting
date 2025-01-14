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

if __name__ == "__main__":
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