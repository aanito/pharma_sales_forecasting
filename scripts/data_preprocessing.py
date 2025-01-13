import pandas as pd
import numpy as np
from loguru import logger

logger.add("../logs/data_preprocessing.log")

@logger.catch
def load_data():
    logger.info("Loading data...")
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    store = pd.read_csv("data/raw/store.csv")
    return train, test, store

@logger.catch
def clean_data(train, test, store):
    logger.info("Cleaning data...")
    # Handle missing values
    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)
    store.fillna(0, inplace=True)
    train['Open'].fillna(1, inplace=True)
    test['Open'].fillna(1, inplace=True)
    return train, test, store

@logger.catch
def preprocess_features(train, test):
    logger.info("Preprocessing features...")
    # Convert date columns to datetime
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    # Extract new features
    for df in [train, test]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    return train, test

@logger.catch
def scale_features(train, test):
    from sklearn.preprocessing import StandardScaler
    logger.info("Scaling features...")
    scaler = StandardScaler()
    numeric_features = ['Customers', 'CompetitionDistance', 'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfWeek']
    train[numeric_features] = scaler.fit_transform(train[numeric_features])
    test[numeric_features] = scaler.transform(test[numeric_features])
    return train, test
