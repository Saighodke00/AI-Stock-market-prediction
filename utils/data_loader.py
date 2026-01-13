import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fetch_data(ticker, period="2y"):
    """
    Fetches historical stock data from yfinance.
    """
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            return None
        # Handle MultiIndex columns (yfinance update quirk)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def clean_data(df):
    """
    Removes missing values and ensures correct data types.
    """
    df = df.dropna()
    return df

def normalize_data(df, feature_column='Close'):
    """
    Normalizes the data using MinMaxScaler.
    Returns the scaler and the scaled data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[feature_column]])
    return scaler, scaled_data

def create_sequences(data, time_step=60):
    """
    Creates sequences for LSTM model.
    """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
