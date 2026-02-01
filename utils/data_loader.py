import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

def fetch_data(ticker, period="2y", interval="1d"):
    """
    Fetches historical stock data and its benchmark macro.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty: return None
        
        # Determine Macro Index
        macro_ticker = "^GSPC" 
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            macro_ticker = "^NSEI" 
        
        macro_data = yf.download(macro_ticker, period=period, interval=interval, progress=False)
        
        # Clean multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if isinstance(macro_data.columns, pd.MultiIndex):
            macro_data.columns = macro_data.columns.get_level_values(0)

        # Align macro
        if 'Close' in macro_data.columns:
            macro_close = macro_data['Close']
            if isinstance(macro_close, pd.DataFrame):
                macro_close = macro_close.iloc[:, 0]
            macro_aligned = macro_close.reindex(data.index, method='ffill')
            data['Macro_Close'] = macro_aligned
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def clean_data(df):
    return df.ffill().dropna()

def normalize_data(df, feature_columns, split_ratio=0.8):
    """
    Leakage-free normalization: Fit on training set only.
    """
    data = df[feature_columns].values
    train_size = int(len(data) * split_ratio)
    
    scaler = RobustScaler()
    scaler.fit(data[:train_size])
    
    scaled_data = scaler.transform(data)
    return scaler, scaled_data

def create_sequences(df, feature_columns, lookback=60):
    """
    Creates sequences for Causal Engine.
    Returns: X (scaled), y_dir, y_mag, scaler, scaled_data
    """
    # 1. Generate Targets (Shifted log-returns)
    df = df.copy()
    df['target_log_ret'] = df['log_ret'].shift(-1)
    df['target_dir'] = np.where(df['target_log_ret'] > 0, 1, 0)
    
    # 2. Normalize (Leakage-free Robust Scaler)
    scaler, scaled_data = normalize_data(df, feature_columns)
    
    # 3. Prepare Tensors (Drop rows with NaN targets - usually just the last row)
    df_train = df.dropna(subset=['target_log_ret'])
    scaled_train = scaled_data[:len(df_train)]
    
    X, Y_dir, Y_mag = [], [], []
    y_dir = df_train['target_dir'].values
    y_mag = df_train['target_log_ret'].values
    
    for i in range(lookback, len(scaled_train)):
        X.append(scaled_train[i-lookback:i])
        Y_dir.append(y_dir[i])
        Y_mag.append(y_mag[i])
        
    return np.array(X), np.array(Y_dir), np.array(Y_mag), scaler, scaled_data

def add_noise(data, noise_level=0.001):
    """Causal noise injection."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise
