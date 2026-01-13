import pandas as pd
# import pandas_ta as ta -- Removed to avoid dependency issues

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    """
    Adds RSI, SMA, and MACD technical indicators to the dataframe.
    """
    # SMA (Simple Moving Average)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index) - Manual Implementation for stability
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    # EMA 12
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    # EMA 26
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    df['MACD_12_26_9'] = macd
    df['MACDs_12_26_9'] = macd.ewm(span=9, adjust=False).mean() # Signal line
    
    return df
