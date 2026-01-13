import numpy as np
import tensorflow as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_model(input_shape):
    """
    Builds the LSTM model.
    """
    model = Sequential()
    # First LSTM layer with Return Sequences to feed into the next LSTM layer
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    # Second LSTM layer
    model.add(LSTM(50, return_sequences=False))
    # Output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X, y, epochs=20, batch_size=32):
    """
    Trains the model.
    """
    # Use a small number of epochs for quick testing if needed, but 20 is standard
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def predict_next_day(model, last_sequence, scaler):
    """
    Predicts the next day's price.
    last_sequence: numpy array of shape (1, time_step, 1)
    """
    prediction = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]
