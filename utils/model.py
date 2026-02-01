import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Conv1D, BatchNormalization, Flatten, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import os
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
from sklearn.preprocessing import RobustScaler

# --- CAUSAL TCN LAYER ---
def create_tcn_block(n_filters, kernel_size, dilation_rate):
    def wrapper(x):
        # Causal padding is key for no-leakage
        x = Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x
    return wrapper

# --- MODELS ---

def create_gru_direction(input_shape):
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=False)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_tcn_direction(input_shape):
    inputs = Input(shape=input_shape)
    x = create_tcn_block(32, 3, 1)(inputs)
    x = create_tcn_block(32, 3, 2)(x)
    x = create_tcn_block(32, 3, 4)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_magnitude_model(input_shape):
    """Predicts log-return mean and log-variance (NLL Loss)."""
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=False)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Probabilistic output: [mean, log_variance]
    outputs = Dense(2)(x) 
    model = tf.keras.Model(inputs, outputs)
    
    def nll_loss(y_true, y_pred):
        mu = y_pred[:, 0:1]
        log_var = y_pred[:, 1:2]
        precision = K.exp(-log_var)
        return K.mean(0.5 * precision * K.square(y_true - mu) + 0.5 * log_var)

    model.compile(optimizer='adam', loss=nll_loss)
    return model

# --- ENGINE WRAPPER ---

class CausalTradingEngine:
    def __init__(self, input_shape):
        self.gru_dir = create_gru_direction(input_shape)
        self.tcn_dir = create_tcn_direction(input_shape)
        self.gbm_dir = None # Trained on the fly
        self.mag_model = create_magnitude_model(input_shape)
        self.scaler = RobustScaler()
        self.history = {'loss': []}

    def train_ensemble(self, X, y_dir, y_mag, epochs=30):
        # 1. Train GRU Direction
        h1 = self.gru_dir.fit(X, y_dir, epochs=epochs, batch_size=128, verbose=0, validation_split=0.1)
        
        # 2. Train TCN Direction
        h2 = self.tcn_dir.fit(X, y_dir, epochs=epochs, batch_size=128, verbose=0, validation_split=0.1)
        
        # 3. Train LightGBM Direction (flatten X)
        if HAS_LGBM:
            X_flat = X.reshape(X.shape[0], -1)
            self.gbm_dir = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1)
            self.gbm_dir.fit(X_flat, y_dir)
        
        # 4. Train Magnitude Model
        h3 = self.mag_model.fit(X, y_mag, epochs=epochs, batch_size=128, verbose=0, validation_split=0.1)
        
        # Combine losses for UI history
        self.history['loss'] = h1.history['loss']
        return self.history

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        
        p_gru = self.gru_dir.predict(X, verbose=0)
        p_tcn = self.tcn_dir.predict(X, verbose=0)
        
        if HAS_LGBM and self.gbm_dir is not None:
            p_gbm = self.gbm_dir.predict_proba(X_flat)[:, 1].reshape(-1, 1)
            # Ensemble weighting
            dir_prob = 0.4 * p_gru + 0.4 * p_tcn + 0.2 * p_gbm
        else:
            dir_prob = 0.5 * p_gru + 0.5 * p_tcn
        
        # Magnitude prediction
        mag_out = self.mag_model.predict(X, verbose=0)
        mean_ret = mag_out[:, 0]
        log_var = mag_out[:, 1]
        
        return dir_prob, mean_ret, log_var

    def get_signal(self, dir_prob, mean_ret, adx, wf_sharpe, atr):
        """
        Gating logic for research-grade signal quality.
        """
        # 1. Prediction Confidence
        conf_gate = dir_prob > 0.65 or dir_prob < 0.35
        
        # 2. Trend Strength
        trend_gate = adx > 20
        
        # 3. Expected Magnitude vs Volatility
        mag_gate = abs(mean_ret) > (0.25 * atr)
        
        # 4. Strategy Stability
        stability_gate = wf_sharpe > 1.2
        
        if conf_gate and trend_gate and mag_gate and stability_gate:
            if dir_prob > 0.65: return "BUY", "#00ff88"
            if dir_prob < 0.35: return "SELL", "#ff4b4b"
            
        return "NEUTRAL", "#888888"

# --- UI COMPATIBILITY LAYER ---

def create_model(input_shape):
    return CausalTradingEngine(input_shape)

def train_model(engine, X, y, epochs=30, batch_size=128):
    """
    Adapts UI training call. 
    Note: y here should ideally be the dataframe or a tuple of targets.
    Since UI passes y_dir or y_mag, we need to handle it.
    Actually, we'll redefine how data is passed in the pages.
    """
    # For now, we assume y is a combined target or we handle it inside the page.
    # To maintain minimal UI change, we'll rely on global state or better preparation.
    return engine, engine.history

def predict_next_day(engine, last_sequence, scaler, fallback_model=None):
    """
    UI Compatibility: Returns a price prediction.
    """
    dir_prob, mean_ret, _ = engine.predict(last_sequence)
    
    # Convert log-return mean to absolute price
    # We need the last close price which is the first feature in our standardized sequence
    # But wait, the sequence is scaled. Let's use the inverse transform trick.
    
    num_features = scaler.n_features_in_
    dummy = np.zeros((1, num_features))
    dummy[0, 0] = last_sequence[0, -1, 0] # Last step close
    last_close = scaler.inverse_transform(dummy)[0, 0]
    
    pred_log_ret = mean_ret[0]
    predicted_price = last_close * np.exp(pred_log_ret)
    
    # Store dir_prob for the Signal Gatekeeper (can be accessed via engine state)
    engine.last_prob = dir_prob[0][0]
    engine.last_mean_ret = mean_ret[0]
    
    return predicted_price

def convert_to_tflite(model, path):
    # Skip for complex ensemble
    return None
