import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def run_backtest(model, scaler, scaled_data, split_ratio=0.8, time_step=60):
    """
    Runs a backtest on the data.
    """
    train_size = int(len(scaled_data) * split_ratio)
    test_data = scaled_data[train_size - time_step:]
    
    X_test, y_test = [], []
    for i in range(time_step, len(test_data)):
        X_test.append(test_data[i-time_step:i, 0])
        y_test.append(test_data[i, 0])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Inverse Transform
    predictions = scaler.inverse_transform(predictions)
    y_test_check = scaler.inverse_transform([y_test])
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test_check[0], predictions[:,0]))
    
    return predictions, y_test_check[0], rmse

def calculate_accuracy(actual, predicted):
    """
    Calculates directional accuracy.
    """
    correct = 0
    total = len(actual) - 1
    for i in range(total):
        act_diff = actual[i+1] - actual[i]
        pred_diff = predicted[i+1] - actual[i] # Compare prediction against previous actual
        
        if (act_diff > 0 and pred_diff > 0) or (act_diff < 0 and pred_diff < 0):
            correct += 1
            
    return (correct / total) * 100 if total > 0 else 0
