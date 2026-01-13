import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the current directory to path so we can import utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import fetch_data, clean_data, normalize_data, create_sequences
from utils.indicators import add_technical_indicators
from utils.model import create_model, train_model, predict_next_day

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Apex AI - Stock Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK MODE & GLASSMORPHISM ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 14px;
        color: #888;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #fff;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #111;
    }
    /* Buttons */
    .stButton>button {
        background-color: #00d2aa;
        color: white;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ Apex AI Controls")
    ticker = st.text_input("Stock Ticker", "TSLA", help="Enter stock symbol (e.g., AAPL, BTC-USD)").upper()
    prediction_horizon = st.selectbox("Prediction Horizon", ["Next Day", "Next 3 Days", "Next 7 Days"])
    
    st.markdown("---")
    st.subheader("Indicators")
    show_sma = st.toggle("Show SMA(50)", value=True)
    show_rsi = st.toggle("Show RSI", value=True)
    
    st.markdown("---")
    st.info("System uses LSTM Neural Networks trained on 2 years of historical data.")

# --- MAIN APP ---

@st.cache_data
def get_data(ticker):
    """Fetches and processes data, cached by Streamlit."""
    data = fetch_data(ticker)
    if data is None:
        return None
    data = clean_data(data)
    data = add_technical_indicators(data)
    # Drop NaN created by indicators
    data = data.dropna()
    return data

if ticker:
    # 1. Load Data
    with st.spinner('Fetching market data...'):
        df = get_data(ticker)

    if df is not None:
        current_price = df['Close'].iloc[-1]
        
        # 2. Prepare Data for AI
        
        # Normalize
        scaler, scaled_data = normalize_data(df, feature_column='Close')
        
        # Create Sequences
        time_step = 60
        X, y = create_sequences(scaled_data, time_step)
        
        # Reshape for LSTM (samples, time steps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # 3. Model Training (or Load)
        @st.cache_resource
        def get_trained_model(ticker_name, _X, _y):
            # We use ticker_name to invalidate cache if ticker changes
            model = create_model(input_shape=(_X.shape[1], 1))
            train_model(model, _X, _y, epochs=20, batch_size=32) 
            return model

        with st.spinner('Training AI Model...'):
            model = get_trained_model(ticker, X, y)
        
        # 4. Prediction
        last_60_days = scaled_data[-time_step:]
        last_60_days = last_60_days.reshape(1, time_step, 1)
        
        predicted_price = predict_next_day(model, last_60_days, scaler)
        
        # 5. Logic (Strict Requirements)
        rsi_val = float(df['RSI'].iloc[-1])
        predicted_price = float(predicted_price)
        current_price = float(current_price)
        
        # Logic: 
        # BUY if Predicted > Current * 1.03 AND RSI < 35
        # SELL if Predicted < Current * 0.97 OR RSI > 70
        # HOLD otherwise
        
        if predicted_price > current_price * 1.03 and rsi_val < 35:
            signal = "BUY"
            signal_color = "#00FF00" # Green
            reason = "Price predicted to rise >3% and RSI indicates oversold conditions (<35)."
            gauge_val = 85
        elif predicted_price < current_price * 0.97 or rsi_val > 70:
            signal = "SELL"
            signal_color = "#FF0000" # Red
            reason = "Price predicted to fall >3% or RSI indicates overbought conditions (>70)."
            gauge_val = 15
        else:
            signal = "HOLD"
            signal_color = "#FFFF00" # Yellow
            reason = "No significant trend detected or conflicting signals."
            gauge_val = 50

        confidence = 82 # Static as per request
        
        # --- UI LAYOUT ---
        st.subheader(f"Analyzing {ticker}...")

        # Hero Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${current_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Target (Next Day)</div>
                <div class="metric-value" style="color: #00d2aa;">${predicted_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Signal</div>
                <div class="metric-value" style="color: {signal_color};">{signal}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Gauge & Reasoning
        c1, c2 = st.columns([1, 2])
        
        with c1:
            # Simple Gauge using Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = gauge_val,
                title = {'text': "AI Recommendation Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': signal_color},
                    'steps': [
                        {'range': [0, 33], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [33, 66], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [66, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            st.markdown("### 🤖 AI Analysis")
            st.info(f"**Why {signal}?**\n\n{reason}\n\n*RSI is at {rsi_val:.1f}, indicating market condition.*")
            
        # Main Chart
        st.subheader("Price History & Forecast")
        
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'], 
            mode='lines', 
            name='Historical Price',
            line=dict(color='#00d2aa')
        ))
        
        # SMA
        if show_sma:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1)
            ))
            
        # Prediction Point (Visualized as a line extension for next day)
        # We need to add the next day date
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[current_price, predicted_price],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='cyan', dash='dash')
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Summary
        st.subheader("Technical Indicators")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.dataframe(df.tail(5)[['Close', 'RSI', 'SMA_50']].style.format("{:.2f}"))
            
        # --- BACKTESTING SECTION ---
        st.markdown("---")
        st.subheader("🧪 Backtesting Results")
        
        if st.button("Run Backtest (Latest Data)"):
            from utils.backtest import run_backtest, calculate_accuracy
            
            with st.spinner("Running historical backtest..."):
                predictions, actuals, rmse = run_backtest(model, scaler, scaled_data)
                accuracy = calculate_accuracy(actuals, predictions)
                
                # Show Backtest Metrics
                b1, b2 = st.columns(2)
                b1.metric("RMSE Error", f"{rmse:.4f}")
                b2.metric("Directional Accuracy", f"{accuracy:.1f}%")
                
                # Plot Backtest
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual Price', line=dict(color='gray')))
                fig_bt.add_trace(go.Scatter(y=predictions[:,0], mode='lines', name='Predicted Price', line=dict(color='cyan')))
                fig_bt.update_layout(title="Backtest: Actual vs Predicted", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bt, use_container_width=True)
        
    else:
        st.error("No data found for this ticker. Please try again.")

else:
    st.info("Enter a stock ticker in the sidebar to begin.")
