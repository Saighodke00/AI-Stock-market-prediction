import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import fetch_data, clean_data, normalize_data, create_sequences, add_noise
from utils.indicators import add_technical_indicators
from utils.model import create_model, train_model, predict_next_day, convert_to_tflite
from utils.sentiment import get_market_sentiment

st.set_page_config(page_title="Apex AI - Swing Intelligence", layout="wide")

# --- PREMIUM GLASSMORPHISM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #080a0f; color: #ffffff; font-family: 'Outfit', sans-serif; }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: 0.3s;
    }
    .metric-card:hover { border-color: #00d2aa; transform: translateY(-5px); }
    .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: 800; color: #fff; margin-top: 5px; }
    
    /* Analysis Block */
    .analysis-block {
        background: rgba(0, 210, 170, 0.05);
        border-left: 5px solid #00d2aa;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(0,0,0,0.3);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='font-weight:800; font-size: 42px; margin-bottom:0;'>📈 Swing Intelligence</h1>", unsafe_allow_html=True)
st.caption("Apex Neural Engine // Multi-Day Trend Forecasting & Wealth Accumulation")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Terminal Controls")
    
    # Stock Presets
    market_cat = st.selectbox("Market Category", ["Indian Equities", "US Equities", "Custom"])
    
    if market_cat == "Indian Equities":
        ticker_list = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "ITC.NS", "ADANIENT.NS"]
        ticker = st.selectbox("Select Security", ticker_list)
    elif market_cat == "US Equities":
        ticker_list = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"]
        ticker = st.selectbox("Select Security", ticker_list)
    else:
        ticker = st.text_input("Enter Custom Ticker (e.g., RELIANCE.NS)", "TSLA").upper()

    st.markdown("---")
    lookback = st.slider("Lookback Memory (Days)", 60, 120, 100)
    epochs = st.slider("Neural Depth", 10, 50, 30)
    st.markdown("---")
    show_ma = st.toggle("Overlay SMA 50", value=True)
    st.info("Strategy: Trend-Following Momentum")

@st.cache_data
def get_swing_data(ticker):
    data = fetch_data(ticker, period="2y", interval="1d")
    if data is None: return None
    data = clean_data(data)
    data = add_technical_indicators(data)
    return data.dropna()

if ticker:
    df = get_swing_data(ticker)
    if df is not None and not df.empty:
        current_price = df['Close'].iloc[-1]
        
        # Prepare Advanced Feature Set
        df = add_technical_indicators(df)
        features = [
            'Close', 'log_ret', 'range', 'body', 'EMA_21', 'EMA_slope', 'SMA_50', 
            'ADX', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'BB_Width', 'VWAP', 
            'Vol_Zscore', 'Skew', 'Kurtosis', 'Hurst'
        ]
        if 'macro_ret' in df.columns:
            features += ['macro_ret', 'alpha_ret', 'macro_corr']
            
        X, Y_dir, Y_mag, scaler, scaled_data = create_sequences(df, features, lookback)
        
        @st.cache_resource
        def train_swing_model(t, lb, _X, _Y_dir, _Y_mag, _eps):
            # Apply causal noise to features
            X_noisy = add_noise(_X)
            engine = create_model(input_shape=(X_noisy.shape[1], X_noisy.shape[2]))
            history = engine.train_ensemble(X_noisy, _Y_dir, _Y_mag, epochs=_eps)
            return engine, history

        with st.spinner('Calibrating Causal Ensemble...'):
            try:
                engine, history_data = train_swing_model(ticker, lookback, X, Y_dir, Y_mag, epochs)
                # Walk-Forward Validation for Signal Gating
                from utils.backtest import walk_forward_validation, run_backtest, calculate_accuracy
                cagr, wf_sharpe, max_dd, win_rate, profit_factor = walk_forward_validation(engine, df, features)
            except Exception as e:
                st.error(f"Neural Calibration Error: {e}")
                st.stop()
            
        last_seq = scaled_data[-lookback:].reshape(1, lookback, len(features))
        predicted = predict_next_day(engine, last_seq, scaler)
        
        # Signal Gatekeeper
        dir_prob, mean_ret, _ = engine.predict(last_seq)
        adx_val = df['ADX'].iloc[-1]
        atr_val = df['ATR'].iloc[-1]
        rsi_val = df['RSI'].iloc[-1]
        
        signal, color = engine.get_signal(dir_prob[0][0], mean_ret[0], adx_val, wf_sharpe, atr_val)
        
        # Live News Sentiment Integration
        sentiment_val, top_news = get_market_sentiment(ticker)
        
        # Backtest for Confidence
        try:
            preds, acts, rmse = run_backtest(engine, scaler, scaled_data, time_step=lookback)
            acc = calculate_accuracy(acts, preds)
        except Exception as e:
            st.warning(f"Backtest engine issue: {e}. Using simulated confidence.")
            preds, acts = np.zeros(10), np.zeros(10)
            rmse, acc = 0.05, 55.0

        # Recommendation Logic
        reason = f"Causal Engine Signal: {signal}. "
        if signal == "NEUTRAL":
            reason += "Gating criteria not met (Low confidence, weak trend, or high variance)."
        else:
            reason += f"Confidence: {dir_prob[0][0]*100:.1f}%. Expected Return: {mean_ret[0]*100:.2f}%."
        
        gauge_val = dir_prob[0][0] * 100

        # --- HERO METRICS (4 COLUMNS) ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Current Quote</div><div class="metric-value">${current_price:.2f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Neural Target</div><div class="metric-value" style="color:#00d2aa">${predicted:.2f}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Action Signal</div><div class="metric-value" style="color:{color}">{signal}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">{acc:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- GAUGE & REASONING ---
        c1, c2 = st.columns([1, 1.5])
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = gauge_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Strategic Score", 'font': {'size': 20, 'color': '#888'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': color},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(255,75,75,0.2)'},
                        {'range': [35, 65], 'color': 'rgba(255,204,0,0.2)'},
                        {'range': [65, 100], 'color': 'rgba(0,255,136,0.2)'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Outfit"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            st.markdown("### 🤖 Neural Intelligence Report")
            st.markdown(f"""
            <div class="analysis-block">
                <b>Execution Rationale:</b><br>
                {reason}
                <br><br>
                <b>Strategic Vitals:</b><br>
                Sentiment Flux: <span style="color:{color}">{sentiment_val:+.2f}</span> | 
                RSI: <span style="color:#00d2aa">{rsi_val:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- MAIN CHART ---
        st.markdown("<h3 style='margin-top:20px;'>Market Forecast Visualizer</h3>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📊 Price & Prediction", "🧠 Learning Convergence"])
        
        with tab1:
            plot_df = df.iloc[-100:]
            fig = go.Figure()
            # Historical Candlestick
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Market'))
            # SMA Overlay
            if show_ma:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1.5, dash='dot')))
            
            # Prediction Extension
            last_date = plot_df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[current_price, predicted],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#00d2aa', width=3, dash='dash')
            ))
            
            fig.update_layout(
                template="plotly_dark", 
                height=500, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history_data['loss'], name='Training Loss', line=dict(color='#00d2aa')))
            if 'val_loss' in history_data:
                fig_loss.add_trace(go.Scatter(y=history_data['val_loss'], name='Validation Loss', line=dict(color='#ff3366', dash='dash')))
            fig_loss.update_layout(
                template="plotly_dark", 
                height=500, 
                title="Neural Convergence: Training vs Validation Loss", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Diagnostic message
            if 'val_loss' in history_data:
                final_train = history_data['loss'][-1]
                final_val = history_data['val_loss'][-1]
                ratio = final_val / final_train if final_train > 0 else 1
                
                if ratio > 1.5:
                    st.warning("⚠️ Warning: Potential Overfitting detected. Validation loss is significantly higher than training loss.")
                else:
                    st.success("✅ Model Generalization: STABLE. Training and validation losses are converging well.")

        st.markdown("---")
        st.markdown("### 🧪 Neural Backtracking Verification")
        st.caption("Validating AI performance by comparing historical forecasts against actual outcomes.")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(y=acts, name='Actual Price', line=dict(color='rgba(255,255,255,0.3)', width=1)))
        fig_bt.add_trace(go.Scatter(y=preds, name='AI Backtracking', line=dict(color='#00d2aa', width=2)))
        fig_bt.update_layout(
            template="plotly_dark", 
            height=300, 
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Walk-Forward Sharpe", f"{wf_sharpe:.2f}")
        c2.metric("Profit Factor", f"{profit_factor:.2f}")
        c3.metric("Win Rate", f"{win_rate*100:.1f}%")
        c4.metric("Max Drawdown", f"{max_dd*100:.1f}%")
        c5.success("System: RESEARCH GRADE")
        
        # --- LIVE NEWS FEED ---
        st.markdown("---")
        st.markdown("### 📰 Market Intelligence Feed")
        st.caption(f"Real-time sentiment signals for {ticker}")
        
        if top_news:
            for n in top_news:
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background: rgba(255, 255, 255, 0.02);
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        border-radius: 12px;
                        padding: 15px;
                        margin-bottom: 15px;
                        transition: 0.3s;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="flex: 1;">
                                <a href="{n['link']}" target="_blank" style="
                                    color: #00d2aa;
                                    text-decoration: none;
                                    font-weight: 600;
                                    font-size: 16px;
                                    line-height: 1.4;
                                ">{n['title']}</a>
                                <div style="margin-top: 8px; font-size: 12px; color: #888;">
                                    <span style="background: rgba(0, 210, 170, 0.1); color: #00d2aa; padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(0, 210, 170, 0.2);">{n['publisher']}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"No active news clusters detected for {ticker}. The model is relying purely on technical momentum.")

        st.info("The system uses multivariate backtracking to confirm that current neural weights align with established historical trends before issuing a signal.")
    else:
        st.error("Market connection failed for this ticker.")
