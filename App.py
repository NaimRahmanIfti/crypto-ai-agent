import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import CryptoDataCollector
from utils.feature_engineering import FeatureEngineering
from models.online_model import PricePredictor

# Page configuration
st.set_page_config(
    page_title="Crypto AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-up {
        color: #00c853;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-down {
        color: #d50000;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'collector' not in st.session_state:
    st.session_state.collector = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'feature_eng' not in st.session_state:
    st.session_state.feature_eng = FeatureEngineering()
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame()
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# Title
st.markdown('<div class="main-header">🚀 Real-time Crypto AI Price Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Exchange and Symbol selection
    exchange = st.selectbox("Exchange", ["kraken", "binance"], index=0)
    # Symbol selection based on exchange
    if exchange == "kraken":
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
        default_symbol = "BTC/USD"
    else:
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
        default_symbol = "BTC/USDT"
    
    symbol = st.selectbox("Trading Pair", symbols, index=0)
    
    # Model selection
    model_type = st.selectbox("AI Model", ["logistic", "forest", "tree"], index=0)
    
    st.markdown("---")
    
    # Initialize button
    if st.button("🔄 Initialize System", type="primary"):
        with st.spinner("Initializing..."):
            try:
                # Create collector
                st.session_state.collector = CryptoDataCollector(
                    exchange_name=exchange,
                    symbol=symbol
                )
                
                # Create predictor
                st.session_state.predictor = PricePredictor(model_type=model_type)
                
                # Fetch historical data for training
                st.info("Fetching historical data for training...")
                df = st.session_state.collector.fetch_ohlcv(timeframe='1m', limit=200)
                
                if df is not None and len(df) > 0:
                    # Compute features
                    st.info("Computing features...")
                    df_features = st.session_state.feature_eng.compute_all_features(df)
                    
                    # Train model
                    st.info("Training AI model...")
                    feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                    st.session_state.predictor.train_on_historical(df_features, feature_cols)
                    
                    # Store data
                    st.session_state.data_history = df_features
                    st.session_state.is_trained = True
                    
                    st.success("✅ System initialized and trained!")
                else:
                    st.error("Failed to fetch historical data")
            except Exception as e:
                st.error(f"Initialization error: {e}")
    
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    if st.session_state.is_trained:
        st.success("✅ System Active")
        stats = st.session_state.predictor.get_stats()
        st.metric("Training Samples", stats['training_count'])
        st.metric("Overall Accuracy", f"{stats['overall_accuracy']:.2%}")
        st.metric("Recent Accuracy (50)", f"{stats['recent_accuracy_50']:.2%}")
    else:
        st.warning("⚠️ System Not Initialized")
    
    st.markdown("---")
    st.info("💡 **Tip:** Initialize the system first, then the dashboard will update automatically with live predictions!")

# Main content
if not st.session_state.is_trained:
    st.info("👈 Please initialize the system using the sidebar to start.")
else:
    # Create main layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Auto-refresh placeholder
    placeholder = st.empty()
    
    # Live update loop
    for i in range(100):  # Continuous updates
        with placeholder.container():
            try:
                # Fetch latest data
                latest_data = st.session_state.collector.fetch_ohlcv(timeframe='1m', limit=100)
                
                if latest_data is not None and len(latest_data) > 0:
                    # Compute features for latest data
                    df_features = st.session_state.feature_eng.compute_all_features(latest_data)
                    
                    if len(df_features) > 0:
                        # Get latest row for prediction
                        latest_row = df_features.iloc[-1]
                        feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                        features_dict = {col: latest_row[col] for col in feature_cols if col in latest_row.index}
                        
                        # Make prediction
                        prediction = st.session_state.predictor.predict_next(features_dict)
                        
                        # Store prediction
                        prediction['actual_price'] = latest_row['close']
                        st.session_state.prediction_history.append(prediction)
                        
                        # Update data history
                        st.session_state.data_history = df_features
                        
                        # Display current price and prediction
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.subheader("💰 Current Price")
                            current_price = latest_row['close']
                            price_change = latest_row['price_change_pct']
                            
                            col_price1, col_price2 = st.columns(2)
                            with col_price1:
                                st.metric(
                                    label=symbol,
                                    value=f"${current_price:,.2f}",
                                    delta=f"{price_change:+.2f}%"
                                )
                            with col_price2:
                                st.metric(
                                    label="Volume",
                                    value=f"${latest_row['volume']:,.0f}"
                                )
                        
                        with col2:
                            st.subheader("🤖 AI Prediction")
                            direction_class = "prediction-up" if prediction['direction'] == 'UP' else "prediction-down"
                            st.markdown(f'<div class="{direction_class}">{"📈 " + prediction["direction"]}</div>', unsafe_allow_html=True)
                            st.metric("Confidence", f"{prediction['confidence']:.1%}")
                        
                        with col3:
                            st.subheader("📊 Model Stats")
                            stats = st.session_state.predictor.get_stats()
                            st.metric("Accuracy", f"{stats['overall_accuracy']:.2%}")
                            st.metric("Recent (50)", f"{stats['recent_accuracy_50']:.2%}")
                        
                        # Price chart
                        st.subheader("📈 Price Chart with Predictions")
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # Candlestick chart
                        fig.add_trace(
                            go.Candlestick(
                                x=df_features['timestamp'],
                                open=df_features['open'],
                                high=df_features['high'],
                                low=df_features['low'],
                                close=df_features['close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )
                        
                        # Add moving averages
                        if 'SMA_21' in df_features.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_features['timestamp'],
                                    y=df_features['SMA_21'],
                                    name='SMA 21',
                                    line=dict(color='orange', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        if 'EMA_21' in df_features.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_features['timestamp'],
                                    y=df_features['EMA_21'],
                                    name='EMA 21',
                                    line=dict(color='blue', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        # Volume chart
                        colors = ['red' if row['close'] < row['open'] else 'green' for idx, row in df_features.iterrows()]
                        fig.add_trace(
                            go.Bar(
                                x=df_features['timestamp'],
                                y=df_features['volume'],
                                name='Volume',
                                marker_color=colors
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            title=f"{symbol} - Live Price Chart",
                            xaxis_title="Time",
                            yaxis_title="Price (USD)",
                            height=600,
                            showlegend=True,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Technical indicators
                        st.subheader("📊 Technical Indicators")
                        
                        col_ind1, col_ind2, col_ind3, col_ind4 = st.columns(4)
                        
                        with col_ind1:
                            if 'RSI' in latest_row:
                                rsi_value = latest_row['RSI']
                                rsi_color = "🟢" if 30 <= rsi_value <= 70 else "🔴"
                                st.metric("RSI", f"{rsi_value:.2f} {rsi_color}")
                        
                        with col_ind2:
                            if 'MACD' in latest_row:
                                st.metric("MACD", f"{latest_row['MACD']:.2f}")
                        
                        with col_ind3:
                            if 'BB_width' in latest_row:
                                st.metric("BB Width", f"{latest_row['BB_width']:.2f}")
                        
                        with col_ind4:
                            if 'ATR' in latest_row:
                                st.metric("ATR", f"{latest_row['ATR']:.2f}")
                        
                        # Prediction history
                        if len(st.session_state.prediction_history) > 1:
                            st.subheader("🎯 Recent Predictions")
                            
                            recent_preds = st.session_state.prediction_history[-10:]
                            pred_df = pd.DataFrame(recent_preds)
                            
                            # Display as table
                            display_df = pred_df[['timestamp', 'direction', 'confidence', 'actual_price']].copy()
                            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
                            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                            display_df['actual_price'] = display_df['actual_price'].apply(lambda x: f"${x:,.2f}")
                            
                            st.dataframe(display_df, use_container_width=True)
                        
                        # Last update time
                        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
            except Exception as e:
                st.error(f"Update error: {e}")
        
        # Wait before next update
        time.sleep(30)  # Update every 30 seconds
        
        # Force rerun
        st.rerun()