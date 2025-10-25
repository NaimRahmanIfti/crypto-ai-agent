# app_trading_AUTO_REFRESH.py - True Auto-Refresh Every Minute
# No manual refresh needed!

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import CryptoDataCollector
from utils.feature_engineering import FeatureEngineering
from models.online_model import PricePredictor

try:
    from utils.trading_strategy import TradingStrategy
    TRADING_STRATEGY_AVAILABLE = True
except ImportError:
    TRADING_STRATEGY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AI Crypto Trader - Live",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
        border-right: 2px solid #667eea;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #667eea); }
        to { filter: drop-shadow(0 0 20px #764ba2); }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #ffffff;
        margin-top: -1rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .buy-signal {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 200, 83, 0.4);
        animation: pulse 2s ease-in-out infinite;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(244, 67, 54, 0.4);
        animation: pulse 2s ease-in-out infinite;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(255, 152, 0, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .live-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white;
        font-weight: bold;
        animation: blink 2s ease-in-out infinite;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .countdown {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        background: rgba(255, 255, 255, 0.2);
        color: white;
        font-weight: bold;
        margin-left: 1rem;
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.2);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Timeframe configs
TIMEFRAMES = {
    '1m': {'name': '⚡ 1 Minute', 'update': 60, 'limit': 200},
    '5m': {'name': '🔥 5 Minutes', 'update': 60, 'limit': 150},
    '15m': {'name': '📊 15 Minutes', 'update': 60, 'limit': 100},
    '1h': {'name': '⏰ 1 Hour', 'update': 300, 'limit': 80},
    '4h': {'name': '📈 4 Hours', 'update': 600, 'limit': 60},
}

EXCHANGE_PAIRS = {
    'kraken': {
        'Major 🥇': ['BTC/USD', 'ETH/USD'],
        'Popular ⭐': ['SOL/USD', 'XRP/USD', 'ADA/USD', 'DOGE/USD', 'DOT/USD', 'MATIC/USD'],
        'DeFi 🔷': ['AAVE/USD', 'UNI/USD', 'LINK/USD'],
    },
    'binance': {
        'Major 🥇': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'Popular ⭐': ['SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT'],
        'DeFi 🔷': ['AAVE/USDT', 'UNI/USDT', 'LINK/USDT'],
        'Meme 🐕': ['SHIB/USDT', 'PEPE/USDT', 'BONK/USDT'],
    }
}

COIN_INFO = {
    'BTC': '🥇 Bitcoin - Digital Gold',
    'ETH': '🥈 Ethereum - Smart Contracts',
    'SOL': '⚡ Solana - Fast Blockchain',
    'DOGE': '🐕 Dogecoin - Meme King',
    'SHIB': '🐕 Shiba Inu - Meme Token',
}

# Initialize session state
for key in ['collector', 'predictor', 'trading_strategy', 'data_history', 'is_trained']:
    if key not in st.session_state:
        st.session_state[key] = None
        
if 'feature_eng' not in st.session_state:
    st.session_state.feature_eng = FeatureEngineering()
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'update_interval' not in st.session_state:
    st.session_state.update_interval = 60
if 'current_timeframe' not in st.session_state:
    st.session_state.current_timeframe = '5m'

# Header
st.markdown('<div class="main-header">🚀 AI CRYPTO TRADER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">⚡ Live Auto-Updating Trading Signals</div>', unsafe_allow_html=True)

# Live indicator
if st.session_state.is_trained:
    st.markdown('<div class="live-indicator">🔴 LIVE</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    st.markdown("---")
    
    exchange = st.selectbox("🏦 Exchange", ["kraken", "binance"], index=0)
    pairs_by_category = EXCHANGE_PAIRS[exchange]
    category = st.selectbox("📊 Category", list(pairs_by_category.keys()))
    symbols = pairs_by_category[category]
    symbol = st.selectbox("💰 Trading Pair", symbols, index=0)
    
    coin_name = symbol.split('/')[0]
    if coin_name in COIN_INFO:
        st.markdown(f'<div class="info-box">{COIN_INFO[coin_name]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Timeframe
    st.markdown("## ⏰ TIMEFRAME")
    timeframe_options = list(TIMEFRAMES.keys())
    timeframe_names = [TIMEFRAMES[tf]['name'] for tf in timeframe_options]
    selected_tf_idx = st.selectbox(
        "📅 Chart Timeframe",
        range(len(timeframe_options)),
        format_func=lambda x: timeframe_names[x],
        index=1
    )
    selected_timeframe = timeframe_options[selected_tf_idx]
    
    st.markdown("---")
    
    model_type = st.selectbox("🤖 AI Model", ["logistic", "tree"], index=0)
    
    st.markdown("---")
    st.markdown("## 🎯 TRADING SETUP")
    
    risk_reward = st.slider("⚖️ Risk/Reward", 1.0, 3.0, 2.0, 0.5)
    
    # Auto-suggest hold time
    if selected_timeframe == '1m':
        default_hold = 3
    elif selected_timeframe == '5m':
        default_hold = 15
    elif selected_timeframe == '15m':
        default_hold = 45
    elif selected_timeframe == '1h':
        default_hold = 180
    else:
        default_hold = 720
    
    hold_minutes = st.slider("⏱️ Max Hold (min)", 1, 1440, default_hold)
    
    st.markdown("---")
    
    if st.button("🚀 INITIALIZE SYSTEM", type="primary"):
        with st.spinner(f"Initializing {symbol}..."):
            try:
                st.session_state.collector = CryptoDataCollector(exchange, symbol)
                st.session_state.predictor = PricePredictor(model_type)
                st.session_state.current_timeframe = selected_timeframe
                st.session_state.update_interval = 60  # Always update every 1 minute
                
                if TRADING_STRATEGY_AVAILABLE:
                    try:
                        st.session_state.trading_strategy = TradingStrategy(
                            risk_reward_ratio=risk_reward,
                            max_hold_time_seconds=hold_minutes * 60
                        )
                    except TypeError:
                        st.session_state.trading_strategy = TradingStrategy(
                            risk_reward_ratio=risk_reward,
                            max_hold_time_minutes=hold_minutes
                        )
                
                df = st.session_state.collector.fetch_ohlcv(
                    timeframe=selected_timeframe,
                    limit=TIMEFRAMES[selected_timeframe]['limit']
                )
                
                if df is not None and len(df) > 0:
                    df_features = st.session_state.feature_eng.compute_all_features(df)
                    feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                    st.session_state.predictor.train_on_historical(df_features, feature_cols)
                    st.session_state.data_history = df_features
                    st.session_state.is_trained = True
                    st.session_state.current_symbol = symbol
                    st.session_state.last_update_time = time.time()
                    st.success(f"✅ {symbol} Ready!")
                    st.rerun()
                else:
                    st.error("Failed to fetch data")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("## 📊 STATUS")
    
    if st.session_state.is_trained:
        st.success("✅ SYSTEM ACTIVE")
        st.metric("🪙 Trading", st.session_state.get('current_symbol', symbol))
        st.metric("📊 Timeframe", TIMEFRAMES[st.session_state.current_timeframe]['name'])
        st.metric("🔄 Update", "Every 1 minute")
        
        if st.session_state.last_update_time:
            elapsed = int(time.time() - st.session_state.last_update_time)
            next_update = max(0, 60 - elapsed)
            st.metric("⏱️ Next Update", f"{next_update}s")
    else:
        st.warning("⚠️ Not Initialized")

# Main content
if not st.session_state.is_trained:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("""
        ### 👋 Welcome!
        
        **🎯 Quick Start:**
        1. Select exchange and coin
        2. Choose timeframe
        3. Click 'Initialize System'
        4. Watch live signals auto-update!
        
        💡 **Auto-Update:** Updates every 1 minute automatically!
        """)
else:
    # Check if it's time to update
    current_time = time.time()
    
    if st.session_state.last_update_time is None:
        should_update = True
    else:
        time_since_update = current_time - st.session_state.last_update_time
        should_update = time_since_update >= st.session_state.update_interval
    
    if should_update:
        # Fetch new data
        try:
            latest_data = st.session_state.collector.fetch_ohlcv(
                timeframe=st.session_state.current_timeframe,
                limit=TIMEFRAMES[st.session_state.current_timeframe]['limit']
            )
            
            if latest_data is not None and len(latest_data) > 0:
                df_features = st.session_state.feature_eng.compute_all_features(latest_data)
                
                if len(df_features) > 0:
                    # Update stored data
                    st.session_state.data_history = df_features
                    st.session_state.last_update_time = current_time
        except Exception as e:
            st.error(f"Update error: {e}")
    
    # Display current data
    if st.session_state.data_history is not None and len(st.session_state.data_history) > 0:
        df_features = st.session_state.data_history
        latest_row = df_features.iloc[-1]
        feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
        features_dict = {col: latest_row[col] for col in feature_cols if col in latest_row.index}
        
        prediction = st.session_state.predictor.predict_next(features_dict)
        current_price = latest_row['close']
        
        if st.session_state.trading_strategy:
            trading_signal = st.session_state.trading_strategy.generate_trading_signal(
                df_features, prediction, current_price
            )
            formatted_signal = st.session_state.trading_strategy.format_signal_for_display(trading_signal)
            action = trading_signal['action']
        else:
            action = 'HOLD'
            formatted_signal = {
                'action': f"{'🟢 UP' if prediction['direction'] == 'UP' else '🔴 DOWN'}",
                'message': f"Predicted: {prediction['direction']}"
            }
            trading_signal = {'action': 'HOLD'}
        
        # Display
        coin_symbol = st.session_state.current_symbol.split('/')[0]
        
        # Time info
        if st.session_state.last_update_time:
            elapsed = int(time.time() - st.session_state.last_update_time)
            next_update = max(0, 60 - elapsed)
            time_str = datetime.fromtimestamp(st.session_state.last_update_time).strftime('%H:%M:%S')
            st.markdown(f'<div class="countdown">⏱️ Updated: {time_str} | Next: {next_update}s</div>', unsafe_allow_html=True)
        
        st.markdown(f"### 🎯 {coin_symbol} - Live Trading Signal")
        
        if action == 'BUY':
            st.markdown(f'<div class="buy-signal">{formatted_signal["action"]}<br><span style="font-size:1.2rem;">{formatted_signal["message"]}</span></div>', unsafe_allow_html=True)
        elif action == 'SELL':
            st.markdown(f'<div class="sell-signal">{formatted_signal["action"]}<br><span style="font-size:1.2rem;">{formatted_signal["message"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hold-signal">{formatted_signal["action"]}<br><span style="font-size:1.2rem;">{formatted_signal.get("message", "Waiting")}</span></div>', unsafe_allow_html=True)
        
        if action in ['BUY', 'SELL'] and st.session_state.trading_strategy:
            cols = st.columns(4)
            with cols[0]:
                st.metric("📍 Entry", formatted_signal['entry'], formatted_signal['potential_profit'])
            with cols[1]:
                st.metric("🎯 Target", formatted_signal['take_profit'], formatted_signal['risk_reward'])
            with cols[2]:
                st.metric("🛑 Stop", formatted_signal['stop_loss'], formatted_signal['potential_loss'])
            with cols[3]:
                st.metric("⏱️ Hold", formatted_signal['hold_time'], formatted_signal['confidence'])
        
        st.markdown("---")
        
        # Chart
        st.markdown(f"### 📈 {st.session_state.current_symbol} - {TIMEFRAMES[st.session_state.current_timeframe]['name']}")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'RSI')
        )
        
        fig.add_trace(
            go.Candlestick(
                x=df_features['timestamp'],
                open=df_features['open'],
                high=df_features['high'],
                low=df_features['low'],
                close=df_features['close'],
                name='Price',
                increasing_line_color='#00c853',
                decreasing_line_color='#f44336'
            ),
            row=1, col=1
        )
        
        if action in ['BUY', 'SELL'] and st.session_state.trading_strategy:
            fig.add_hline(y=trading_signal['entry_price'], line_dash="solid", 
                         line_color="#2196F3", line_width=2, annotation_text="ENTRY", row=1, col=1)
            fig.add_hline(y=trading_signal['take_profit'], line_dash="dash", 
                         line_color="#00c853", line_width=2, annotation_text="TARGET", row=1, col=1)
            fig.add_hline(y=trading_signal['stop_loss'], line_dash="dash", 
                         line_color="#f44336", line_width=2, annotation_text="STOP", row=1, col=1)
        
        if 'RSI' in df_features.columns:
            fig.add_trace(
                go.Scatter(x=df_features['timestamp'], y=df_features['RSI'],
                          name='RSI', line=dict(color='#667eea', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#f44336", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00c853", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.markdown("### 📊 Market Metrics")
        cols = st.columns(5)
        with cols[0]:
            st.metric("💵 Price", f"${current_price:,.2f}")
        with cols[1]:
            st.metric("📈 RSI", f"{latest_row.get('RSI', 50):.1f}")
        with cols[2]:
            st.metric("🎯 Prediction", prediction['direction'])
        with cols[3]:
            st.metric("💪 Confidence", f"{prediction['confidence']:.1%}")
        with cols[4]:
            st.metric("📊 Timeframe", st.session_state.current_timeframe)
    
    # Auto-refresh after interval
    time.sleep(1)  # Small delay to prevent CPU overload
    st.rerun()