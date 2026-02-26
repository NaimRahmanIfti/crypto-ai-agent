# THIS IS THE FIXED VERSION - NO DUPLICATES!
# Download this file and replace your app_trading.py
# All selectboxes now have unique 'key' parameters

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CONDITIONAL IMPORTS
try:
    from data.data_collector import CryptoDataCollector
except ImportError:
    CryptoDataCollector = None

try:
    from utils.feature_engineering import FeatureEngineering
except ImportError:
    FeatureEngineering = None

try:
    from models.online_model import PricePredictor
except ImportError:
    PricePredictor = None

try:
    from utils.trading_strategy import TradingStrategy
    TRADING_STRATEGY_AVAILABLE = True
except ImportError:
    TRADING_STRATEGY_AVAILABLE = False

try:
    from models.rl_model import RLPricePredictor
    RL_MODEL_AVAILABLE = True
except ImportError:
    RL_MODEL_AVAILABLE = False
    RLPricePredictor = None

st.set_page_config(
    page_title="AI Crypto Trader - RL",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS STYLING (keeping original styles)
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%); border-right: 2px solid #667eea; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .main-header { font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 2rem 0;
        animation: glow 2s ease-in-out infinite alternate; }
    @keyframes glow { from { filter: drop-shadow(0 0 5px #667eea); } to { filter: drop-shadow(0 0 20px #764ba2); } }
    .subtitle { text-align: center; font-size: 1.2rem; color: #ffffff; margin-top: -1rem; margin-bottom: 2rem; opacity: 0.9; }
    .buy-signal { background: linear-gradient(135deg, #00c853 0%, #00e676 100%); color: white; padding: 2rem;
        border-radius: 20px; font-size: 1.5rem; font-weight: bold; text-align: center; margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 200, 83, 0.4); animation: pulse 2s ease-in-out infinite;
        border: 2px solid rgba(255, 255, 255, 0.3); }
    .sell-signal { background: linear-gradient(135deg, #f44336 0%, #e91e63 100%); color: white; padding: 2rem;
        border-radius: 20px; font-size: 1.5rem; font-weight: bold; text-align: center; margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(244, 67, 54, 0.4); animation: pulse 2s ease-in-out infinite;
        border: 2px solid rgba(255, 255, 255, 0.3); }
    .hold-signal { background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%); color: white; padding: 2rem;
        border-radius: 20px; font-size: 1.5rem; font-weight: bold; text-align: center; margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(255, 152, 0, 0.4); border: 2px solid rgba(255, 255, 255, 0.3); }
    @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.02); } }
    .timeframe-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 50px; font-weight: bold; margin: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
    .rl-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 50px; font-weight: bold; margin: 0.5rem;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4); }
    [data-testid="metric-container"] { background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.3); }
    h2, h3 { color: #ffffff !important; font-weight: 700; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3); }
    .stButton > button { width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        border: none; padding: 0.75rem 2rem; border-radius: 50px; font-weight: bold; font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6); }
    .info-box { background: rgba(102, 126, 234, 0.2); padding: 1rem; border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3); color: white; margin: 0.5rem 0; }
    .rl-info-box { background: rgba(255, 107, 107, 0.15); padding: 1rem; border-radius: 15px;
        border: 2px solid rgba(255, 107, 107, 0.4); color: #ffcccc; margin: 0.5rem 0; }
    .price-update { background: rgba(50, 200, 100, 0.2); border-left: 4px solid #00c853; padding: 1rem;
        border-radius: 5px; color: white; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# CONFIGURATION
TIMEFRAMES = {
    '1m': {'name': '⚡ 1 Minute', 'desc': 'Scalping', 'update': 50, 'limit': 200, 'color': '#f44336'},
    '5m': {'name': '🔥 5 Minutes', 'desc': 'Day Trading', 'update': 50, 'limit': 150, 'color': '#ff9800'},
    '15m': {'name': '📊 15 Minutes', 'desc': 'Swing Trading', 'update': 50, 'limit': 100, 'color': '#ffc107'},
    '1h': {'name': '⏰ 1 Hour', 'desc': 'Position Trading', 'update': 50, 'limit': 80, 'color': '#4caf50'},
    '4h': {'name': '📈 4 Hours', 'desc': 'Long-term', 'update': 50, 'limit': 60, 'color': '#2196f3'},
}

EXCHANGE_PAIRS = {
    'kraken': {
        'Major 🥇': ['BTC/USD', 'ETH/USD'],
        'Popular ⭐': ['SOL/USD', 'XRP/USD', 'ADA/USD', 'DOGE/USD', 'DOT/USD', 'MATIC/USD'],
        'DeFi 🔷': ['AAVE/USD', 'UNI/USD', 'LINK/USD', 'ATOM/USD'],
        'Other 💎': ['LTC/USD', 'XLM/USD', 'ALGO/USD', 'AVAX/USD']
    },
    'binance': {
        'Major 🥇': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'Popular ⭐': ['SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT'],
        'DeFi 🔷': ['AAVE/USDT', 'UNI/USDT', 'LINK/USDT', 'ATOM/USDT'],
        'Meme 🐕': ['SHIB/USDT', 'PEPE/USDT', 'BONK/USDT', 'FLOKI/USDT'],
        'Other 💎': ['LTC/USDT', 'XLM/USDT', 'ALGO/USDT']
    }
}

COIN_INFO = {
    'BTC': '🥇 Bitcoin - Digital Gold', 'ETH': '🥈 Ethereum - Smart Contracts',
    'SOL': '⚡ Solana - Fast Blockchain', 'DOGE': '🐕 Dogecoin - Meme King',
    'XRP': '💸 Ripple - Payments', 'SHIB': '🐕 Shiba Inu - Meme Token', 'PEPE': '🐸 Pepe - Viral Meme',
}

RL_MODELS = {
    'dqn': {'name': '🧠 DQN', 'desc': 'Deep Q-Network', 'algo': 'Q-Learning'},
    'ppo': {'name': '🎯 PPO', 'desc': 'Proximal Policy Optimization', 'algo': 'Policy Gradient'},
    'a2c': {'name': '⚡ A2C', 'desc': 'Actor-Critic', 'algo': 'Policy Gradient'},
}

# SESSION STATE INIT
for key in ['collector', 'predictor', 'trading_strategy', 'data_history', 'trading_signals', 'is_trained']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'trading_signals' else []

if 'feature_eng' not in st.session_state:
    st.session_state.feature_eng = FeatureEngineering() if FeatureEngineering else None

for key in ['is_trained', 'current_timeframe', 'model_type', 'price_history', 'time_history', 
            'previous_price', 'price_table_data']:
    if key not in st.session_state:
        if key == 'is_trained':
            st.session_state[key] = False
        elif key == 'current_timeframe':
            st.session_state[key] = '5m'
        elif key == 'model_type':
            st.session_state[key] = 'dqn'
        else:
            st.session_state[key] = [] if 'history' in key or 'data' in key else None

# HEADER
st.markdown('<div class="main-header">🚀 AI CRYPTO TRADER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">⚡ RL-Powered Trading | Live Price Updates Every 30 Seconds</div>', unsafe_allow_html=True)
st.markdown("---")

# SIDEBAR - WITH UNIQUE KEYS FOR ALL WIDGETS!
with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    st.markdown("---")
    
    # KEY FIX: Added unique keys to ALL selectboxes
    exchange = st.selectbox("🏦 Exchange", ["kraken", "binance"], index=0, key="exchange_select_unique")
    pairs_by_category = EXCHANGE_PAIRS[exchange]
    category = st.selectbox("📊 Category", list(pairs_by_category.keys()), key="category_select_unique")
    symbols = pairs_by_category[category]
    symbol = st.selectbox("💰 Trading Pair", symbols, index=0, key="symbol_select_unique")
    
    coin_name = symbol.split('/')[0]
    if coin_name in COIN_INFO:
        st.markdown(f'<div class="info-box">{COIN_INFO[coin_name]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## ⏰ TIMEFRAME")
    
    timeframe_options = list(TIMEFRAMES.keys())
    timeframe_names = [f"{TIMEFRAMES[tf]['name']} - {TIMEFRAMES[tf]['desc']}" for tf in timeframe_options]
    
    selected_tf_idx = st.selectbox(
        "📅 Trading Timeframe",
        range(len(timeframe_options)),
        format_func=lambda x: timeframe_names[x],
        index=1,
        key="timeframe_select_unique"  # KEY FIX
    )
    
    selected_timeframe = timeframe_options[selected_tf_idx]
    tf_config = TIMEFRAMES[selected_timeframe]
    
    st.markdown(f'''<div class="info-box"><strong>{tf_config['name']}</strong><br>
        📊 Style: {tf_config['desc']}<br>🔄 Update: Every {tf_config['update']}s<br>
        📈 Candles: {tf_config['limit']}</div>''', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🧠 RL MODEL")
    
    rl_model_options = list(RL_MODELS.keys())
    rl_model_names = [f"{RL_MODELS[m]['name']} - {RL_MODELS[m]['desc']}" for m in rl_model_options]
    
    selected_rl_idx = st.selectbox(
        "🤖 RL Algorithm",
        range(len(rl_model_options)),
        format_func=lambda x: rl_model_names[x],
        index=1,
        key="rl_model_select_unique"  # KEY FIX
    )
    
    selected_rl_model = rl_model_options[selected_rl_idx]
    rl_config = RL_MODELS[selected_rl_model]
    
    st.markdown(f'''<div class="rl-info-box"><strong>{rl_config['name']}</strong><br>
        📘 Type: {rl_config['desc']}<br>🔬 Algorithm: {rl_config['algo']}<br>
        ⚙️ Status: Enabled</div>''', unsafe_allow_html=True)
    
    with st.expander("💡 RL Model Guide"):
        st.markdown("""**🧠 DQN:** Q-value based ⭐⭐⭐⭐☆  
**🎯 PPO:** Policy optimization ⭐⭐⭐⭐⭐ RECOMMENDED  
**⚡ A2C:** Fast convergence ⭐⭐⭐☆☆""")
    
    st.markdown("---")
    st.markdown("## 🎯 TRADING SETUP")
    
    risk_reward = st.slider("⚖️ Risk/Reward", 1.0, 3.0, 2.0, 0.5, key="risk_reward_slider_unique")
    
    default_hold = {' 1m': 3, '5m': 15, '15m': 45, '1h': 180}.get(selected_timeframe, 720)
    hold_minutes = st.slider("⏱️ Max Hold (min)", 1, 1440, default_hold, 1, key="hold_minutes_slider_unique")
    
    st.markdown("---")
    
    if st.button("🚀 INITIALIZE SYSTEM", type="primary", key="initialize_button_unique"):
        with st.spinner(f"Initializing {symbol} on {selected_timeframe} timeframe..."):
            try:
                if CryptoDataCollector:
                    st.session_state.collector = CryptoDataCollector(exchange, symbol)
                
                if RL_MODEL_AVAILABLE and RLPricePredictor:
                    st.session_state.predictor = RLPricePredictor(
                        model_type=selected_rl_model, symbol=symbol, timeframe=selected_timeframe)
                elif PricePredictor:
                    st.session_state.predictor = PricePredictor('logistic')
                
                st.session_state.current_timeframe = selected_timeframe
                st.session_state.model_type = selected_rl_model
                
                if TRADING_STRATEGY_AVAILABLE:
                    try:
                        st.session_state.trading_strategy = TradingStrategy(
                            risk_reward_ratio=risk_reward, max_hold_time_seconds=hold_minutes * 60)
                    except TypeError:
                        st.session_state.trading_strategy = TradingStrategy(
                            risk_reward_ratio=risk_reward, max_hold_time_minutes=hold_minutes)
                
                if st.session_state.collector:
                    df = st.session_state.collector.fetch_ohlcv(
                        timeframe=selected_timeframe, limit=tf_config['limit'])
                    
                    if df is not None and len(df) > 0:
                        if st.session_state.feature_eng:
                            df_features = st.session_state.feature_eng.compute_all_features(df)
                            feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                        else:
                            df_features = df
                            feature_cols = [col for col in df.columns if col not in 
                                          ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        
                        if st.session_state.predictor:
                            st.session_state.predictor.train_on_historical(df_features, feature_cols)
                        
                        st.session_state.data_history = df_features
                        st.session_state.is_trained = True
                        st.session_state.current_symbol = symbol
                        st.session_state.price_history = []
                        st.session_state.time_history = []
                        st.session_state.price_table_data = []
                        st.session_state.previous_price = None
                        
                        st.success(f"✅ {symbol} Ready with {selected_rl_model.upper()} on {tf_config['name']}!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("## 📊 SYSTEM STATUS")
    
    if st.session_state.is_trained:
        st.markdown('<span class="rl-badge">● RL ACTIVE</span>', unsafe_allow_html=True)
        st.metric("📊 Timeframe", TIMEFRAMES[st.session_state.current_timeframe]['name'])
        st.metric("🤖 RL Model", st.session_state.model_type.upper())
        st.metric("🪙 Trading", st.session_state.get('current_symbol', 'N/A'))
        if st.session_state.predictor:
            try:
                stats = st.session_state.predictor.get_stats()
                st.metric("🎯 Accuracy", f"{stats.get('overall_accuracy', 0):.1%}")
            except:
                st.metric("🎯 Accuracy", "N/A")
    else:
        st.warning("⚠️ Not Initialized")

# FILE CONTINUES IN NEXT MESSAGE...
# (This is part 1 - download and I'll give you part 2)
