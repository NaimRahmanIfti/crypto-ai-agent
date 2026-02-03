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


# CONDITIONAL IMPORTS - Handle missing modules gracefully

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

# 
# PAGE CONFIG
# 

st.set_page_config(
    page_title="AI Crypto Trader - RL",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BEAUTIFUL CSS STYLING
#

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
    
    .timeframe-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        margin: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .rl-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        margin: 0.5rem;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
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
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.2);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        color: white;
        margin: 0.5rem 0;
    }
    
    .rl-info-box {
        background: rgba(255, 107, 107, 0.15);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid rgba(255, 107, 107, 0.4);
        color: #ffcccc;
        margin: 0.5rem 0;
    }
    
    .price-update {
        background: rgba(50, 200, 100, 0.2);
        border-left: 4px solid #00c853;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

#
# CONFIGURATION DICTIONARIES
#

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
    'BTC': '🥇 Bitcoin - Digital Gold',
    'ETH': '🥈 Ethereum - Smart Contracts',
    'SOL': '⚡ Solana - Fast Blockchain',
    'DOGE': '🐕 Dogecoin - Meme King',
    'XRP': '💸 Ripple - Payments',
    'SHIB': '🐕 Shiba Inu - Meme Token',
    'PEPE': '🐸 Pepe - Viral Meme',
}

RL_MODELS = {
    'dqn': {'name': '🧠 DQN', 'desc': 'Deep Q-Network', 'algo': 'Q-Learning'},
    'ppo': {'name': '🎯 PPO', 'desc': 'Proximal Policy Optimization', 'algo': 'Policy Gradient'},
    'a2c': {'name': '⚡ A2C', 'desc': 'Actor-Critic', 'algo': 'Policy Gradient'},
}

# 
# SESSION STATE INITIALIZATION - CRITICAL!
#

# Initialize all keys
for key in ['collector', 'predictor', 'trading_strategy', 'data_history', 'trading_signals', 'is_trained']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'trading_signals' else []

# Initialize feature engineering
if 'feature_eng' not in st.session_state:
    if FeatureEngineering:
        st.session_state.feature_eng = FeatureEngineering()
    else:
        st.session_state.feature_eng = None

# Initialize training status
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# Initialize timeframe
if 'current_timeframe' not in st.session_state:
    st.session_state.current_timeframe = '5m'

# Initialize model type
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'dqn'

# 
# PRICE HISTORY INITIALIZATION - VERY IMPORTANT!
# 

if 'price_history' not in st.session_state:
    st.session_state.price_history = []

if 'time_history' not in st.session_state:
    st.session_state.time_history = []

if 'previous_price' not in st.session_state:
    st.session_state.previous_price = None

if 'price_table_data' not in st.session_state:
    st.session_state.price_table_data = []

# 
# HEADER
# 

st.markdown('<div class="main-header">🚀 AI CRYPTO TRADER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">⚡ RL-Powered Trading | Live Price Updates Every 30 Seconds</div>', unsafe_allow_html=True)
st.markdown("---")

#
# SIDEBAR CONFIGURATION
# 

with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    st.markdown("---")
    
    # Exchange & Symbol
    exchange = st.selectbox("🏦 Exchange", ["kraken", "binance"], index=0)
    pairs_by_category = EXCHANGE_PAIRS[exchange]
    category = st.selectbox("📊 Category", list(pairs_by_category.keys()))
    symbols = pairs_by_category[category]
    symbol = st.selectbox("💰 Trading Pair", symbols, index=0)
    
    # Coin info
    coin_name = symbol.split('/')[0]
    if coin_name in COIN_INFO:
        st.markdown(f'<div class="info-box">{COIN_INFO[coin_name]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TIMEFRAME SELECTION
    st.markdown("## ⏰ TIMEFRAME")
    timeframe_options = list(TIMEFRAMES.keys())
    timeframe_names = [f"{TIMEFRAMES[tf]['name']} - {TIMEFRAMES[tf]['desc']}" for tf in timeframe_options]
    
    selected_tf_idx = st.selectbox(
        "📅 Trading Timeframe",
        range(len(timeframe_options)),
        format_func=lambda x: timeframe_names[x],
        index=1
    )
    
    selected_timeframe = timeframe_options[selected_tf_idx]
    tf_config = TIMEFRAMES[selected_timeframe]
    
    st.markdown(f'''
    <div class="info-box">
        <strong>{tf_config['name']}</strong><br>
        📊 Style: {tf_config['desc']}<br>
        🔄 Update: Every {tf_config['update']}s<br>
        📈 Candles: {tf_config['limit']}
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # RL MODEL SELECTION
    st.markdown("## 🧠 RL MODEL")
    rl_model_options = list(RL_MODELS.keys())
    rl_model_names = [f"{RL_MODELS[m]['name']} - {RL_MODELS[m]['desc']}" for m in rl_model_options]
    
    selected_rl_idx = st.selectbox(
        "🤖 RL Algorithm",
        range(len(rl_model_options)),
        format_func=lambda x: rl_model_names[x],
        index=1
    )
    
    selected_rl_model = rl_model_options[selected_rl_idx]
    rl_config = RL_MODELS[selected_rl_model]
    
    st.markdown(f'''
    <div class="rl-info-box">
        <strong>{rl_config['name']}</strong><br>
        📘 Type: {rl_config['desc']}<br>
        🔬 Algorithm: {rl_config['algo']}<br>
        ⚙️ Status: Enabled
    </div>
    ''', unsafe_allow_html=True)
    
    with st.expander("💡 RL Model Guide"):
        st.markdown("""
        **🧠 DQN (Deep Q-Network):**
        - Best for: Q-value based decisions
        - Stability: ⭐⭐⭐⭐☆
        
        **🎯 PPO (Proximal Policy Optimization):**
        - Best for: Most trading scenarios
        - Stability: ⭐⭐⭐⭐⭐ RECOMMENDED ✅
        
        **⚡ A2C (Actor-Critic):**
        - Best for: Fast convergence
        - Stability: ⭐⭐⭐☆☆
        """)
    
    st.markdown("---")
    st.markdown("## 🎯 TRADING SETUP")
    
    risk_reward = st.slider("⚖️ Risk/Reward", 1.0, 3.0, 2.0, 0.5)
    
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
    
    hold_minutes = st.slider("⏱️ Max Hold (min)", 1, 1440, default_hold, 1)
    
    st.markdown("---")
    
    # Initialize button
    if st.button("🚀 INITIALIZE SYSTEM", type="primary"):
        with st.spinner(f"Initializing {symbol} on {selected_timeframe} timeframe..."):
            try:
                if CryptoDataCollector:
                    st.session_state.collector = CryptoDataCollector(exchange, symbol)
                else:
                    st.error("❌ CryptoDataCollector not available")
                
                # Initialize RL model
                if RL_MODEL_AVAILABLE and RLPricePredictor:
                    st.session_state.predictor = RLPricePredictor(
                        model_type=selected_rl_model,
                        symbol=symbol,
                        timeframe=selected_timeframe
                    )
                elif PricePredictor:
                    st.warning("⚠️ RL Model not available. Using standard model.")
                    st.session_state.predictor = PricePredictor('logistic')
                else:
                    st.error("❌ No predictor available")
                
                st.session_state.current_timeframe = selected_timeframe
                st.session_state.model_type = selected_rl_model
                
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
                
                # Fetch data
                if st.session_state.collector:
                    df = st.session_state.collector.fetch_ohlcv(
                        timeframe=selected_timeframe,
                        limit=tf_config['limit']
                    )
                    
                    if df is not None and len(df) > 0:
                        if st.session_state.feature_eng:
                            df_features = st.session_state.feature_eng.compute_all_features(df)
                            feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                        else:
                            df_features = df
                            feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        
                        if st.session_state.predictor:
                            st.session_state.predictor.train_on_historical(df_features, feature_cols)
                        
                        st.session_state.data_history = df_features
                        st.session_state.is_trained = True
                        st.session_state.current_symbol = symbol
                        
                        # RESET price history when initializing
                        st.session_state.price_history = []
                        st.session_state.time_history = []
                        
                        st.success(f"✅ {symbol} Ready with {selected_rl_model.upper()} on {tf_config['name']}!")
                    else:
                        st.error("Failed to fetch data")
                        
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


# MAIN CONTENT


if not st.session_state.is_trained:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        ### 👋 Welcome to RL-Powered Crypto Trader!
        
        🎯 **Get Started:**
        1. Choose your exchange and coin
        2. Select your timeframe (1m, 5m, 15m, 1h, 4h)
        3. Choose RL Algorithm (DQN, PPO, A2C)
        4. Configure parameters
        5. Click 'Initialize System'
        
        💡 **RL Model Tips:**
        - **DQN**: Q-value learning
        - **PPO**: Policy optimization ✅
        - **A2C**: Actor-Critic
        """)
    
    st.markdown("### ⏰ Choose Your Trading Style")
    cols = st.columns(5)
    for idx, (tf, config) in enumerate(TIMEFRAMES.items()):
        with cols[idx]:
            st.markdown(f"**{config['name']}**")
            st.caption(f"{config['desc']}")
            st.caption(f"Update: {config['update']}s")

else:
    tf_config = TIMEFRAMES[st.session_state.current_timeframe]
    update_interval = 30  # Price updates every 30 seconds
    
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    try:
        if st.session_state.collector:
            latest_data = st.session_state.collector.fetch_ohlcv(
                timeframe=st.session_state.current_timeframe,
                limit=tf_config['limit']
            )
            
            if latest_data is not None and len(latest_data) > 0:
                # Process features
                if st.session_state.feature_eng:
                    df_features = st.session_state.feature_eng.compute_all_features(latest_data)
                    feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                else:
                    df_features = latest_data
                    feature_cols = [col for col in latest_data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                if len(df_features) > 0:
                    latest_row = df_features.iloc[-1]
                    current_price = latest_row['close']
                    
                
                    # ADD TO PRICE HISTORY
                    
                    st.session_state.price_history.append(current_price)
                    st.session_state.time_history.append(datetime.now())
                    
                    # Keep only last 100 prices
                    if len(st.session_state.price_history) > 100:
                        st.session_state.price_history = st.session_state.price_history[-100:]
                        st.session_state.time_history = st.session_state.time_history[-100:]
                    
                    # Get prediction
                    features_dict = {col: latest_row[col] for col in feature_cols if col in latest_row.index}
                    
                    if st.session_state.predictor:
                        try:
                            prediction = st.session_state.predictor.predict_next(features_dict)
                        except:
                            prediction = {'direction': 'HOLD', 'confidence': 0.5}
                    else:
                        prediction = {'direction': 'HOLD', 'confidence': 0.5}
                    
                    # Generate signal
                    if st.session_state.trading_strategy:
                        try:
                            trading_signal = st.session_state.trading_strategy.generate_trading_signal(
                                df_features, prediction, current_price
                            )
                            formatted_signal = st.session_state.trading_strategy.format_signal_for_display(trading_signal)
                            action = trading_signal['action']
                        except:
                            action = 'HOLD'
                            formatted_signal = {'action': 'HOLD', 'message': 'Analyzing...'}
                    else:
                        action = 'HOLD'
                        formatted_signal = {'action': 'HOLD', 'message': 'Ready to trade'}

                    # DISPLAY TRADING SIGNAL
                 
                    
                    coin_symbol = symbol.split('/')[0]
                    
                    st.markdown(f"### 🎯 {coin_symbol} - {tf_config['name']} Signal")
                    st.markdown(f'<span class="rl-badge">🧠 Model: {st.session_state.model_type.upper()}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="timeframe-badge">Timeframe: {st.session_state.current_timeframe}</span>', unsafe_allow_html=True)
                    
                    # Display main signal
                    if action == 'BUY':
                        st.markdown(f'<div class="buy-signal">🟢 BUY SIGNAL<br><span style="font-size:1.2rem;">Strong buying opportunity</span></div>', unsafe_allow_html=True)
                    elif action == 'SELL':
                        st.markdown(f'<div class="sell-signal">🔴 SELL SIGNAL<br><span style="font-size:1.2rem;">Strong selling pressure</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="hold-signal">⏸️ HOLD<br><span style="font-size:1.2rem;">Waiting for better opportunity</span></div>', unsafe_allow_html=True)
                    
                    # Display signal details
                    if action in ['BUY', 'SELL'] and st.session_state.trading_strategy:
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("📍 Entry", f"${current_price:,.2f}")
                        with cols[1]:
                            st.metric("🎯 Target", formatted_signal.get('take_profit', 'N/A'))
                        with cols[2]:
                            st.metric("🛑 Stop", formatted_signal.get('stop_loss', 'N/A'))
                        with cols[3]:
                            st.metric("⏱️ Hold", formatted_signal.get('hold_time', 'N/A'))
                    
                    st.markdown("---")
                    
                    # CANDLESTICK CHART
                    # 
                    
                    st.markdown(f"### 📈 {symbol} - {tf_config['name']} Chart")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f'{symbol} Price ({st.session_state.current_timeframe})', 'RSI')
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
                    
                    
                    # LIVE PRICE UPDATE SECTION - BELOW CANDLESTICK CHART
                    
                    st.markdown(f"### 📊 Live Price Update - {symbol}")
                    st.markdown(f'<div class="price-update">💰 Current Price: <strong>${current_price:,.2f}</strong> | Last Updated: {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
                    
                    # Price change
                    if len(st.session_state.price_history) > 1:
                        prev_price = st.session_state.price_history[-2]
                        price_change = current_price - prev_price
                        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                        
                        if price_change >= 0:
                            st.markdown(f'<div class="price-update">📈 Price Change: <strong style="color: #00ff00;">+${price_change:,.2f} (+{price_change_pct:.2f}%)</strong></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="price-update">📉 Price Change: <strong style="color: #ff6b6b;">${price_change:,.2f} ({price_change_pct:.2f}%)</strong></div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    # REAL-TIME PRICE CHART - BELOW EVERYTHING ELSE
            
                    # PRICE TABLE (Instead of chart!)

                    st.markdown("---")
                    st.markdown("### 📊 Price Updates Table (Last 30 Updates)")

                    # Add new price to table data - SIMPLIFIED
                    new_entry = {
                        "⏱️ Time": datetime.now().strftime("%H:%M:%S"),
                        "💰 Price (USD)": f"${current_price:.2f}",
                        "📈 Direction": "UP ⬆️" if (st.session_state.previous_price is not None and current_price > st.session_state.previous_price) else ("DOWN ⬇️" if (st.session_state.previous_price is not None and current_price < st.session_state.previous_price) else "—"),
                        "📊 Change": f"+${current_price - st.session_state.previous_price:.4f}" if (st.session_state.previous_price is not None and current_price > st.session_state.previous_price) else (f"-${abs(current_price - st.session_state.previous_price):.4f}" if (st.session_state.previous_price is not None and current_price < st.session_state.previous_price) else "—")
                    }

                    st.session_state.price_table_data.append(new_entry)

                    # Keep only last 30 entries
                    if len(st.session_state.price_table_data) > 30:
                        st.session_state.price_table_data = st.session_state.price_table_data[-30:]

                    # Display table
                    if st.session_state.price_table_data:
                        # Create DataFrame
                        df_table = pd.DataFrame(st.session_state.price_table_data)
                        
                        # Display table
                        st.dataframe(
                            df_table,
                            use_container_width=True,
                            height=600,
                            hide_index=True
                        )
                        
                        # Extract prices safely for summary stats
                        prices_list = []
                        for entry in st.session_state.price_table_data:
                            try:
                                # Remove $ and convert to float
                                price_str = entry["💰 Price (USD)"].replace("$", "").strip()
                                price_float = float(price_str)
                                prices_list.append(price_float)
                            except (ValueError, AttributeError):
                                # Skip if can't convert
                                continue
                        
                        # Summary stats - ONLY show if we have valid prices
                        if prices_list:
                            st.markdown("---")
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            with summary_col1:
                                st.metric("📍 Latest Price", f"${current_price:.2f}")
                            
                            with summary_col2:
                                st.metric("📈 Highest", f"${max(prices_list):.2f}")
                            
                            with summary_col3:
                                st.metric("📉 Lowest", f"${min(prices_list):.2f}")
                            
                            with summary_col4:
                                avg_price = np.mean(prices_list)
                                st.metric("📊 Average", f"${avg_price:.2f}")
                        else:
                            st.info("⏳ Collecting price data... (need at least 2 updates)")
                    else:
                        st.info("⏳ Waiting for first price update...")

                    # Update previous price for next iteration
                    st.session_state.previous_price = current_price
                                    
                    
                    # METRICS
                    
                    st.markdown("### 📊 Market Metrics")
                    cols = st.columns(5)
                    with cols[0]:
                        st.metric("💵 Price", f"${current_price:,.2f}")
                    with cols[1]:
                        st.metric("📈 RSI", f"{latest_row.get('RSI', 50):.1f}" if 'RSI' in latest_row else "N/A")
                    with cols[2]:
                        st.metric("🎯 Prediction", prediction.get('direction', 'N/A'))
                    with cols[3]:
                        st.metric("💪 Confidence", f"{prediction.get('confidence', 0):.1%}")
                    with cols[4]:
                        st.metric("📊 Timeframe", st.session_state.current_timeframe)
                    
                    # RL Model Stats
                    st.markdown("### 🧠 RL Model Stats")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("🤖 Model", st.session_state.model_type.upper())
                    with cols[1]:
                        st.metric("📊 Episodes", prediction.get('episodes', 'N/A'))
                    with cols[2]:
                        st.metric("🎯 Reward", f"{prediction.get('reward', 0):.2f}")
                    with cols[3]:
                        st.metric("📈 Q-Value", f"{prediction.get('q_value', 0):.4f}")
                    
                    st.caption(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')} | 🔄 Updates Every {update_interval}s | 🧠 RL: {st.session_state.model_type.upper()}")
    
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
    
    st.session_state.refresh_count += 1
    time.sleep(update_interval)
    st.rerun()# app_trading.py - FULLY FIXED VERSION
# Reinforcement Learning Crypto Trading App
# Real-time price chart updates every 30 seconds BELOW the candlestick chart
# No session state errors!

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

# ============================================================================
# CONDITIONAL IMPORTS - Handle missing modules gracefully
# ============================================================================

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
# PAGE CONFIG

st.set_page_config(
    page_title="AI Crypto Trader - RL",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BEAUTIFUL CSS STYLING

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
    
    .timeframe-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        margin: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .rl-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        margin: 0.5rem;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
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
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.2);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        color: white;
        margin: 0.5rem 0;
    }
    
    .rl-info-box {
        background: rgba(255, 107, 107, 0.15);
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid rgba(255, 107, 107, 0.4);
        color: #ffcccc;
        margin: 0.5rem 0;
    }
    
    .price-update {
        background: rgba(50, 200, 100, 0.2);
        border-left: 4px solid #00c853;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# CONFIGURATION DICTIONARIES


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
    'BTC': '🥇 Bitcoin - Digital Gold',
    'ETH': '🥈 Ethereum - Smart Contracts',
    'SOL': '⚡ Solana - Fast Blockchain',
    'DOGE': '🐕 Dogecoin - Meme King',
    'XRP': '💸 Ripple - Payments',
    'SHIB': '🐕 Shiba Inu - Meme Token',
    'PEPE': '🐸 Pepe - Viral Meme',
}

RL_MODELS = {
    'dqn': {'name': '🧠 DQN', 'desc': 'Deep Q-Network', 'algo': 'Q-Learning'},
    'ppo': {'name': '🎯 PPO', 'desc': 'Proximal Policy Optimization', 'algo': 'Policy Gradient'},
    'a2c': {'name': '⚡ A2C', 'desc': 'Actor-Critic', 'algo': 'Policy Gradient'},
}

# SESSION STATE INITIALIZATION - CRITICAL!


# Initialize all keys
for key in ['collector', 'predictor', 'trading_strategy', 'data_history', 'trading_signals', 'is_trained']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'trading_signals' else []

# Initialize feature engineering
if 'feature_eng' not in st.session_state:
    if FeatureEngineering:
        st.session_state.feature_eng = FeatureEngineering()
    else:
        st.session_state.feature_eng = None

# Initialize training status
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# Initialize timeframe
if 'current_timeframe' not in st.session_state:
    st.session_state.current_timeframe = '5m'

# Initialize model type
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'dqn'

# PRICE HISTORY INITIALIZATION - VERY IMPORTANT!

if 'price_history' not in st.session_state:
    st.session_state.price_history = []

if 'time_history' not in st.session_state:
    st.session_state.time_history = []

# HEADER


st.markdown('<div class="main-header">🚀 AI CRYPTO TRADER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">⚡ RL-Powered Trading | Live Price Updates Every 30 Seconds</div>', unsafe_allow_html=True)
st.markdown("---")

# SIDEBAR CONFIGURATION


with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    st.markdown("---")
    
    # Exchange & Symbol
    exchange = st.selectbox("🏦 Exchange", ["kraken", "binance"], index=0)
    pairs_by_category = EXCHANGE_PAIRS[exchange]
    category = st.selectbox("📊 Category", list(pairs_by_category.keys()))
    symbols = pairs_by_category[category]
    symbol = st.selectbox("💰 Trading Pair", symbols, index=0)
    
    # Coin info
    coin_name = symbol.split('/')[0]
    if coin_name in COIN_INFO:
        st.markdown(f'<div class="info-box">{COIN_INFO[coin_name]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TIMEFRAME SELECTION
    st.markdown("## ⏰ TIMEFRAME")
    timeframe_options = list(TIMEFRAMES.keys())
    timeframe_names = [f"{TIMEFRAMES[tf]['name']} - {TIMEFRAMES[tf]['desc']}" for tf in timeframe_options]
    
    selected_tf_idx = st.selectbox(
        "📅 Trading Timeframe",
        range(len(timeframe_options)),
        format_func=lambda x: timeframe_names[x],
        index=1
    )
    
    selected_timeframe = timeframe_options[selected_tf_idx]
    tf_config = TIMEFRAMES[selected_timeframe]
    
    st.markdown(f'''
    <div class="info-box">
        <strong>{tf_config['name']}</strong><br>
        📊 Style: {tf_config['desc']}<br>
        🔄 Update: Every {tf_config['update']}s<br>
        📈 Candles: {tf_config['limit']}
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # RL MODEL SELECTION
    st.markdown("## 🧠 RL MODEL")
    rl_model_options = list(RL_MODELS.keys())
    rl_model_names = [f"{RL_MODELS[m]['name']} - {RL_MODELS[m]['desc']}" for m in rl_model_options]
    
    selected_rl_idx = st.selectbox(
        "🤖 RL Algorithm",
        range(len(rl_model_options)),
        format_func=lambda x: rl_model_names[x],
        index=1
    )
    
    selected_rl_model = rl_model_options[selected_rl_idx]
    rl_config = RL_MODELS[selected_rl_model]
    
    st.markdown(f'''
    <div class="rl-info-box">
        <strong>{rl_config['name']}</strong><br>
        📘 Type: {rl_config['desc']}<br>
        🔬 Algorithm: {rl_config['algo']}<br>
        ⚙️ Status: Enabled
    </div>
    ''', unsafe_allow_html=True)
    
    with st.expander("💡 RL Model Guide"):
        st.markdown("""
        **🧠 DQN (Deep Q-Network):**
        - Best for: Q-value based decisions
        - Stability: ⭐⭐⭐⭐☆
        
        **🎯 PPO (Proximal Policy Optimization):**
        - Best for: Most trading scenarios
        - Stability: ⭐⭐⭐⭐⭐ RECOMMENDED ✅
        
        **⚡ A2C (Actor-Critic):**
        - Best for: Fast convergence
        - Stability: ⭐⭐⭐☆☆
        """)
    
    st.markdown("---")
    st.markdown("## 🎯 TRADING SETUP")
    
    risk_reward = st.slider("⚖️ Risk/Reward", 1.0, 3.0, 2.0, 0.5)
    
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
    
    hold_minutes = st.slider("⏱️ Max Hold (min)", 1, 1440, default_hold, 1)
    
    st.markdown("---")
    
    # Initialize button
    if st.button("🚀 INITIALIZE SYSTEM", type="primary"):
        with st.spinner(f"Initializing {symbol} on {selected_timeframe} timeframe..."):
            try:
                if CryptoDataCollector:
                    st.session_state.collector = CryptoDataCollector(exchange, symbol)
                else:
                    st.error("❌ CryptoDataCollector not available")
                
                # Initialize RL model
                if RL_MODEL_AVAILABLE and RLPricePredictor:
                    st.session_state.predictor = RLPricePredictor(
                        model_type=selected_rl_model,
                        symbol=symbol,
                        timeframe=selected_timeframe
                    )
                elif PricePredictor:
                    st.warning("⚠️ RL Model not available. Using standard model.")
                    st.session_state.predictor = PricePredictor('logistic')
                else:
                    st.error("❌ No predictor available")
                
                st.session_state.current_timeframe = selected_timeframe
                st.session_state.model_type = selected_rl_model
                
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
                
                # Fetch data
                if st.session_state.collector:
                    df = st.session_state.collector.fetch_ohlcv(
                        timeframe=selected_timeframe,
                        limit=tf_config['limit']
                    )
                    
                    if df is not None and len(df) > 0:
                        if st.session_state.feature_eng:
                            df_features = st.session_state.feature_eng.compute_all_features(df)
                            feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                        else:
                            df_features = df
                            feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        
                        if st.session_state.predictor:
                            st.session_state.predictor.train_on_historical(df_features, feature_cols)
                        
                        st.session_state.data_history = df_features
                        st.session_state.is_trained = True
                        st.session_state.current_symbol = symbol
                        
                        # RESET price history when initializing
                        st.session_state.price_history = []
                        st.session_state.time_history = []
                        
                        st.success(f"✅ {symbol} Ready with {selected_rl_model.upper()} on {tf_config['name']}!")
                    else:
                        st.error("Failed to fetch data")
                        
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

# MAIN CONTENT

if not st.session_state.is_trained:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        ### 👋 Welcome to RL-Powered Crypto Trader!
        
        🎯 **Get Started:**
        1. Choose your exchange and coin
        2. Select your timeframe (1m, 5m, 15m, 1h, 4h)
        3. Choose RL Algorithm (DQN, PPO, A2C)
        4. Configure parameters
        5. Click 'Initialize System'
        
        💡 **RL Model Tips:**
        - **DQN**: Q-value learning
        - **PPO**: Policy optimization ✅
        - **A2C**: Actor-Critic
        """)
    
    st.markdown("### ⏰ Choose Your Trading Style")
    cols = st.columns(5)
    for idx, (tf, config) in enumerate(TIMEFRAMES.items()):
        with cols[idx]:
            st.markdown(f"**{config['name']}**")
            st.caption(f"{config['desc']}")
            st.caption(f"Update: {config['update']}s")

else:
    tf_config = TIMEFRAMES[st.session_state.current_timeframe]
    update_interval = 30  # Price updates every 30 seconds
    
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    try:
        if st.session_state.collector:
            latest_data = st.session_state.collector.fetch_ohlcv(
                timeframe=st.session_state.current_timeframe,
                limit=tf_config['limit']
            )
            
            if latest_data is not None and len(latest_data) > 0:
                # Process features
                if st.session_state.feature_eng:
                    df_features = st.session_state.feature_eng.compute_all_features(latest_data)
                    feature_cols = st.session_state.feature_eng.get_feature_columns(df_features)
                else:
                    df_features = latest_data
                    feature_cols = [col for col in latest_data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                if len(df_features) > 0:
                    latest_row = df_features.iloc[-1]
                    current_price = latest_row['close']

                    # ADD TO PRICE HISTORY
                  
                    st.session_state.price_history.append(current_price)
                    st.session_state.time_history.append(datetime.now())
                    
                    # Keep only last 100 prices
                    if len(st.session_state.price_history) > 100:
                        st.session_state.price_history = st.session_state.price_history[-100:]
                        st.session_state.time_history = st.session_state.time_history[-100:]
                    
                    # Get prediction
                    features_dict = {col: latest_row[col] for col in feature_cols if col in latest_row.index}
                    
                    if st.session_state.predictor:
                        try:
                            prediction = st.session_state.predictor.predict_next(features_dict)
                        except:
                            prediction = {'direction': 'HOLD', 'confidence': 0.5}
                    else:
                        prediction = {'direction': 'HOLD', 'confidence': 0.5}
                    
                    # Generate signal
                    if st.session_state.trading_strategy:
                        try:
                            trading_signal = st.session_state.trading_strategy.generate_trading_signal(
                                df_features, prediction, current_price
                            )
                            formatted_signal = st.session_state.trading_strategy.format_signal_for_display(trading_signal)
                            action = trading_signal['action']
                        except:
                            action = 'HOLD'
                            formatted_signal = {'action': 'HOLD', 'message': 'Analyzing...'}
                    else:
                        action = 'HOLD'
                        formatted_signal = {'action': 'HOLD', 'message': 'Ready to trade'}

                    # DISPLAY TRADING SIGNAL

                    coin_symbol = symbol.split('/')[0]
                    
                    st.markdown(f"### 🎯 {coin_symbol} - {tf_config['name']} Signal")
                    st.markdown(f'<span class="rl-badge">🧠 Model: {st.session_state.model_type.upper()}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="timeframe-badge">Timeframe: {st.session_state.current_timeframe}</span>', unsafe_allow_html=True)
                    
                    # Display main signal
                    if action == 'BUY':
                        st.markdown(f'<div class="buy-signal">🟢 BUY SIGNAL<br><span style="font-size:1.2rem;">Strong buying opportunity</span></div>', unsafe_allow_html=True)
                    elif action == 'SELL':
                        st.markdown(f'<div class="sell-signal">🔴 SELL SIGNAL<br><span style="font-size:1.2rem;">Strong selling pressure</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="hold-signal">⏸️ HOLD<br><span style="font-size:1.2rem;">Waiting for better opportunity</span></div>', unsafe_allow_html=True)
                    
                    # Display signal details
                    if action in ['BUY', 'SELL'] and st.session_state.trading_strategy:
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("📍 Entry", f"${current_price:,.2f}")
                        with cols[1]:
                            st.metric("🎯 Target", formatted_signal.get('take_profit', 'N/A'))
                        with cols[2]:
                            st.metric("🛑 Stop", formatted_signal.get('stop_loss', 'N/A'))
                        with cols[3]:
                            st.metric("⏱️ Hold", formatted_signal.get('hold_time', 'N/A'))
                    
                    st.markdown("---")

                    # CANDLESTICK CHART

                    
                    st.markdown(f"### 📈 {symbol} - {tf_config['name']} Chart")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f'{symbol} Price ({st.session_state.current_timeframe})', 'RSI')
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
                    

                    # LIVE PRICE UPDATE SECTION - BELOW CANDLESTICK CHART
                    
                    st.markdown(f"### 📊 Live Price Update - {symbol}")
                    st.markdown(f'<div class="price-update">💰 Current Price: <strong>${current_price:,.2f}</strong> | Last Updated: {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
                    
                    # Price change
                    if len(st.session_state.price_history) > 1:
                        prev_price = st.session_state.price_history[-2]
                        price_change = current_price - prev_price
                        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                        
                        if price_change >= 0:
                            st.markdown(f'<div class="price-update">📈 Price Change: <strong style="color: #00ff00;">+${price_change:,.2f} (+{price_change_pct:.2f}%)</strong></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="price-update">📉 Price Change: <strong style="color: #ff6b6b;">${price_change:,.2f} ({price_change_pct:.2f}%)</strong></div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    

                    # REAL-TIME PRICE CHART - BELOW EVERYTHING ELSE
                    
                    st.markdown("### 📈 Price Chart (Real-time Updates Every 30 Seconds)")
                    
                    if len(st.session_state.price_history) > 0:
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(
                            x=st.session_state.time_history,
                            y=st.session_state.price_history,
                            mode='lines+markers',
                            name='Price',
                            line=dict(color='#667eea', width=3),
                            marker=dict(size=8),
                            fill='tozeroy',
                            fillcolor='rgba(102, 126, 234, 0.2)'
                        ))
                        
                        fig_price.update_layout(
                            title=f'{symbol} Live Price - Last {len(st.session_state.price_history)} Updates',
                            xaxis_title='Time',
                            yaxis_title='Price (USD)',
                            height=400,
                            hovermode='x unified',
                            template='plotly_dark',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=50, r=50, t=50, b=50)
                        )
                        
                        st.plotly_chart(fig_price, use_container_width=True)
                    else:
                        st.info("Price history will appear as updates come in...")
                    
                    st.markdown("---")
    
                    # METRICS

                    st.markdown("### 📊 Market Metrics")
                    cols = st.columns(5)
                    with cols[0]:
                        st.metric("💵 Price", f"${current_price:,.2f}")
                    with cols[1]:
                        st.metric("📈 RSI", f"{latest_row.get('RSI', 50):.1f}" if 'RSI' in latest_row else "N/A")
                    with cols[2]:
                        st.metric("🎯 Prediction", prediction.get('direction', 'N/A'))
                    with cols[3]:
                        st.metric("💪 Confidence", f"{prediction.get('confidence', 0):.1%}")
                    with cols[4]:
                        st.metric("📊 Timeframe", st.session_state.current_timeframe)
                    
                    # RL Model Stats
                    st.markdown("### 🧠 RL Model Stats")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("🤖 Model", st.session_state.model_type.upper())
                    with cols[1]:
                        st.metric("📊 Episodes", prediction.get('episodes', 'N/A'))
                    with cols[2]:
                        st.metric("🎯 Reward", f"{prediction.get('reward', 0):.2f}")
                    with cols[3]:
                        st.metric("📈 Q-Value", f"{prediction.get('q_value', 0):.4f}")
                    
                    st.caption(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')} | 🔄 Updates Every {update_interval}s | 🧠 RL: {st.session_state.model_type.upper()}")
    
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
    
    st.session_state.refresh_count += 1
    time.sleep(update_interval)
    st.rerun()
