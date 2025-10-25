# FILE 2: config.py
# Configuration file for Crypto AI Agent
# You can modify these settings to customize the application

# Exchange Settings
DEFAULT_EXCHANGE = 'binance'  # Options: 'binance', 'kraken'
DEFAULT_SYMBOL = 'BTC/USDT'   # Options: 'BTC/USDT', 'ETH/USDT', 'BNB/USDT', etc.

# Model Settings
DEFAULT_MODEL = 'logistic'    # Options: 'logistic', 'forest', 'tree'

# Data Collection Settings
OHLCV_LIMIT = 200            # Number of historical candles to fetch for training
TIMEFRAME = '1m'             # Timeframe for candles: '1m', '5m', '15m', '1h'
UPDATE_INTERVAL = 30         # Seconds between dashboard updates

# Feature Engineering Settings
MOVING_AVERAGE_PERIODS = [7, 14, 21, 50]  # Periods for MA calculations
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

# Prediction Settings
PREDICTION_HORIZON = 1       # Periods ahead to predict
RECENT_ACCURACY_WINDOW = 50  # Window for recent accuracy calculation

# Dashboard Settings
CHART_HEIGHT = 600           # Height of price chart in pixels
PREDICTION_HISTORY_LIMIT = 10  # Number of recent predictions to display

# Color Settings
UP_COLOR = '#00c853'         # Green for UP predictions
DOWN_COLOR = '#d50000'       # Red for DOWN predictions

# System Settings
AUTO_SAVE_MODEL = True       # Automatically save model after training
MODEL_SAVE_PATH = 'saved_model.pkl'
DATA_SAVE_PATH = 'historical_data.csv'

# API Settings (Optional - for advanced users)
# If you have API keys, you can add them here
BINANCE_API_KEY = None
BINANCE_API_SECRET = None
KRAKEN_API_KEY = None
KRAKEN_API_SECRET = None