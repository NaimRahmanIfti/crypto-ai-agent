# FILE 4: feature_engineering.py (UPDATED for 'ta' library)
# Location: utils/feature_engineering.py
# This file creates technical indicators and features for the AI model

import pandas as pd
import numpy as np
import ta  # Using 'ta' library instead of pandas_ta

class FeatureEngineering:
    """Compute technical indicators and features for price prediction"""
    
    def __init__(self):
        self.features_list = []
    
    def add_moving_averages(self, df, periods=[7, 14, 21, 50]):
        """
        Add Simple and Exponential Moving Averages
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for moving averages
        """
        for period in periods:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def add_rsi(self, df, period=14):
        """
        Add Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with price data
            period: RSI period (default: 14)
        """
        df['RSI'] = ta.momentum.rsi(df['close'], window=period)
        return df
    
    def add_macd(self, df, fast=12, slow=26, signal=9):
        """
        Add MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        macd_indicator = ta.trend.MACD(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_hist'] = macd_indicator.macd_diff()
        return df
    
    def add_bollinger_bands(self, df, period=20, std=2):
        """
        Add Bollinger Bands
        
        Args:
            df: DataFrame with price data
            period: Period for moving average
            std: Number of standard deviations
        """
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std)
        df['BB_upper'] = bb_indicator.bollinger_hband()
        df['BB_middle'] = bb_indicator.bollinger_mavg()
        df['BB_lower'] = bb_indicator.bollinger_lband()
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        return df
    
    def add_atr(self, df, period=14):
        """
        Add Average True Range (ATR) - volatility indicator
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
        """
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        return df
    
    def add_volume_indicators(self, df):
        """Add volume-based indicators"""
        # Volume Moving Average
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Rate of Change
        df['Volume_ROC'] = df['volume'].pct_change(periods=1) * 100
        
        return df
    
    def add_momentum_indicators(self, df):
        """Add momentum indicators"""
        # Rate of Change
        df['ROC'] = df['close'].pct_change(periods=10) * 100
        
        # Momentum
        df['Momentum'] = df['close'] - df['close'].shift(4)
        
        return df
    
    def add_temporal_features(self, df):
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_price_features(self, df):
        """Add price-based features"""
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = (df['hl_range'] / df['close']) * 100
        
        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def create_target(self, df, periods=1):
        """
        Create target variable for prediction
        
        Args:
            df: DataFrame with price data
            periods: Number of periods ahead to predict (default: 1)
        """
        # Binary target: 1 if price goes up, 0 if down
        df['target_direction'] = (df['close'].shift(-periods) > df['close']).astype(int)
        
        # Regression target: future price
        df['target_price'] = df['close'].shift(-periods)
        
        # Percentage change
        df['target_change_pct'] = ((df['target_price'] - df['close']) / df['close']) * 100
        
        return df
    
    def compute_all_features(self, df):
        """
        Compute all features at once
        
        Args:
            df: DataFrame with OHLCV data and timestamp
        """
        print("Computing features...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Add all features
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_volume_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_price_features(df)
        df = self.add_temporal_features(df)
        df = self.create_target(df)
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        print(f"Features computed. Shape: {df.shape}")
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature column names (excluding target and metadata)"""
        exclude_cols = ['timestamp', 'target_direction', 'target_price', 'target_change_pct']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

if __name__ == "__main__":
    # Test feature engineering
    from data_collector import CryptoDataCollector
    
    collector = CryptoDataCollector(symbol='BTC/USDT')
    df = collector.fetch_ohlcv(limit=200)
    
    if df is not None:
        fe = FeatureEngineering()
        df_features = fe.compute_all_features(df)
        print("\nFeatures created:")
        print(df_features.head())
        print(f"\nFeature columns: {fe.get_feature_columns(df_features)}")