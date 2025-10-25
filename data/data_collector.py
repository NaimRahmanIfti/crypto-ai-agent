# data_collector.py - UPDATED with Kraken as default
# Location: data/data_collector.py

import ccxt
import pandas as pd
import time
from datetime import datetime
import json

class CryptoDataCollector:
    """Collects real-time cryptocurrency data from exchanges"""
    
    def __init__(self, exchange_name='kraken', symbol='BTC/USD'):  # Changed to Kraken
        """
        Initialize the data collector
        
        Args:
            exchange_name: Name of the exchange (default: kraken)
            symbol: Trading pair symbol (default: BTC/USD)
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.exchange = getattr(ccxt, exchange_name)()
        self.data_buffer = []
        
    def fetch_current_price(self):
        """Fetch the current price and volume"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            data = {
                'timestamp': datetime.now(),
                'price': ticker.get('last', ticker.get('close', 0)),
                'volume': ticker.get('quoteVolume', ticker.get('baseVolume', ticker.get('volume', 0))),
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'change': ticker.get('percentage', 0)
            }
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def fetch_ohlcv(self, timeframe='1m', limit=100):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            timeframe: Timeframe for candles (default: 1m)
            limit: Number of candles to fetch (default: 100)
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return None
    
    def start_collecting(self, interval=60):
        """
        Start collecting data at regular intervals
        
        Args:
            interval: Time between data collection in seconds (default: 60)
        """
        print(f"Starting data collection for {self.symbol} on {self.exchange_name}")
        while True:
            data = self.fetch_current_price()
            if data:
                self.data_buffer.append(data)
                print(f"Collected data at {data['timestamp']}: Price=${data['price']:,.2f}")
            time.sleep(interval)
    
    def get_buffer_as_dataframe(self):
        """Convert the data buffer to a pandas DataFrame"""
        if self.data_buffer:
            return pd.DataFrame(self.data_buffer)
        return pd.DataFrame()
    
    def save_data(self, filename='crypto_data.csv'):
        """Save collected data to CSV file"""
        df = self.get_buffer_as_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")

if __name__ == "__main__":
    # Test the collector with Kraken
    collector = CryptoDataCollector(exchange_name='kraken', symbol='BTC/USD')
    
    # Fetch current price
    print("Current Price:")
    print(collector.fetch_current_price())
    
    # Fetch historical OHLCV
    print("\nHistorical OHLCV:")
    df = collector.fetch_ohlcv(limit=10)
    print(df)