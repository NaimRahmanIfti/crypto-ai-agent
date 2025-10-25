# setup_test.py - UPDATED to use Kraken
import sys

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'ccxt',
        'river',
        'ta',
        'websocket',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'websocket':
                __import__('websocket')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n❌ Some packages are missing!")
        print("   Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def test_data_collection():
    """Test if data collection works"""
    print("\nTesting data collection...")
    
    try:
        from data.data_collector import CryptoDataCollector
        
        # Use Kraken instead of Binance
        collector = CryptoDataCollector(exchange_name='kraken', symbol='BTC/USD')
        data = collector.fetch_current_price()
        
        if data:
            print(f"✅ Data collection working - BTC Price: ${data['price']:,.2f}")
            return True
        else:
            print("❌ Failed to fetch data")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_feature_engineering():
    """Test if feature engineering works"""
    print("\nTesting feature engineering...")
    
    try:
        from data.data_collector import CryptoDataCollector
        from utils.feature_engineering import FeatureEngineering
        
        # Use Kraken instead of Binance
        collector = CryptoDataCollector(exchange_name='kraken', symbol='BTC/USD')
        df = collector.fetch_ohlcv(limit=100)
        
        if df is not None and len(df) > 0:
            fe = FeatureEngineering()
            df_features = fe.compute_all_features(df)
            
            print(f"✅ Feature engineering working - {len(df_features)} samples with {len(df_features.columns)} features")
            return True
        else:
            print("❌ Failed to fetch historical data")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_model():
    """Test if AI model works"""
    print("\nTesting AI model...")
    
    try:
        from models.online_model import PricePredictor
        import numpy as np
        
        predictor = PricePredictor(model_type='logistic')
        
        # Test with dummy data
        features = {
            'price': 50000.0,
            'volume': 1000.0,
            'rsi': 55.0
        }
        
        result = predictor.predict_next(features)
        
        print(f"✅ AI model working - Prediction: {result['direction']}, Confidence: {result['confidence']:.2%}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all setup checks"""
    print("=" * 60)
    print("Crypto AI Agent - Setup & Test Script")
    print("=" * 60)
    
    all_passed = True
    
    # Check Python version
    if not check_python_version():
        all_passed = False
    
    # Check dependencies
    if not check_dependencies():
        all_passed = False
        return
    
    # Test components
    if not test_data_collection():
        all_passed = False
    
    if not test_feature_engineering():
        all_passed = False
    
    if not test_ai_model():
        all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! You're ready to go!")
        print("\nTo start the dashboard, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()