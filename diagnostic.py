#!/usr/bin/env python3
# diagnostic.py - Check your Crypto AI Agent setup

import os
import sys

print("🔍 Crypto AI Agent - Diagnostic Tool")
print("=" * 50)
print()

# Check current directory
print("📁 Current Directory:")
print(f"   {os.getcwd()}")
print()

# Check if files exist
print("📝 Checking Files:")
files_to_check = [
    'app_trading.py',
    'utils/trading_strategy.py',
    'data/data_collector.py',
    'models/online_model.py',
    'utils/feature_engineering.py'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - MISSING!")
print()

# Check trading_strategy.py parameter
print("🔍 Checking trading_strategy.py...")
try:
    with open('utils/trading_strategy.py', 'r') as f:
        content = f.read()
        
    if 'max_hold_time_seconds' in content:
        print("   ✅ Has 'max_hold_time_seconds' - CORRECT!")
    elif 'max_hold_time_minutes' in content:
        print("   ❌ Has 'max_hold_time_minutes' - NEEDS UPDATE!")
        print()
        print("   🔧 FIX:")
        print("   Your trading_strategy.py is the OLD version.")
        print("   You need to replace it with the NEW version.")
        print()
        print("   Option 1: Download the new file")
        print("   Option 2: Edit manually (see instructions below)")
    else:
        print("   ⚠️  Cannot determine version")
except Exception as e:
    print(f"   ❌ Error reading file: {e}")

print()

# Check if we can import
print("🧪 Testing Imports:")
try:
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    from utils.trading_strategy import TradingStrategy
    print("   ✅ Can import TradingStrategy")
    
    # Try to create instance
    try:
        strategy = TradingStrategy(risk_reward_ratio=2.0, max_hold_time_seconds=900)
        print("   ✅ Can create with 'max_hold_time_seconds' - WORKING!")
        print()
        print("🎉 Your setup is CORRECT!")
        print()
        print("🚀 You can run:")
        print("   streamlit run app_trading.py")
    except TypeError as e:
        print(f"   ❌ Cannot create with 'max_hold_time_seconds': {e}")
        print()
        print("   🔧 YOUR FILE NEEDS UPDATE!")
        print()
        print("   MANUAL FIX:")
        print("   1. Open: utils/trading_strategy.py")
        print("   2. Find line ~14:")
        print("      def __init__(self, risk_reward_ratio=2.0, max_hold_time_minutes=60):")
        print()
        print("   3. Change to:")
        print("      def __init__(self, risk_reward_ratio=2.0, max_hold_time_seconds=3600):")
        print()
        print("   4. Find line ~28:")
        print("      self.max_hold_time_minutes = max_hold_time_minutes")
        print()
        print("   5. Change to:")
        print("      self.max_hold_time_seconds = max_hold_time_seconds")
        print()
        print("   6. Save and run again!")
        
except Exception as e:
    print(f"   ❌ Import error: {e}")

print()
print("=" * 50)