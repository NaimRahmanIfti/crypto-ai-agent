# trading_strategy_AGGRESSIVE.py
# Location: utils/trading_strategy.py
# More signals, better entry/exit logic

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradingStrategy:
    """
    AGGRESSIVE trading strategy - More BUY/SELL signals
    """
    
    def __init__(self, risk_reward_ratio=2.0, max_hold_time_seconds=3600):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_hold_time_seconds = max_hold_time_seconds
        self.active_signals = []
        
    def calculate_atr_percent(self, df, period=14):
        """Calculate ATR as percentage of price"""
        if 'ATR' in df.columns and len(df) > 0:
            latest_atr = df['ATR'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            return (latest_atr / latest_price) * 100
        return 0.5
    
    def calculate_volatility(self, df, period=20):
        """Calculate price volatility"""
        if len(df) >= period:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=period).std().iloc[-1] * 100
            return volatility if not pd.isna(volatility) else 1.0
        return 1.0
    
    def detect_trend(self, df):
        """Detect market trend"""
        if len(df) < 20:
            return 'sideways'
        
        latest = df.iloc[-1]
        
        # Use multiple moving averages
        if 'SMA_7' in df.columns and 'SMA_21' in df.columns:
            sma_7 = latest['SMA_7']
            sma_21 = latest['SMA_21']
            price = latest['close']
            
            # Strong uptrend: Price > SMA7 > SMA21
            if price > sma_7 > sma_21:
                return 'uptrend'
            # Strong downtrend: Price < SMA7 < SMA21
            elif price < sma_7 < sma_21:
                return 'downtrend'
        
        return 'sideways'
    
    def calculate_momentum(self, df):
        """Calculate price momentum"""
        if len(df) < 10:
            return 0
        
        # Recent price change
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) * 100
        return price_change
    
    def calculate_signal_strength(self, prediction_confidence, rsi, macd_hist, trend, momentum):
        """
        Calculate signal strength - AGGRESSIVE VERSION
        Lower thresholds for more signals
        """
        strength_score = 0
        
        # Confidence (0-3 points) - LOWERED
        if prediction_confidence > 0.60:
            strength_score += 3
        elif prediction_confidence > 0.52:
            strength_score += 2
        elif prediction_confidence > 0.48:  # Lower threshold
            strength_score += 1
        
        # RSI confirmation (0-2 points)
        if rsi:
            if rsi < 35 or rsi > 65:  # Wider range
                strength_score += 2
            elif rsi < 45 or rsi > 55:
                strength_score += 1
        
        # MACD confirmation (0-2 points)
        if macd_hist:
            if abs(macd_hist) > 30:  # Lower threshold
                strength_score += 2
            elif abs(macd_hist) > 10:
                strength_score += 1
        
        # Momentum (0-2 points) - NEW
        if abs(momentum) > 1.0:
            strength_score += 2
        elif abs(momentum) > 0.5:
            strength_score += 1
        
        # Trend (0-1 point)
        if trend in ['uptrend', 'downtrend']:
            strength_score += 1
        
        # AGGRESSIVE thresholds
        if strength_score >= 6:
            return 'STRONG'
        elif strength_score >= 3:
            return 'MEDIUM'
        else:
            return 'WEAK'
    
    def calculate_stop_loss(self, entry_price, direction, atr_percent):
        """Calculate stop loss - Tighter for more trades"""
        # Tighter stop loss = more trades
        stop_distance_percent = max(atr_percent * 1.2, 0.4)  # Tighter
        
        if direction == 'BUY':
            stop_loss = entry_price * (1 - stop_distance_percent / 100)
        else:
            stop_loss = entry_price * (1 + stop_distance_percent / 100)
        
        return stop_loss, stop_distance_percent
    
    def calculate_take_profit(self, entry_price, stop_loss, direction):
        """Calculate take profit"""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.risk_reward_ratio
        
        if direction == 'BUY':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return take_profit
    
    def estimate_hold_time(self, signal_strength, volatility):
        """Estimate hold time in seconds"""
        if signal_strength == 'STRONG':
            base_seconds = self.max_hold_time_seconds * 0.6
        elif signal_strength == 'MEDIUM':
            base_seconds = self.max_hold_time_seconds * 0.4
        else:
            base_seconds = self.max_hold_time_seconds * 0.25
        
        # Adjust for volatility
        if volatility > 2.0:
            base_seconds = base_seconds * 0.7
        elif volatility < 0.5:
            base_seconds = base_seconds * 1.2
        
        return int(min(base_seconds, self.max_hold_time_seconds))
    
    def format_time(self, seconds):
        """Format seconds to readable format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s" if secs else f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            parts = [f"{hours}h"]
            if minutes:
                parts.append(f"{minutes}m")
            return " ".join(parts)
    
    def determine_action(self, predicted_direction, rsi, trend, signal_strength, 
                        confidence, current_price, bb_upper, bb_lower, momentum, macd_hist):
        """
        AGGRESSIVE ACTION LOGIC - More BUY/SELL signals
        """
        
        # LOWERED minimum confidence
        if confidence < 0.48:  # Was 0.55, now 0.48
            return 'HOLD'
        
        # Allow WEAK signals if other conditions are good
        if signal_strength == 'WEAK' and confidence < 0.55:
            return 'HOLD'
        
        # BUY CONDITIONS - More aggressive
        if predicted_direction == 'UP':
            buy_score = 0
            
            # RSI oversold or neutral
            if rsi < 35:
                buy_score += 3
            elif rsi < 50:
                buy_score += 2
            elif rsi < 60:
                buy_score += 1
            
            # Positive momentum
            if momentum > 0.3:
                buy_score += 2
            elif momentum > 0:
                buy_score += 1
            
            # MACD positive
            if macd_hist and macd_hist > 0:
                buy_score += 1
            
            # Trend support
            if trend == 'uptrend':
                buy_score += 2
            elif trend == 'sideways':
                buy_score += 1
            
            # Confidence boost
            if confidence > 0.60:
                buy_score += 2
            elif confidence > 0.52:
                buy_score += 1
            
            # Bollinger bounce
            if current_price < bb_lower * 1.015:  # Near lower band
                buy_score += 2
            
            # BUY if score is good
            if buy_score >= 4:  # Lower threshold
                return 'BUY'
        
        # SELL CONDITIONS - More aggressive
        elif predicted_direction == 'DOWN':
            sell_score = 0
            
            # RSI overbought or neutral
            if rsi > 65:
                sell_score += 3
            elif rsi > 50:
                sell_score += 2
            elif rsi > 40:
                sell_score += 1
            
            # Negative momentum
            if momentum < -0.3:
                sell_score += 2
            elif momentum < 0:
                sell_score += 1
            
            # MACD negative
            if macd_hist and macd_hist < 0:
                sell_score += 1
            
            # Trend support
            if trend == 'downtrend':
                sell_score += 2
            elif trend == 'sideways':
                sell_score += 1
            
            # Confidence boost
            if confidence > 0.60:
                sell_score += 2
            elif confidence > 0.52:
                sell_score += 1
            
            # Bollinger rejection
            if current_price > bb_upper * 0.985:  # Near upper band
                sell_score += 2
            
            # SELL if score is good
            if sell_score >= 4:  # Lower threshold
                return 'SELL'
        
        return 'HOLD'
    
    def generate_trading_signal(self, df, prediction_result, current_price):
        """Generate trading signal"""
        latest_row = df.iloc[-1]
        
        # Extract indicators
        rsi = latest_row.get('RSI', 50)
        macd_hist = latest_row.get('MACD_hist', 0)
        bb_upper = latest_row.get('BB_upper', current_price * 1.02)
        bb_lower = latest_row.get('BB_lower', current_price * 0.98)
        
        # Market analysis
        trend = self.detect_trend(df)
        atr_percent = self.calculate_atr_percent(df)
        volatility = self.calculate_volatility(df)
        momentum = self.calculate_momentum(df)
        
        # Get prediction
        predicted_direction = prediction_result['direction']
        confidence = prediction_result['confidence']
        
        # Calculate signal strength
        signal_strength = self.calculate_signal_strength(
            confidence, rsi, macd_hist, trend, momentum
        )
        
        # Determine action
        action = self.determine_action(
            predicted_direction, rsi, trend, signal_strength, confidence,
            current_price, bb_upper, bb_lower, momentum, macd_hist
        )
        
        # Calculate trade parameters
        if action in ['BUY', 'SELL']:
            entry_price = current_price
            stop_loss, stop_distance = self.calculate_stop_loss(
                entry_price, action, atr_percent
            )
            take_profit = self.calculate_take_profit(entry_price, stop_loss, action)
            
            if action == 'BUY':
                potential_profit_pct = ((take_profit - entry_price) / entry_price) * 100
                potential_loss_pct = ((entry_price - stop_loss) / entry_price) * 100
            else:
                potential_profit_pct = ((entry_price - take_profit) / entry_price) * 100
                potential_loss_pct = ((stop_loss - entry_price) / entry_price) * 100
            
            estimated_hold_seconds = self.estimate_hold_time(signal_strength, volatility)
            estimated_exit_time = datetime.now() + timedelta(seconds=estimated_hold_seconds)
        else:
            entry_price = None
            stop_loss = None
            take_profit = None
            potential_profit_pct = 0
            potential_loss_pct = 0
            estimated_hold_seconds = 0
            estimated_exit_time = None
        
        signal = {
            'timestamp': datetime.now(),
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'current_price': current_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'potential_profit_pct': potential_profit_pct,
            'potential_loss_pct': potential_loss_pct,
            'risk_reward_ratio': self.risk_reward_ratio if action != 'HOLD' else 0,
            'estimated_hold_seconds': estimated_hold_seconds,
            'estimated_exit_time': estimated_exit_time,
            'trend': trend,
            'volatility': volatility,
            'momentum': momentum,
            'rsi': rsi,
            'macd_histogram': macd_hist,
            'predicted_direction': predicted_direction,
        }
        
        return signal
    
    def format_signal_for_display(self, signal):
        """Format signal for display"""
        if signal['action'] == 'HOLD':
            return {
                'action': '⏸️ HOLD',
                'message': f"Waiting for better setup (Confidence: {signal['confidence']:.1%})",
                'details': f"Market unclear - being patient pays off!"
            }
        
        hold_time_formatted = self.format_time(signal['estimated_hold_seconds'])
        
        action_emoji = '🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL'
        strength_emoji = {'STRONG': '💪', 'MEDIUM': '👍', 'WEAK': '👌'}
        
        formatted = {
            'action': f"{action_emoji} ({strength_emoji[signal['signal_strength']]} {signal['signal_strength']})",
            'entry': f"${signal['entry_price']:,.2f}",
            'stop_loss': f"${signal['stop_loss']:,.2f}",
            'take_profit': f"${signal['take_profit']:,.2f}",
            'potential_profit': f"+{signal['potential_profit_pct']:.2f}%",
            'potential_loss': f"-{signal['potential_loss_pct']:.2f}%",
            'risk_reward': f"1:{signal['risk_reward_ratio']:.1f}",
            'hold_time': hold_time_formatted,
            'hold_seconds': signal['estimated_hold_seconds'],
            'exit_time': signal['estimated_exit_time'].strftime('%H:%M:%S') if signal['estimated_exit_time'] else 'N/A',
            'confidence': f"{signal['confidence']:.1%}",
            'trend': signal['trend'].upper(),
            'volatility': f"{signal['volatility']:.2f}%",
            'momentum': f"{signal['momentum']:+.2f}%"
        }
        
        if signal['action'] == 'BUY':
            message = f"Buy at ${signal['entry_price']:,.2f}, target ${signal['take_profit']:,.2f} in ~{hold_time_formatted}"
        else:
            message = f"Sell at ${signal['entry_price']:,.2f}, target ${signal['take_profit']:,.2f} in ~{hold_time_formatted}"
        
        formatted['message'] = message
        
        return formatted

if __name__ == "__main__":
    print("AGGRESSIVE Trading Strategy - More signals! Ready!")