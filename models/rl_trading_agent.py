# rl_trading_agent.py
# Location: models/rl_trading_agent.py
# Reinforcement Learning Agent for Crypto Trading

import numpy as np
import pandas as pd
from collections import deque
import random
import pickle
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

class TradingEnvironment:
    """
    Trading environment for RL agent
    """
    
    def __init__(self, data, initial_balance=10000, commission=0.001):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return None
        
        row = self.data.iloc[self.current_step]
        
        # State features
        state = [
            row.get('close', 0),
            row.get('RSI', 50) / 100,  # Normalize to 0-1
            row.get('MACD', 0) / 100,
            row.get('MACD_signal', 0) / 100,
            row.get('BB_upper', 0) / row.get('close', 1),
            row.get('BB_lower', 0) / row.get('close', 1),
            row.get('SMA_7', 0) / row.get('close', 1),
            row.get('SMA_21', 0) / row.get('close', 1),
            row.get('volume', 0) / 1000000,  # Scale down
            float(self.position),  # Current position
            self.balance / self.initial_balance,  # Normalized balance
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        
        Actions:
        0 = HOLD
        1 = BUY
        2 = SELL
        """
        if self.current_step >= len(self.data) - 1:
            return None, 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        info = {}
        
        # Execute action
        if action == 1:  # BUY
            if self.position <= 0:  # Can buy if no position or short
                # Close short position if exists
                if self.position == -1:
                    profit = (self.entry_price - current_price) / self.entry_price
                    profit *= (1 - self.commission * 2)  # Commission both ways
                    self.balance *= (1 + profit)
                    reward += profit * 100  # Scale reward
                    self.trades.append({
                        'type': 'close_short',
                        'price': current_price,
                        'profit': profit
                    })
                
                # Open long position
                self.position = 1
                self.entry_price = current_price
                reward -= 0.1  # Small penalty for trading (commission)
                info['action'] = 'BUY'
        
        elif action == 2:  # SELL
            if self.position >= 0:  # Can sell if no position or long
                # Close long position if exists
                if self.position == 1:
                    profit = (current_price - self.entry_price) / self.entry_price
                    profit *= (1 - self.commission * 2)
                    self.balance *= (1 + profit)
                    reward += profit * 100  # Scale reward
                    self.trades.append({
                        'type': 'close_long',
                        'price': current_price,
                        'profit': profit
                    })
                
                # Open short position
                self.position = -1
                self.entry_price = current_price
                reward -= 0.1  # Small penalty for trading
                info['action'] = 'SELL'
        
        else:  # HOLD
            # Calculate unrealized P&L
            if self.position == 1:  # Long position
                unrealized_profit = (current_price - self.entry_price) / self.entry_price
                reward += unrealized_profit * 10  # Small reward for good holds
            elif self.position == -1:  # Short position
                unrealized_profit = (self.entry_price - current_price) / self.entry_price
                reward += unrealized_profit * 10
            
            info['action'] = 'HOLD'
        
        # Move to next step
        self.current_step += 1
        next_state = self._get_state()
        done = (self.current_step >= len(self.data) - 1) or (self.balance <= 0)
        
        # Final reward at end
        if done:
            final_return = (self.balance - self.initial_balance) / self.initial_balance
            reward += final_return * 100
            info['final_balance'] = self.balance
            info['total_return'] = final_return
        
        return next_state, reward, done, info


class RLTradingAgent:
    """
    Deep Q-Learning Agent for Crypto Trading
    """
    
    def __init__(self, state_size=11, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        
        # Statistics
        self.training_episodes = 0
        self.total_rewards = []
        
        if TF_AVAILABLE:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        else:
            self.model = None
            print("TensorFlow not available. Using random agent.")
    
    def _build_model(self):
        """Build neural network for Q-learning"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action based on state
        
        Returns:
        - 0: HOLD
        - 1: BUY
        - 2: SELL
        """
        if not TF_AVAILABLE or self.model is None:
            # Random agent if TF not available
            return random.randrange(self.action_size)
        
        # Epsilon-greedy policy
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Predict Q-values
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        if not TF_AVAILABLE or self.model is None:
            return
        
        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Predict Q-values for starting state
        target = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next state
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values with Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train model
        self.model.fit(states, target, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes=100, update_target_freq=10):
        """Train agent on environment"""
        print(f"Training RL Agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and state is not None:
                # Choose action
                action = self.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Remember experience
                if next_state is not None:
                    self.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train on batch
                self.replay()
            
            # Update target network periodically
            if episode % update_target_freq == 0:
                self.update_target_model()
            
            # Statistics
            self.total_rewards.append(total_reward)
            self.training_episodes += 1
            
            # Progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.total_rewards[-10:]) if len(self.total_rewards) >= 10 else total_reward
                print(f"Episode {episode}/{episodes} | "
                      f"Total Reward: {total_reward:.2f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {steps}")
        
        print(f"Training complete! Total episodes: {self.training_episodes}")
        return self.total_rewards
    
    def predict_action(self, state):
        """
        Predict best action for given state (no exploration)
        
        Returns:
        - action: 0 (HOLD), 1 (BUY), 2 (SELL)
        - confidence: probability of chosen action
        """
        if not TF_AVAILABLE or self.model is None:
            return random.randrange(self.action_size), 0.33
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]
        
        action = np.argmax(q_values)
        
        # Calculate confidence (softmax of Q-values)
        exp_q = np.exp(q_values - np.max(q_values))
        softmax_q = exp_q / exp_q.sum()
        confidence = softmax_q[action]
        
        return int(action), float(confidence)
    
    def get_q_values(self, state):
        """Get Q-values for all actions"""
        if not TF_AVAILABLE or self.model is None:
            return [0, 0, 0]
        
        state = np.reshape(state, [1, self.state_size])
        return self.model.predict(state, verbose=0)[0].tolist()
    
    def save(self, filepath='rl_agent.pkl'):
        """Save agent to file"""
        if TF_AVAILABLE and self.model:
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        # Save other attributes
        agent_data = {
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes,
            'total_rewards': self.total_rewards,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"Agent data saved to {filepath}")
    
    def load(self, filepath='rl_agent.pkl'):
        """Load agent from file"""
        if TF_AVAILABLE:
            model_path = filepath.replace('.pkl', '_model.h5')
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                self.target_model = keras.models.load_model(model_path)
                print(f"Model loaded from {model_path}")
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            self.epsilon = agent_data.get('epsilon', self.epsilon)
            self.training_episodes = agent_data.get('training_episodes', 0)
            self.total_rewards = agent_data.get('total_rewards', [])
            
            print(f"Agent data loaded from {filepath}")
            print(f"Episodes trained: {self.training_episodes}")
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'training_episodes': self.training_episodes,
            'epsilon': self.epsilon,
            'avg_reward_last_10': np.mean(self.total_rewards[-10:]) if len(self.total_rewards) >= 10 else 0,
            'total_episodes': len(self.total_rewards),
            'tf_available': TF_AVAILABLE,
        }


# Example usage
if __name__ == "__main__":
    print("RL Trading Agent Module")
    print(f"TensorFlow available: {TF_AVAILABLE}")
    
    if TF_AVAILABLE:
        print("\nTo use:")
        print("1. Create agent: agent = RLTradingAgent()")
        print("2. Create environment: env = TradingEnvironment(data)")
        print("3. Train: agent.train(env, episodes=100)")
        print("4. Predict: action, confidence = agent.predict_action(state)")
