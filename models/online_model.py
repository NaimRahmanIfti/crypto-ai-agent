# FILE 5: online_model.py
# Location: models/online_model.py
# This file contains the online learning AI model for predictions

from river import linear_model, ensemble, preprocessing, metrics
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingAdaptiveTreeClassifier
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class OnlineLearningModel:
    """Online learning model for real-time cryptocurrency price prediction"""
    
    def __init__(self, model_type='logistic'):
        """
        Initialize the online learning model
        
        Args:
            model_type: Type of model ('logistic', 'forest', 'tree')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.scaler = preprocessing.StandardScaler()
        self.accuracy_metric = metrics.Accuracy()
        self.predictions = []
        self.actuals = []
        self.training_count = 0
        
    def _create_model(self, model_type):
        """Create the appropriate model based on type"""
        if model_type == 'logistic':
            # Logistic Regression for classification
            return linear_model.LogisticRegression()
        elif model_type == 'forest':
            # Adaptive Random Forest
            return ensemble.AdaptiveRandomForestClassifier(
                n_models=10,
                seed=42
            )
        elif model_type == 'tree':
            # Hoeffding Adaptive Tree
            return HoeffdingAdaptiveTreeClassifier(
                grace_period=100,
                delta=1e-5,
                seed=42
            )
        else:
            # Default to Logistic Regression
            return linear_model.LogisticRegression()
    
    def preprocess_features(self, features):
        """
        Preprocess features using online scaling
        
        Args:
            features: Dictionary of features
        """
        # Learn and transform in two steps
        self.scaler.learn_one(features)
        return self.scaler.transform_one(features)
    
    def predict(self, features):
        """
        Make a prediction
        
        Args:
            features: Dictionary of features
        
        Returns:
            Dictionary with prediction and probability
        """
        # Preprocess features
        processed_features = self.preprocess_features(features.copy())
        
        # Get prediction
        prediction = self.model.predict_one(processed_features)
        
        # Get probability if available
        try:
            proba = self.model.predict_proba_one(processed_features)
            confidence = proba.get(1, 0.5) if proba else 0.5
        except:
            confidence = 0.5
        
        result = {
            'prediction': int(prediction) if prediction is not None else 0,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        return result
    
    def learn(self, features, target):
        """
        Update the model with new data (online learning)
        
        Args:
            features: Dictionary of features
            target: True target value (0 or 1)
        """
        # Preprocess features
        processed_features = self.preprocess_features(features.copy())
        
        # Learn from this example
        self.model.learn_one(processed_features, target)
        
        # Update metrics
        prediction = self.model.predict_one(processed_features)
        if prediction is not None:
            self.accuracy_metric.update(target, prediction)
        
        self.training_count += 1
    
    def predict_and_learn(self, features, target):
        """
        Make prediction first, then learn from the result
        
        Args:
            features: Dictionary of features
            target: True target value
        
        Returns:
            Dictionary with prediction results
        """
        # Make prediction
        result = self.predict(features)
        
        # Store prediction and actual
        self.predictions.append(result['prediction'])
        self.actuals.append(target)
        
        # Learn from this example
        self.learn(features, target)
        
        return result
    
    def get_accuracy(self):
        """Get current model accuracy"""
        return self.accuracy_metric.get()
    
    def get_recent_performance(self, window=50):
        """
        Calculate accuracy over recent predictions
        
        Args:
            window: Number of recent predictions to consider
        """
        if len(self.predictions) < window:
            window = len(self.predictions)
        
        if window == 0:
            return 0.0
        
        recent_preds = self.predictions[-window:]
        recent_actuals = self.actuals[-window:]
        
        correct = sum([1 for p, a in zip(recent_preds, recent_actuals) if p == a])
        return correct / window
    
    def get_statistics(self):
        """Get model statistics"""
        stats = {
            'training_count': self.training_count,
            'overall_accuracy': self.get_accuracy(),
            'recent_accuracy_50': self.get_recent_performance(50),
            'recent_accuracy_20': self.get_recent_performance(20),
            'total_predictions': len(self.predictions)
        }
        return stats
    
    def save_model(self, filename='model.pkl'):
        """Save the model to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'predictions': self.predictions,
                'actuals': self.actuals,
                'training_count': self.training_count
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='model.pkl'):
        """Load the model from disk"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.predictions = data.get('predictions', [])
                self.actuals = data.get('actuals', [])
                self.training_count = data.get('training_count', 0)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"Model file {filename} not found")

class PricePredictor:
    """High-level predictor that combines model with feature processing"""
    
    def __init__(self, model_type='logistic'):
        self.model = OnlineLearningModel(model_type=model_type)
        self.prediction_history = []
    
    def prepare_features(self, row):
        """
        Convert DataFrame row to feature dictionary
        
        Args:
            row: Single row from DataFrame with features
        """
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'target_direction', 'target_price', 'target_change_pct']
        features = {k: v for k, v in row.items() if k not in exclude_cols and not pd.isna(v)}
        return features
    
    def predict_next(self, features_dict):
        """
        Predict the next price direction
        
        Args:
            features_dict: Dictionary of features
        
        Returns:
            Prediction result with direction and confidence
        """
        result = self.model.predict(features_dict)
        
        # Add interpretation
        result['direction'] = 'UP' if result['prediction'] == 1 else 'DOWN'
        
        # Store in history
        self.prediction_history.append(result)
        
        return result
    
    def train_on_historical(self, df, feature_columns):
        """
        Train model on historical data
        
        Args:
            df: DataFrame with features and targets
            feature_columns: List of feature column names
        """
        print(f"Training on {len(df)} historical samples...")
        
        for idx, row in df.iterrows():
            # Prepare features
            features = {col: row[col] for col in feature_columns if col in row.index}
            target = int(row['target_direction'])
            
            # Learn
            self.model.learn(features, target)
        
        print(f"Training complete. Accuracy: {self.model.get_accuracy():.4f}")
    
    def get_stats(self):
        """Get predictor statistics"""
        return self.model.get_statistics()

if __name__ == "__main__":
    # Test the model
    print("Testing Online Learning Model...")
    
    # Create model
    predictor = PricePredictor(model_type='logistic')
    
    # Simulate some training
    for i in range(100):
        # Random features
        features = {
            'price': np.random.random(),
            'volume': np.random.random(),
            'rsi': np.random.random() * 100
        }
        target = np.random.randint(0, 2)
        
        predictor.model.learn(features, target)
    
    # Make prediction
    test_features = {'price': 0.5, 'volume': 0.3, 'rsi': 45}
    result = predictor.predict_next(test_features)
    
    print(f"\nPrediction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nModel Stats: {predictor.get_stats()}")