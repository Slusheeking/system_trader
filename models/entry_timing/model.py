#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry Timing Model
-----------------
This module implements an LSTM-Transformer hybrid model for determining
optimal entry points for day trading.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, LSTM, Dropout, BatchNormalization, Input,
    Bidirectional, Attention, LayerNormalization, MultiHeadAttention 
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('entry_timing_model')


class EntryTimingModel:
    """
    Entry Timing Model using LSTM-Transformer architecture to determine
    optimal entry points for day trading.
    """

    def __init__(self, config: Dict):
        """
        Initialize the entry timing model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model hyperparameters
        self.sequence_length = config.get('sequence_length', 30)  # 30 minute lookback
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 100)
        self.lstm_units = config.get('lstm_units', 128)
        self.attention_heads = config.get('attention_heads', 4)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Feature parameters
        self.feature_groups = config.get('feature_groups', {
            'price': True,
            'volume': True,
            'technical': True,
            'orderbook': False  # If order book data is available
        })
        
        # Target parameters
        self.prediction_horizon = config.get('prediction_horizon', 20)  # minutes
        self.profit_threshold = config.get('profit_threshold', 0.003)  # 0.3% min profit
        self.entry_window = config.get('entry_window', 5)  # minutes to enter
        
        # Initialize model
        self.model = None
        self.feature_columns = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
        # Model performance metrics
        self.metrics = {}
        
        # Input/output shapes
        self.input_shape = None
        self.output_shape = None
        
        # For tracking training
        self.history = None
        
        logger.info("Entry Timing Model initialized")
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess data for the sequence model.
        
        Args:
            data: DataFrame with price and feature data
            
        Returns:
            Tuple of (X sequences, y targets, feature columns)
        """
        # Filter required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return None, None, []
        
        # Get unique symbols
        symbols = data['symbol'].unique()
        
        # Select feature columns
        feature_columns = []
        
        # Price features
        if self.feature_groups.get('price', True):
            feature_columns.extend(['open', 'high', 'low', 'close'])
        
        # Volume features
        if self.feature_groups.get('volume', True):
            feature_columns.append('volume')
        
        # Technical indicators (add if present in data)
        if self.feature_groups.get('technical', True):
            tech_cols = [
                'rsi_14', 'macd', 'macd_signal', 'bb_position',
                'stoch_k', 'stoch_d', 'adx_14', 'atr_ratio'
            ]
            for col in tech_cols:
                if col in data.columns:
                    feature_columns.append(col)
        
        # Order book features (if available)
        if self.feature_groups.get('orderbook', False):
            ob_cols = [
                'bid_ask_spread', 'bid_depth', 'ask_depth', 'order_imbalance'
            ]
            for col in ob_cols:
                if col in data.columns:
                    feature_columns.append(col)
        
        logger.info(f"Selected {len(feature_columns)} features for preprocessing")
        
        # Scale the price data (used for target calculation)
        price_data = data[['close']].values
        self.price_scaler.fit(price_data)
        
        # Scale the features
        feature_data = data[feature_columns].values
        self.feature_scaler.fit(feature_data)
        
        # Create sequences and targets
        X_sequences = []
        y_targets = []
        
        # Iterate through each symbol
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Sort by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate forward return
            symbol_data['forward_return'] = symbol_data['close'].shift(-self.prediction_horizon) / symbol_data['close'] - 1
            
            # Create target (1 if forward return exceeds threshold, 0 otherwise)
            symbol_data['target'] = (symbol_data['forward_return'] > self.profit_threshold).astype(int)
            
            # Drop rows with NaN in target
            symbol_data = symbol_data.dropna(subset=['target'])
            
            # Scale features
            scaled_features = self.feature_scaler.transform(symbol_data[feature_columns].values)
            
            # Create sequences
            for i in range(len(symbol_data) - self.sequence_length):
                X_sequences.append(scaled_features[i:i+self.sequence_length])
                y_targets.append(symbol_data['target'].iloc[i+self.sequence_length])
        
        # Convert to numpy arrays
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y, feature_columns
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build the model based on the specified architecture type.

        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)

        Returns:
            Keras Model
        """
        architecture_type = self.config.get('architecture_type', 'lstm_transformer') # Default to current

        logger.info(f"Building model with architecture type: {architecture_type}")

        inputs = Input(shape=input_shape)

        if architecture_type == 'lstm_transformer':
            # Current LSTM-Transformer hybrid architecture
            x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

            x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

            attn_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.lstm_units // self.attention_heads
            )(x, x)

            x = LayerNormalization()(x + attn_output)

            x = LSTM(self.lstm_units // 2)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate/2)(x)

            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate/2)(x)

            outputs = Dense(1, activation='sigmoid')(x)

        elif architecture_type == 'lstm':
            # Simple LSTM architecture
            lstm_units = self.config.get('lstm_units', 64)
            dropout_rate = self.config.get('dropout_rate', 0.3)

            x = LSTM(lstm_units, return_sequences=False)(inputs)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            outputs = Dense(1, activation='sigmoid')(x)

        elif architecture_type == 'cnn_lstm':
            # CNN-LSTM architecture (example, needs CNN layers added)
            lstm_units = self.config.get('lstm_units', 64)
            filters = self.config.get('filters', [32, 64])
            kernel_size = self.config.get('kernel_size', 3)
            dropout_rate = self.config.get('dropout_rate', 0.3)

            # Add CNN layers here...
            # Example:
            # x = Conv1D(filters[0], kernel_size, activation='relu')(inputs)
            # x = MaxPooling1D(pool_size=2)(x)
            # x = Conv1D(filters[1], kernel_size, activation='relu')(x)
            # x = MaxPooling1D(pool_size=2)(x)
            # x = TimeDistributed(Flatten())(x) # If using Conv1D before LSTM

            x = LSTM(lstm_units, return_sequences=False)(inputs) # Placeholder, replace with actual CNN-LSTM
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            outputs = Dense(1, activation='sigmoid')(x)

        # Add other architecture types here...

        else:
            raise ValueError(f"Unknown architecture type: {architecture_type}")

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        return model
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train the entry timing model.
        
        Args:
            data: DataFrame with price and feature data
            validation_split: Proportion of data to use for validation
            
        Returns:
            Dict with training metrics
        """
        logger.info("Preprocessing data for model training")
        
        # Preprocess data
        X, y, feature_columns = self._preprocess_data(data)
        if X is None or len(X) == 0:
            logger.error("Failed to preprocess data")
            return {}
        
        self.feature_columns = feature_columns
        self.input_shape = (self.sequence_length, len(feature_columns))
        
        # Split data chronologically
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build model
        logger.info("Building LSTM-Transformer model")
        self.model = self._build_model(self.input_shape)
        
        # Model summary
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=5,
                mode='max',
                min_lr=1e-6
            )
        ]
        
        # Train model
        logger.info("Training model")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating model")
        evaluation = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        self.metrics = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'auc': evaluation[2],
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        logger.info(f"Model evaluation metrics: {self.metrics}")
        
        return self.metrics
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry signals for new data.
        
        Args:
            data: DataFrame with price and feature data
            
        Returns:
            DataFrame with entry signals and confidence scores
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return pd.DataFrame()
        
        # Verify data contains required columns
        required_columns = ['timestamp', 'symbol'] + self.feature_columns
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Get unique symbols
        symbols = data['symbol'].unique()
        
        results = []
        
        # Process each symbol
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Sort by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Scale features
            scaled_features = self.feature_scaler.transform(symbol_data[self.feature_columns].values)
            
            # Create sequences (only predict for the latest data point)
            # We need at least sequence_length data points
            if len(symbol_data) < self.sequence_length:
                logger.warning(f"Not enough data points for {symbol} (need {self.sequence_length}, got {len(symbol_data)})")
                continue
            
            # Create the latest sequence
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, len(self.feature_columns))
            
            # Make prediction
            entry_confidence = self.model.predict(last_sequence)[0][0]
            
            # Create result entry
            result = {
                'symbol': symbol,
                'timestamp': symbol_data['timestamp'].iloc[-1],
                'close': symbol_data['close'].iloc[-1],
                'entry_confidence': entry_confidence,
                'entry_signal': 1 if entry_confidence > 0.75 else 0  # Higher threshold for actual trading
            }
            
            results.append(result)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        return result_df
    
    def save(self, path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Boolean indicating success
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            # Save keras model
            model_path = f"{path}_keras"
            self.model.save(model_path)
            
            # Save other components
            components = {
                'feature_columns': self.feature_columns,
                'price_scaler': self.price_scaler,
                'feature_scaler': self.feature_scaler,
                'metrics': self.metrics,
                'config': self.config,
                'input_shape': self.input_shape
            }
            
            with open(f"{path}_components.pkl", 'wb') as f:
                pickle.dump(components, f)
            
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load keras model
            model_path = f"{path}_keras"
            self.model = load_model(model_path)
            
            # Load other components
            with open(f"{path}_components.pkl", 'rb') as f:
                components = pickle.load(f)
            
            self.feature_columns = components['feature_columns']
            self.price_scaler = components['price_scaler']
            self.feature_scaler = components['feature_scaler']
            self.metrics = components['metrics']
            self.config = components['config']
            self.input_shape = components['input_shape']
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def analyze_sequence(self, data: pd.DataFrame) -> Dict:
        """
        Analyze a price sequence to explain model decision.
        
        Args:
            data: DataFrame with price and feature data for a single symbol
            
        Returns:
            Dict with analysis results
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return {}
        
        # Verify data contains required columns
        required_columns = ['timestamp', 'symbol'] + self.feature_columns
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return {}
        
        # Ensure data is for a single symbol
        if len(data['symbol'].unique()) > 1:
            logger.error("Data contains multiple symbols, expected single symbol")
            return {}
        
        symbol = data['symbol'].iloc[0]
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Scale features
        scaled_features = self.feature_scaler.transform(data[self.feature_columns].values)
        
        # Create sequence
        if len(data) < self.sequence_length:
            logger.error(f"Not enough data points (need {self.sequence_length}, got {len(data)})")
            return {}
        
        # Use the latest sequence for analysis
        sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, len(self.feature_columns))
        
        # Get model prediction
        entry_confidence = self.model.predict(sequence)[0][0]
        
        # Create visualization data
        price_data = data['close'].iloc[-self.sequence_length:].values
        volume_data = data['volume'].iloc[-self.sequence_length:].values
        timestamp_data = data['timestamp'].iloc[-self.sequence_length:].values
        
        # Get feature series
        feature_series = {}
        for i, feature in enumerate(self.feature_columns):
            feature_series[feature] = scaled_features[-self.sequence_length:, i]
        
        # Prepare analysis result
        analysis = {
            'symbol': symbol,
            'entry_confidence': float(entry_confidence),
            'entry_signal': 1 if entry_confidence > 0.75 else 0,
            'price_data': price_data.tolist(),
            'volume_data': volume_data.tolist(),
            'timestamp_data': [str(ts) for ts in timestamp_data],
            'feature_series': {k: v.tolist() for k, v in feature_series.items()}
        }
        
        return analysis
    
    def predict_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Make a prediction using real-time data from Redis.
        
        Args:
            symbol: Symbol to predict for
            
        Returns:
            Dictionary with prediction results
        """
        from data.processors.realtime_data_provider import RealtimeDataProvider
        
        logger.info(f"Making real-time entry prediction for {symbol}")
        
        if self.model is None:
            logger.error("Model not trained yet")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': "Model not trained"
            }
        
        # Get real-time OHLCV data
        df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, limit=self.sequence_length)
        
        if df.empty:
            logger.warning(f"No real-time data available for {symbol}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': "No real-time data available"
            }
        
        # Check if we have enough data
        if len(df) < self.sequence_length:
            logger.warning(f"Not enough real-time data for {symbol} (need {self.sequence_length}, got {len(df)})")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': f"Not enough data (need {self.sequence_length}, got {len(df)})"
            }
        
        # Add required technical indicators if missing
        required_indicators = [col for col in self.feature_columns if col not in df.columns and col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if required_indicators:
            logger.info(f"Calculating missing indicators: {required_indicators}")
            # This would typically call a function to calculate technical indicators
            # For now, we'll just skip these columns
            
        # Get available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) < 3:  # Need at least some basic features
            logger.warning(f"Not enough features available for {symbol}")
            return {
                'symbol': symbol,
                'timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                'error': "Not enough features available"
            }
        
        # Scale features
        try:
            X = df[available_features].values
            X_scaled = self.feature_scaler.transform(X)
            
            # Create sequence
            X_sequence = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, len(available_features))
            
            # Get prediction
            entry_confidence = float(self.model.predict(X_sequence, verbose=0)[0][0])
            entry_signal = entry_confidence > 0.75  # Higher threshold for actual trading
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            
            # Create result
            result = {
                'symbol': symbol,
                'timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                'close': current_price,
                'entry_confidence': entry_confidence,
                'entry_signal': int(entry_signal)
            }
            
            # Add assessment
            if entry_confidence > 0.8:
                result['assessment'] = "Strong entry signal - consider immediate entry"
            elif entry_confidence > 0.6:
                result['assessment'] = "Moderate entry signal - prepare for entry"
            elif entry_confidence > 0.4:
                result['assessment'] = "Possible entry point forming - monitor closely"
            else:
                result['assessment'] = "No significant entry indication"
            
            logger.info(f"Real-time entry prediction for {symbol}: {result['assessment']} (confidence: {entry_confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making real-time entry prediction for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                'error': str(e)
            }
