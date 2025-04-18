#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Peak Detection Model
-----------------
This module provides the model for detecting price peaks in financial time series data.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Optional, Tuple, Union, Any

# Setup logging
logger = logging.getLogger(__name__)

class PeakDetectionModel:
    """
    Model for detecting price peaks in financial time series data.
    
    This class implements a CNN-LSTM hybrid model for detecting price peaks
    to optimize exit points in day trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.sequence_length = config.get('sequence_length', 30)
        self.batch_size = config.get('batch_size', 64)
        self.filters = config.get('filters', [64, 128, 256])
        self.kernel_size = config.get('kernel_size', 3)
        self.lstm_units = config.get('lstm_units', 128)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Prediction parameters
        self.prediction_threshold = config.get('prediction_threshold', 0.5)
        
        # Model
        self.model = None
        self.scaler = StandardScaler()
        
        # Feature columns to use
        self.feature_columns = config.get('feature_columns', None)
        
        logger.info("Peak Detection Model initialized")
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Train the model.
        
        Args:
            data: DataFrame with features and target variable
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training and validation metrics
        """
        logger.info("Training peak detection model")
        
        # Preprocess data
        X, y, feature_names = self._preprocess_data(data)
        
        if X is None or y is None:
            logger.error("Failed to preprocess data")
            return {'train': {}, 'validation': {}}
        
        # Split data into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        # Build model
        self.model = self._build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='max',
                verbose=1
            )
        ]
        
        # Calculate class weights for imbalanced data
        class_weights = {
            0: 1.0,
            1: len(y_train) / (2.0 * np.sum(y_train)) if np.sum(y_train) > 0 else 1.0
        }
        
        logger.info(f"Class weights: {class_weights}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2
        )
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train).flatten()
        y_train_pred_binary = (y_train_pred > self.prediction_threshold).astype(int)
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred_binary),
            'precision': precision_score(y_train, y_train_pred_binary, zero_division=0),
            'recall': recall_score(y_train, y_train_pred_binary, zero_division=0),
            'f1': f1_score(y_train, y_train_pred_binary, zero_division=0),
            'auc': roc_auc_score(y_train, y_train_pred)
        }
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val).flatten()
        y_val_pred_binary = (y_val_pred > self.prediction_threshold).astype(int)
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred_binary),
            'precision': precision_score(y_val, y_val_pred_binary, zero_division=0),
            'recall': recall_score(y_val, y_val_pred_binary, zero_division=0),
            'f1': f1_score(y_val, y_val_pred_binary, zero_division=0),
            'auc': roc_auc_score(y_val, y_val_pred)
        }
        
        logger.info(f"Training metrics: {train_metrics}")
        logger.info(f"Validation metrics: {val_metrics}")
        
        return {
            'train': train_metrics,
            'validation': val_metrics
        }
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the given data.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Generating predictions for {len(data)} samples")
        
        if self.model is None:
            logger.error("Model not trained yet")
            return pd.DataFrame()
        
        # Preprocess data
        X, _, _ = self._preprocess_data(data, is_training=False)
        
        if X is None:
            logger.error("Failed to preprocess data for prediction")
            return pd.DataFrame()
        
        # Generate predictions
        y_pred = self.model.predict(X).flatten()
        
        # Create result DataFrame
        result = data[['symbol', 'timestamp', 'close']].copy()
        result['peak_probability'] = y_pred
        result['peak_detected'] = (y_pred > self.prediction_threshold).astype(int)
        
        logger.info(f"Generated predictions for {len(result)} samples")
        
        return result
    
    def evaluate_prediction(self, data: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate predictions against actual values.
        
        Args:
            data: DataFrame with features and target variable
            predictions: DataFrame with predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating predictions")
        
        # Merge data with predictions
        merged = pd.merge(
            data[['symbol', 'timestamp', 'peak']],
            predictions[['symbol', 'timestamp', 'peak_probability', 'peak_detected']],
            on=['symbol', 'timestamp'],
            how='inner'
        )
        
        if merged.empty:
            logger.warning("No matching data for evaluation")
            return {}
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(merged['peak'], merged['peak_detected']),
            'precision': precision_score(merged['peak'], merged['peak_detected'], zero_division=0),
            'recall': recall_score(merged['peak'], merged['peak_detected'], zero_division=0),
            'f1': f1_score(merged['peak'], merged['peak_detected'], zero_division=0),
            'auc': roc_auc_score(merged['peak'], merged['peak_probability'])
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model
            model_path = os.path.join(path, 'model.h5')
            self.model.save(model_path)
            
            # Save scaler
            import pickle
            scaler_path = os.path.join(path, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save configuration
            import json
            config_path = os.path.join(path, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logger.info(f"Model saved to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(path, 'model.h5')
            self.model = load_model(model_path)
            
            # Load scaler
            import pickle
            scaler_path = os.path.join(path, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load configuration
            import json
            config_path = os.path.join(path, 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Update parameters from config
            self.sequence_length = self.config.get('sequence_length', 30)
            self.batch_size = self.config.get('batch_size', 64)
            self.filters = self.config.get('filters', [64, 128, 256])
            self.kernel_size = self.config.get('kernel_size', 3)
            self.lstm_units = self.config.get('lstm_units', 128)
            self.dropout_rate = self.config.get('dropout_rate', 0.3)
            self.learning_rate = self.config.get('learning_rate', 0.001)
            self.prediction_threshold = self.config.get('prediction_threshold', 0.5)
            self.feature_columns = self.config.get('feature_columns', None)
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the CNN-LSTM hybrid model.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled model
        """
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(
            filters=self.filters[0],
            kernel_size=self.kernel_size,
            activation='relu',
            padding='same',
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        for filter_size in self.filters[1:]:
            model.add(Conv1D(
                filters=filter_size,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same'
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
        
        # LSTM layer
        model.add(LSTM(
            units=self.lstm_units,
            return_sequences=False
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        logger.info(f"Model built with input shape {input_shape}")
        logger.info(model.summary())
        
        return model
    
    def _preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Preprocess data for model training or prediction.
        
        Args:
            data: DataFrame with features and target variable
            is_training: Whether this is for training or prediction
            
        Returns:
            Tuple of (X, y, feature_names) where y is None for prediction
        """
        # Check if data is empty
        if data.empty:
            logger.error("Empty data for preprocessing")
            return None, None, []
        
        # Check if peak column exists for training
        if is_training and 'peak' not in data.columns:
            logger.error("Target column 'peak' not found in data")
            return None, None, []
        
        # Select feature columns
        if self.feature_columns is not None:
            # Use specified feature columns
            feature_cols = [col for col in self.feature_columns if col in data.columns]
            if not feature_cols:
                logger.error("None of the specified feature columns found in data")
                return None, None, []
        else:
            # Use all numeric columns except excluded ones
            excluded_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'peak']
            feature_cols = [col for col in data.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(data[col])]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
        
        # Extract features and target
        X_raw = data[feature_cols].values
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            X_scaled = self.scaler.transform(X_raw)
        
        # Create sequences
        X_sequences = self._create_sequences(X_scaled, self.sequence_length)
        
        if is_training:
            # Extract target
            y_raw = data['peak'].values
            
            # Create target sequences
            y_sequences = self._create_target_sequences(y_raw, self.sequence_length)
            
            return X_sequences, y_sequences, feature_cols
        else:
            return X_sequences, None, feature_cols
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Create sequences for time series data.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            sequence_length: Length of each sequence
            
        Returns:
            3D array of shape (n_sequences, sequence_length, n_features)
        """
        n_samples, n_features = data.shape
        
        # Check if we have enough data
        if n_samples < sequence_length:
            logger.warning(f"Not enough data for sequences: {n_samples} < {sequence_length}")
            return np.array([])
        
        # Create sequences
        sequences = []
        for i in range(n_samples - sequence_length + 1):
            sequences.append(data[i:i+sequence_length])
        
        return np.array(sequences)
    
    def _create_target_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Create target sequences for time series data.
        
        Args:
            data: 1D array of shape (n_samples,)
            sequence_length: Length of each sequence
            
        Returns:
            1D array of shape (n_sequences,)
        """
        n_samples = len(data)
        
        # Check if we have enough data
        if n_samples < sequence_length:
            logger.warning(f"Not enough data for target sequences: {n_samples} < {sequence_length}")
            return np.array([])
        
        # Create target sequences (use the last value of each sequence)
        targets = data[sequence_length-1:]
        
        return targets
