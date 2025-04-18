#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Peak Detection Model Trainer
--------------------------
This module handles training and evaluation of the peak detection model
for optimizing exit points in day trading.
"""

import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from core.trainer import BaseTrainer
from mlflow.tracking import get_tracker
from models.peak_detection.model import PeakDetectionModel
from models.peak_detection.features import PeakDetectionFeatures

# Setup logging
from utils.logging import setup_logger
logger = setup_logger(__name__, category='models')


class PeakDetectionTrainer(BaseTrainer):
    """
    Trainer for the peak detection model.
    
    This class handles data loading, feature generation, model training,
    hyperparameter tuning, and evaluation of the CNN-LSTM hybrid model for
    detecting price peaks to optimize exit points in day trading.
    """
    
    def __init__(self, config: Dict[str, Any], tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary with training parameters
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        # Extract parameters for BaseTrainer
        params = {
            "model_type": "peak_detection",
            "model_config": config.get('model_config', {}),
            "feature_config": config.get('feature_config', {}),
            "validation_split": config.get('validation_split', 0.2),
            "test_split": config.get('test_split', 0.1),
            "cv_folds": config.get('cv_folds', 5),
            "random_seed": config.get('random_seed', 42)
        }
        
        # Initialize BaseTrainer
        super().__init__(
            params=params,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name or "peak_detection"
        )
        
        self.config = config
        
        # Model configuration
        self.model_config = config.get('model_config', {})
        
        # Feature configuration
        self.feature_config = config.get('feature_config', {})
        
        # Training parameters
        self.validation_split = config.get('validation_split', 0.2)
        self.test_split = config.get('test_split', 0.1)
        self.cv_folds = config.get('cv_folds', 5)
        self.random_seed = config.get('random_seed', 42)
        
        # Path settings
        self.model_dir = config.get('model_dir', 'models')
        self.plots_dir = os.path.join(self.model_dir, 'plots')
        self.results_dir = self.model_dir  # For compatibility with BaseTrainer
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # Initialize feature generator
        self.feature_generator = PeakDetectionFeatures(self.feature_config)
        
        # Initialize model
        self.model = PeakDetectionModel(self.model_config)
        
        # Track metrics
        self.metrics = {}
        
        # For hyperparameter tuning
        self.best_params = {}
        
        logger.info("Peak Detection Trainer initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load market data for training.
        
        Returns:
            DataFrame with market data
        """
        # This implementation assumes data_path is provided in the config
        data_path = self.config.get('data_path')
        if not data_path:
            logger.error("No data_path provided in configuration")
            return pd.DataFrame()
            
        return self._load_data_from_path(data_path)
    
    def _load_data_from_path(self, data_path: str) -> pd.DataFrame:
        """
        Load market data for training.
        
        Args:
            data_path: Path to market data file
            
        Returns:
            DataFrame with market data
        """
        logger.info(f"Loading market data from {data_path}")
        
        # Load data
        try:
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Ensure timestamp column is datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            else:
                raise ValueError("Market data must contain a 'timestamp' column")
            
            logger.info(f"Loaded {len(data)} rows of market data")
            return data
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return pd.DataFrame()
    
    def prepare_features(self, data: Any) -> pd.DataFrame:
        """
        Prepare data for training by generating features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with generated features
        """
        logger.info("Preparing data for training")
        
        # Check required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Generate features
        feature_data = self.feature_generator.generate_features(data)
        
        if feature_data.empty:
            logger.error("Feature generation failed")
            return pd.DataFrame()
        
        logger.info(f"Generated features, resulting in {len(feature_data)} rows")
        
        # Drop rows with missing values
        feature_data = feature_data.dropna()
        
        logger.info(f"After dropping NaN values, {len(feature_data)} rows remain")
        
        return feature_data
    
    def generate_target(self, data: Any) -> pd.DataFrame:
        """
        Generate target values for peak detection.
        
        Args:
            data: Raw data from load_data
            
        Returns:
            DataFrame with target values
        """
        # In the peak detection case, the target is generated as part of the feature preparation
        # This is a placeholder to satisfy the BaseTrainer interface
        # The actual target generation is handled in the model's _preprocess_data method
        return data

    def select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Select and return a subset of features for model training based on correlation with the target.

        Args:
            features: Processed feature set
            target: Target values

        Returns:
            Selected features
        """
        num_selected_features = self.config.get('num_selected_features', 50) # Get from config
        correlation_threshold = self.config.get('correlation_threshold', 0.1) # Get from config

        logger.info(f"Performing feature selection based on correlation with target, selecting top {num_selected_features} features with correlation > {correlation_threshold}")

        if features.empty or target.empty:
            logger.warning("Features or target is empty, skipping feature selection")
            return features

        try:
            # Calculate correlation between features and target
            # Ensure features and target have the same index
            features_aligned, target_aligned = features.align(target, join='inner', axis=0)

            if features_aligned.empty:
                 logger.warning("No overlapping index between features and target, skipping feature selection")
                 return features

            correlations = features_aligned.corrwith(target_aligned).abs()

            # Filter features based on correlation threshold
            high_correlation_features = correlations[correlations > correlation_threshold].index.tolist()

            # Select top N features from the high correlation features
            # If fewer high correlation features than num_selected_features, select all
            selected_features_names = correlations.loc[high_correlation_features].sort_values(ascending=False).head(num_selected_features).index.tolist()

            logger.info(f"Selected {len(selected_features_names)} features: {selected_features_names}")

            # Return DataFrame with selected features
            return features[selected_features_names]

        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            logger.warning("Returning all features due to selection error")
            return features

    def train_model(self, features: pd.DataFrame, target: pd.DataFrame) -> Any:
        """
        Train the peak detection model.
        
        Args:
            features: DataFrame with prepared feature data
            target: DataFrame with target data (same as features in this case)
            
        Returns:
            Trained model
        """
        # Check if hyperparameter tuning is enabled
        hyperparameter_tuning = self.config.get('hyperparameter_tuning', False)
        
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            self.best_params = self._hyperparameter_tuning(features)
            
            # Update model configuration with best parameters
            self.model_config.update(self.best_params)
            
            # Log best parameters to MLflow
            self.tracker.log_params(self.best_params)
            
            # Reinitialize model with best parameters
            self.model = PeakDetectionModel(self.model_config)
            
            logger.info(f"Best parameters: {self.best_params}")
        
        # Train the model
        metrics = self.model.train(features, validation_split=self.validation_split)
        
        # Store metrics
        self.metrics = metrics
        
        # Log training metrics to MLflow
        self.tracker.log_metrics(metrics.get('train', {}))
        self.tracker.log_metrics(metrics.get('validation', {}))
        
        return self.model
    
    def evaluate(self, model: Any, features: pd.DataFrame, target: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            model: Trained model
            features: Feature data for evaluation
            target: Target data for evaluation (same as features in this case)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Split data for testing (hold out from training/validation)
        _, test_data = self._train_test_split(features)
        
        if test_data.empty:
            logger.warning("No test data available for evaluation")
            return {}
        
        # Generate predictions on test data
        test_predictions = model.predict(test_data)
        
        if test_predictions.empty:
            logger.warning("No predictions generated for test data")
            return {}
        
        # Evaluate predictions
        test_metrics = model.evaluate_prediction(test_data, test_predictions)
        
        # Store test metrics
        self.metrics['test'] = test_metrics
        
        # Generate evaluation plots
        self._generate_evaluation_plots(features)
        
        return test_metrics
    
    def save_model(self, model: Any) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model: Trained model to save
        """
        # Save model
        model_path = os.path.join(self.model_dir, 'peak_detection_model')
        success = model.save(model_path)
        
        if success:
            logger.info(f"Model saved to {model_path}")
        else:
            logger.error("Failed to save model")
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types
            metrics_json = self._convert_to_json_serializable(self.metrics)
            json.dump(metrics_json, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save feature metadata
        feature_metadata = {
            'feature_list': self.feature_generator.get_feature_list(),
            'feature_stats': self._convert_to_json_serializable(self.feature_generator.get_feature_stats())
        }
        
        feature_path = os.path.join(self.model_dir, 'feature_metadata.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_metadata, f, indent=4)
        
        logger.info(f"Feature metadata saved to {feature_path}")
        
        # Save configuration
        config_path = os.path.join(self.model_dir, 'config.json')
        with open(config_path, 'w') as f:
            config_json = {
                'model_config': self.model_config,
                'feature_config': self.feature_config,
                'training_config': {
                    'validation_split': self.validation_split,
                    'test_split': self.test_split,
                    'cv_folds': self.cv_folds,
                    'random_seed': self.random_seed
                }
            }
            
            if self.best_params:
                config_json['best_params'] = self.best_params
            
            json.dump(config_json, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def _train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training/validation and test sets chronologically.
        
        Args:
            data: DataFrame with prepared data
            
        Returns:
            Tuple of (training/validation data, test data)
        """
        # Get unique symbols
        symbols = data['symbol'].unique()
        
        train_val_dfs = []
        test_dfs = []
        
        # Process each symbol separately to maintain chronological order
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Sort by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate split index
            split_idx = int(len(symbol_data) * (1 - self.test_split))
            
            # Split data
            symbol_train_val = symbol_data.iloc[:split_idx]
            symbol_test = symbol_data.iloc[split_idx:]
            
            # Add to result
            train_val_dfs.append(symbol_train_val)
            test_dfs.append(symbol_test)
        
        # Combine results
        train_val_data = pd.concat(train_val_dfs, ignore_index=True)
        test_data = pd.concat(test_dfs, ignore_index=True)
        
        return train_val_data, test_data
    
    def _hyperparameter_tuning(self, data: pd.DataFrame) -> Dict:
        """
        Perform hyperparameter tuning for the peak detection model.
        
        Args:
            data: DataFrame with prepared data
            
        Returns:
            Dict with best hyperparameters
        """
        logger.info("Tuning hyperparameters for peak detection model")
        
        # Define hyperparameter grid
        param_grid = {
            'sequence_length': [20, 30, 40],
            'batch_size': [32, 64, 128],
            'filters': [[32, 64, 128], [64, 128, 256]],
            'lstm_units': [64, 128, 256],
            'dropout_rate': [0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005]
        }
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Track best parameters and scores
        best_score = -float('inf')
        best_params = {}
        
        # Generate parameter combinations with random search
        max_combinations = 8  # Limit number of combinations to test
        np.random.seed(self.random_seed)
        
        # Create parameter sets
        param_combinations = []
        for _ in range(max_combinations):
            params = {}
            for param, values in param_grid.items():
                params[param] = np.random.choice(values)
            param_combinations.append(params)
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Iterate through parameter combinations
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameter set {i+1}/{len(param_combinations)}: {params}")
            
            # Update model configuration
            model_config = self.model_config.copy()
            model_config.update(params)
            
            # Preprocess data once for each parameter set
            # (to avoid repeating preprocessing for each fold)
            model = PeakDetectionModel(model_config)
            try:
                X, y, _ = model._preprocess_data(data)
                
                if X is None or len(X) == 0:
                    logger.warning(f"Failed to preprocess data for parameter set {i+1}")
                    continue
            except Exception as e:
                logger.warning(f"Error preprocessing data for parameter set {i+1}: {str(e)}")
                continue
            
            # Track cross-validation scores
            fold_scores = []
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"Fold {fold+1}/{self.cv_folds}")
                
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create and train model for this fold
                fold_model = PeakDetectionModel(model_config)
                fold_model.model = fold_model._build_model((X_train.shape[1], X_train.shape[2]))
                
                # Use early stopping to prevent overfitting
                early_stopping = EarlyStopping(
                    monitor='val_auc',
                    patience=5,
                    mode='max',
                    restore_best_weights=True,
                    verbose=0
                )
                
                # Calculate class weights for imbalanced data
                class_weights = {
                    0: 1.0,
                    1: len(y_train) / (2.0 * np.sum(y_train)) if np.sum(y_train) > 0 else 1.0
                }
                
                # Train model
                fold_model.model.fit(
                    X_train, y_train,
                    epochs=20,  # Reduced epochs for faster tuning
                    batch_size=model_config['batch_size'],
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    class_weight=class_weights,
                    verbose=0
                )
                
                # Evaluate on validation set
                y_pred_proba = fold_model.model.predict(X_val).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate F1 score
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred_proba)
                
                # Use AUC + F1 as combined score
                score = auc * 0.7 + f1 * 0.3
                
                fold_scores.append(score)
                
                logger.info(f"Fold {fold+1} score: {score:.4f} (AUC: {auc:.4f}, F1: {f1:.4f})")
            
            # Calculate mean score across folds
            if fold_scores:
                mean_score = np.mean(fold_scores)
                logger.info(f"Mean score for parameter set {i+1}: {mean_score:.4f}")
                
                # Update best parameters if better
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    logger.info(f"New best parameters: {best_params}")
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return best_params
    
    def _generate_evaluation_plots(self, data: pd.DataFrame) -> None:
        """
        Generate evaluation plots for model performance visualization.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            None
        """
        logger.info("Generating evaluation plots")
        
        # Preprocess data for visualization
        try:
            # Process a sample of data for visualization
            sample_indices = np.random.choice(len(data), min(1000, len(data)), replace=False)
            sample_data = data.iloc[sample_indices]
            
            # Generate predictions
            predictions = self.model.predict(sample_data)
            
            if predictions.empty:
                logger.warning("No predictions for plot generation")
                return
            
            # 1. Plot peak probability distribution
            self._plot_probability_distribution(predictions)
            
            # 2. Plot example peaks
            self._plot_example_peaks(sample_data, predictions)
            
            # 3. Plot feature importance if available
            self._plot_feature_importance()
            
            # 4. Plot peak detection performance over time
            self._plot_peak_performance_over_time(sample_data, predictions)
            
        except Exception as e:
            logger.error(f"Error generating evaluation plots: {str(e)}")
    
    def _plot_probability_distribution(self, predictions: pd.DataFrame) -> None:
        """
        Plot distribution of peak probabilities.
        
        Args:
            predictions: DataFrame with peak predictions
            
        Returns:
            None
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Separate predictions by class
            peak_mask = predictions['peak_detected'] == 1
            non_peak_mask = predictions['peak_detected'] == 0
            
            # Plot distributions
            sns.histplot(predictions.loc[peak_mask, 'peak_probability'], 
                        color='red', label='Detected Peaks', alpha=0.7)
            sns.histplot(predictions.loc[non_peak_mask, 'peak_probability'], 
                        color='blue', label='Non-Peaks', alpha=0.5)
            
            # Add threshold line
            threshold = self.model.prediction_threshold
            plt.axvline(x=threshold, color='green', linestyle='--', 
                       label=f'Threshold: {threshold:.2f}')
            
            plt.title('Peak Probability Distribution')
            plt.xlabel('Peak Probability')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'probability_distribution.png'), dpi=300)
            plt.close()
            
            logger.info(f"Probability distribution plot saved")
        except Exception as e:
            logger.error(f"Error generating probability distribution plot: {str(e)}")
    
    def _plot_example_peaks(self, data: pd.DataFrame, predictions: pd.DataFrame) -> None:
        """
        Plot examples of detected peaks.
        
        Args:
            data: DataFrame with market data
            predictions: DataFrame with peak predictions
            
        Returns:
            None
        """
        try:
            # Get high-confidence peaks
            high_conf_peaks = predictions[predictions['peak_probability'] > 0.8]
            
            if high_conf_peaks.empty:
                high_conf_peaks = predictions[predictions['peak_detected'] == 1]
            
            if high_conf_peaks.empty:
                logger.warning("No high-confidence peaks for example plot")
                return
            
            # Select up to 4 random peaks
            sample_peaks = high_conf_peaks.sample(min(4, len(high_conf_peaks)))
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, (_, peak) in enumerate(sample_peaks.iterrows()):
                if i >= 4:  # Only plot up to 4 examples
                    break
                
                # Get data around the peak
                symbol = peak['symbol']
                timestamp = peak['timestamp']
                
                # Filter data for the symbol
                symbol_data = data[data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('timestamp')
                
                # Find index of peak in original data
                try:
                    peak_idx = symbol_data[symbol_data['timestamp'] == timestamp].index[0]
                except IndexError:
                    continue
                
                # Get data before and after peak
                lookback = min(30, peak_idx)  # Up to 30 bars before peak
                lookahead = min(15, len(symbol_data) - peak_idx - 1)  # Up to 15 bars after peak
                
                start_idx = peak_idx - lookback
                end_idx = peak_idx + lookahead
                
                if start_idx < 0 or end_idx >= len(symbol_data):
                    continue
                
                plot_data = symbol_data.iloc[start_idx:end_idx+1].copy()
                
                # Plot price
                ax = axes[i]
                ax.plot(plot_data['timestamp'], plot_data['close'], color='black', lw=1.5)
                
                # Highlight peak
                ax.scatter(timestamp, peak['close'], color='red', s=100, marker='^')
                
                # Add RSI if available
                if 'rsi_14' in plot_data.columns:
                    ax_rsi = ax.twinx()
                    ax_rsi.plot(plot_data['timestamp'], plot_data['rsi_14'], color='purple', alpha=0.7, linestyle='--')
                    ax_rsi.set_ylim(0, 100)
                    ax_rsi.axhline(y=70, color='red', alpha=0.3)
                    ax_rsi.axhline(y=30, color='green', alpha=0.3)
                    ax_rsi.set_ylabel('RSI (14)', color='purple')
                
                # Format plot
                ax.set_title(f"{symbol} - Peak at {timestamp.strftime('%Y-%m-%d %H:%M')}")
                ax.set_ylabel('Price')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            # Hide unused subplots
            for j in range(i+1, 4):
                axes[j].axis('off')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'example_peaks.png'), dpi=300)
            plt.close()
            
            logger.info(f"Example peaks plot saved")
        except Exception as e:
            logger.error(f"Error generating example peaks plot: {str(e)}")
    
    def _plot_feature_importance(self) -> None:
        """
        Plot feature importance if model provides it.
        
        Returns:
            None
        """
        try:
            # Get model feature importance if available
            model_layers = self.model.model.layers if self.model.model else []
            feature_weights = None
            
            # Look for the first Dense layer after LSTM for feature importance
            for layer in model_layers:
                if isinstance(layer, tf.keras.layers.Dense) and layer.name == 'dense_1':
                    feature_weights = layer.get_weights()[0]
                    break
            
            if feature_weights is None:
                logger.warning("Feature importance weights not available")
                return
            
            # Calculate feature importance as the mean absolute weight
            importance = np.mean(np.abs(feature_weights), axis=1)
            
            # Get feature names
            feature_names = self.feature_generator.identify_key_features(20)
            
            if len(feature_names) == 0:
                feature_names = [f"Feature {i}" for i in range(len(importance))]
            
            # Match importance to feature names
            # If lengths don't match, use available features
            if len(importance) != len(feature_names):
                feature_names = [f"Feature {i}" for i in range(len(importance))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance_df)
            
            plt.title('Feature Importance')
            plt.xlabel('Importance (Mean Absolute Weight)')
            plt.ylabel('Feature')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            logger.info(f"Feature importance plot saved")
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {str(e)}")
    
    def _plot_peak_performance_over_time(self, data: pd.DataFrame, predictions: pd.DataFrame) -> None:
        """
        Plot peak detection performance over time.
        
        Args:
            data: DataFrame with market data
            predictions: DataFrame with peak predictions
            
        Returns:
            None
        """
        try:
            # Merge data with predictions
            merged_data = pd.merge(
                data,
                predictions[['symbol', 'timestamp', 'peak_probability', 'peak_detected']],
                on=['symbol', 'timestamp'],
                how='inner'
            )
            
            if merged_data.empty:
                logger.warning("No data for peak performance over time plot")
                return
            
            # Group by date and calculate metrics
            merged_data['date'] = merged_data['timestamp'].dt.date
            
            # Calculate metrics by date
            metrics_by_date = []
            
            for date, group in merged_data.groupby('date'):
                peak_count = group['peak_detected'].sum()
                total_count = len(group)
                
                metrics_by_date.append({
                    'date': date,
                    'peak_count': peak_count,
                    'total_count': total_count,
                    'peak_ratio': peak_count / total_count if total_count > 0 else 0,
                    'avg_probability': group['peak_probability'].mean()
                })
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics_by_date)
            
            if metrics_df.empty:
                logger.warning("No data for metrics by date")
                return
            
            # Plot metrics over time
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot peak counts
            ax1.bar(metrics_df['date'], metrics_df['peak_count'], color='red', alpha=0.7)
            ax1.set_ylabel('Peak Count')
            ax1.set_title('Peak Detection Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Plot peak ratio and average probability
            ax2.plot(metrics_df['date'], metrics_df['peak_ratio'], 
                   color='blue', marker='o', linestyle='-', label='Peak Ratio')
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(metrics_df['date'], metrics_df['avg_probability'], 
                       color='green', marker='s', linestyle='--', label='Avg Probability')
            
            ax2.set_ylabel('Peak Ratio', color='blue')
            ax2_twin.set_ylabel('Avg Probability', color='green')
            ax2.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format x-axis
            plt.xticks(rotation=45)
            fig.autofmt_xdate()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'peak_performance_over_time.png'), dpi=300)
            plt.close()
            
            logger.info(f"Peak performance over time plot saved")
        except Exception as e:
            logger.error(f"Error generating peak performance over time plot: {str(e)}")
            
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
