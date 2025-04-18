#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Trainer
------------
This module implements training and evaluation for the entry timing model.
"""

import logging
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve
)
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

from models.entry_timing.model import EntryTimingModel
from models.entry_timing.features import FeatureEngineer
from utils.logging import setup_logger
from core.trainer import BaseTrainer
from mlflow.tracking import get_tracker

# Setup logging
logger = setup_logger('model_trainer', category='models')


class ModelTrainer(BaseTrainer):
    """
    Trainer for the entry timing model, handling training, evaluation,
    and hyperparameter tuning.
    """

    def __init__(self, config_path: str, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_engineer = None
        self.results_dir = self.config.get('results_dir', 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize BaseTrainer with parameters for MLflow tracking
        super().__init__(
            params=self.config,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name or "entry_timing"
        )
        
        logger.info("Model Trainer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def prepare_data(self, price_data: pd.DataFrame, orderbook_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare data for model training.
        
        Args:
            price_data: DataFrame with OHLCV data
            orderbook_data: Optional DataFrame with order book data
            
        Returns:
            DataFrame with features and targets
        """
        logger.info("Preparing data for model training")
        
        # Initialize feature engineer if not already
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer(self.config.get('features', {}))
        
        # Create features
        features_df = self.feature_engineer.create_features(price_data, orderbook_data)
        
        # Calculate target variable (entry points)
        return self._calculate_targets(features_df)

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

    def _calculate_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate target variables for the entry timing model.
        
        Args:
            data: DataFrame with price and feature data
            
        Returns:
            DataFrame with target variables
        """
        logger.info("Calculating target variables")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Get target parameters from config
        prediction_horizon = self.config.get('prediction_horizon', 20)  # minutes
        profit_threshold = self.config.get('profit_threshold', 0.003)  # 0.3% min profit
        entry_window = self.config.get('entry_window', 5)  # minutes to enter
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        
        result_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Sort by timestamp to ensure correct calculations
            symbol_df = symbol_df.sort_values('timestamp')
            
            # Calculate forward return
            symbol_df['forward_return'] = symbol_df['close'].shift(-prediction_horizon) / symbol_df['close'] - 1
            
            # Create binary target (1 if forward return exceeds threshold, 0 otherwise)
            symbol_df['target'] = (symbol_df['forward_return'] > profit_threshold).astype(int)
            
            # For more advanced targets, we can also add:
            
            # 1. Risk-adjusted target (profit threshold / atr_ratio)
            if 'atr_ratio' in symbol_df.columns:
                # Adjust profit threshold based on volatility
                adjusted_threshold = profit_threshold / (symbol_df['atr_ratio'] / 5)  # normalize by 5% ATR
                symbol_df['risk_adjusted_target'] = (symbol_df['forward_return'] > adjusted_threshold).astype(int)
            
            # 2. Multi-class target (strong buy, buy, neutral, sell)
            symbol_df['multi_target'] = 0  # neutral
            
            # Strong buy if return > 2*threshold
            symbol_df.loc[symbol_df['forward_return'] > profit_threshold * 2, 'multi_target'] = 2
            
            # Buy if return > threshold
            symbol_df.loc[(symbol_df['forward_return'] > profit_threshold) & 
                          (symbol_df['forward_return'] <= profit_threshold * 2), 'multi_target'] = 1
            
            # Sell if return < -threshold
            symbol_df.loc[symbol_df['forward_return'] < -profit_threshold, 'multi_target'] = -1
            
            # 3. Continuous target (scaled forward return)
            symbol_df['continuous_target'] = symbol_df['forward_return'] / profit_threshold
            
            result_dfs.append(symbol_df)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        
        # Drop rows with NaN targets
        initial_rows = len(result)
        result = result.dropna(subset=['target'])
        dropped_rows = initial_rows - len(result)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN targets")
        
        # Log target distribution
        target_dist = result['target'].value_counts(normalize=True)
        logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        return result
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and return raw data required for training.
        
        Returns:
            Raw data for training
        """
        # This method would typically load data from a file or database
        # For now, we'll return None as the actual data loading happens in run_training_pipeline
        logger.info("Abstract load_data method called - implement in subclass or provide data externally")
        return None
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into feature set for model training.
        
        Args:
            data: Raw data
            
        Returns:
            Processed features
        """
        logger.info("Preparing features for model training")
        
        # Initialize feature engineer if not already
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer(self.config.get('features', {}))
        
        # Create features
        return self.feature_engineer.create_features(data)
    
    def generate_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate and return target values/labels from raw data.
        
        Args:
            data: Raw or feature data
            
        Returns:
            Data with target values
        """
        return self._calculate_targets(data)
    
    def train_model(self, features: pd.DataFrame, target: Any) -> Any:
        """
        Train and return the model.
        
        Args:
            features: Processed feature set
            target: Target values (not used here as targets are in features DataFrame)
            
        Returns:
            Trained model object
        """
        logger.info("Training entry timing model")
        
        # Initialize model if not already
        if self.model is None:
            self.model = EntryTimingModel(self.config)
        
        # For this implementation, we expect targets to be in the features DataFrame
        # so we pass the entire DataFrame to the model's train method
        validation_split = self.config.get('validation_split', 0.2)
        metrics = self.model.train(features, validation_split)
        
        # Log metrics to MLflow
        self.tracker.log_metrics(metrics)
        
        # Save model locally
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.results_dir, f"model_{timestamp}")
        self.model.save(save_path)
        
        # Save training metrics locally
        metrics_path = os.path.join(self.results_dir, f"metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return self.model
    
    def evaluate(self, model: Any, features: pd.DataFrame, target: Any) -> Dict[str, float]:
        """
        Evaluate the trained model and return performance metrics.
        
        Args:
            model: Trained model
            features: Feature set used for evaluation
            target: True target values (not used here as targets are in features DataFrame)
            
        Returns:
            Dictionary of metric names and values
        """
        logger.info("Evaluating model on test data")
        
        # Get predictions
        predictions = model.predict(features)
        
        # Merge predictions with test data
        merged = pd.merge(
            predictions, 
            features[['symbol', 'timestamp', 'target']], 
            on=['symbol', 'timestamp'],
            how='inner'
        )
        
        if len(merged) == 0:
            logger.error("No matching data for evaluation")
            return {}
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(merged['target'], merged['entry_signal']),
            'precision': precision_score(merged['target'], merged['entry_signal']),
            'recall': recall_score(merged['target'], merged['entry_signal']),
            'f1': f1_score(merged['target'], merged['entry_signal']),
            'auc': roc_auc_score(merged['target'], merged['entry_confidence'])
        }
        
        # Log confusion matrix visualization
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(merged['target'], merged['entry_signal'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            cm_path = os.path.join(self.results_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            self.tracker.log_artifact(cm_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create confusion matrix visualization: {str(e)}")
        
        return metrics
    
    def save_model(self, model: Any) -> None:
        """
        Persist the trained model to storage.
        
        Args:
            model: Trained model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.results_dir, f"model_{timestamp}")
        model.save(save_path)
        
        # Log model to MLflow
        self.tracker.log_model(model, artifact_path="model")
        logger.info(f"Model saved to {save_path} and logged to MLflow")
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: DataFrame with features and targets
            
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return {}
        
        logger.info("Evaluating model on test data")
        
        # Get predictions
        predictions = self.model.predict(test_data)
        
        # Merge predictions with test data
        merged = pd.merge(
            predictions, 
            test_data[['symbol', 'timestamp', 'target']], 
            on=['symbol', 'timestamp'],
            how='inner'
        )
        
        if len(merged) == 0:
            logger.error("No matching data for evaluation")
            return {}
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(merged['target'], merged['entry_signal']),
            'precision': precision_score(merged['target'], merged['entry_signal']),
            'recall': recall_score(merged['target'], merged['entry_signal']),
            'f1': f1_score(merged['target'], merged['entry_signal']),
            'auc': roc_auc_score(merged['target'], merged['entry_confidence'])
        }
        
        # Confusion matrix
        cm = confusion_matrix(merged['target'], merged['entry_signal'])
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics (per symbol)
        symbol_metrics = {}
        for symbol in merged['symbol'].unique():
            symbol_data = merged[merged['symbol'] == symbol]
            
            # Skip if not enough data
            if len(symbol_data) < 10 or symbol_data['target'].nunique() < 2:
                continue
            
            symbol_metrics[symbol] = {
                'accuracy': accuracy_score(symbol_data['target'], symbol_data['entry_signal']),
                'precision': precision_score(symbol_data['target'], symbol_data['entry_signal']),
                'recall': recall_score(symbol_data['target'], symbol_data['entry_signal']),
                'f1': f1_score(symbol_data['target'], symbol_data['entry_signal']),
                'auc': roc_auc_score(symbol_data['target'], symbol_data['entry_confidence'])
            }
        
        metrics['symbol_metrics'] = symbol_metrics
        
        # Save metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = os.path.join(self.results_dir, f"evaluation_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return metrics
    
    def visualize_predictions(self, test_data: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Visualize model predictions.
        
        Args:
            test_data: DataFrame with features and targets
            output_path: Optional path to save visualization
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return
        
        logger.info("Visualizing model predictions")
        
        # Get predictions
        predictions = self.model.predict(test_data)
        
        # Merge predictions with test data
        merged = pd.merge(
            predictions, 
            test_data[['symbol', 'timestamp', 'close', 'target']], 
            on=['symbol', 'timestamp'],
            how='inner'
        )
        
        if len(merged) == 0:
            logger.error("No matching data for visualization")
            return
        
        # Select a sample symbol with the most data
        symbol_counts = merged['symbol'].value_counts()
        if len(symbol_counts) == 0:
            logger.error("No symbols found for visualization")
            return
        
        sample_symbol = symbol_counts.index[0]
        sample_data = merged[merged['symbol'] == sample_symbol].copy()
        
        # Sort by timestamp
        sample_data = sample_data.sort_values('timestamp')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(sample_data['timestamp'], sample_data['close'], label='Price', color='blue')
        ax1.set_title(f'Model Predictions for {sample_symbol}')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        
        # Plot entry signals
        for i, row in sample_data.iterrows():
            if row['entry_signal'] == 1:
                ax1.scatter(row['timestamp'], row['close'], color='green', marker='^', s=100)
        
        # Plot actual profitable entries
        for i, row in sample_data.iterrows():
            if row['target'] == 1:
                ax1.scatter(row['timestamp'], row['close'], color='red', marker='x', s=50)
        
        # Plot confidence
        ax2.plot(sample_data['timestamp'], sample_data['entry_confidence'], label='Entry Confidence', color='purple')
        ax2.axhline(y=0.75, color='r', linestyle='--', label='Threshold (0.75)')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        
        # Create custom legend for ax1
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', lw=2),
            Line2D([0], [0], marker='^', color='green', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='x', color='red', markersize=10, linestyle='None')
        ]
        ax1.legend(custom_lines, ['Price', 'Predicted Entry', 'Actual Profitable Entry'], loc='upper left')
        
        plt.tight_layout()
        
        # Save figure if output_path is provided
        if output_path is not None:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            # Save to default location
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = os.path.join(self.results_dir, f"visualization_{timestamp}.png")
            plt.savefig(fig_path)
            logger.info(f"Visualization saved to {fig_path}")
        
        plt.close()
    
    def hyperparameter_tuning(self, data: pd.DataFrame, param_grid: Dict, n_trials: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using random search.
        
        Args:
            data: DataFrame with features and targets
            param_grid: Dictionary of parameter ranges to search
            n_trials: Number of random trials
            
        Returns:
            Dict with best parameters and metrics
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        # Prepare data
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]
        
        best_metrics = {'auc': 0}
        best_params = {}
        
        for i in range(n_trials):
            logger.info(f"Trial {i+1}/{n_trials}")
            
            # Generate random parameters
            params = {}
            for param, values in param_grid.items():
                if isinstance(values, list):
                    params[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], int):
                        params[param] = np.random.randint(values[0], values[1])
                    else:
                        params[param] = np.random.uniform(values[0], values[1])
            
            # Create configuration with these parameters
            config = self.config.copy()
            for k, v in params.items():
                config[k] = v
            
            logger.info(f"Trial {i+1} parameters: {params}")
            
            # Initialize and train model
            model = EntryTimingModel(config)
            model.train(train_data, validation_split=0)
            
            # Evaluate on validation data
            predictions = model.predict(val_data)
            
            # Merge predictions with validation data
            merged = pd.merge(
                predictions, 
                val_data[['symbol', 'timestamp', 'target']], 
                on=['symbol', 'timestamp'],
                how='inner'
            )
            
            if len(merged) == 0:
                logger.warning(f"Trial {i+1}: No matching data for evaluation")
                continue
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(merged['target'], merged['entry_signal']),
                'precision': precision_score(merged['target'], merged['entry_signal']),
                'recall': recall_score(merged['target'], merged['entry_signal']),
                'f1': f1_score(merged['target'], merged['entry_signal']),
                'auc': roc_auc_score(merged['target'], merged['entry_confidence'])
            }
            
            logger.info(f"Trial {i+1} metrics: {metrics}")
            
            # Check if this is the best model so far
            if metrics['auc'] > best_metrics['auc']:
                best_metrics = metrics
                best_params = params
                
                # Save best model so far
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.results_dir, f"best_model_{timestamp}")
                model.save(save_path)
        
        # Save best parameters
        best_config = self.config.copy()
        for k, v in best_params.items():
            best_config[k] = v
        
        config_path = os.path.join(self.results_dir, "best_config.json")
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best metrics: {best_metrics}")
        
        return {'best_params': best_params, 'best_metrics': best_metrics}
    
    def cross_validation(self, data: pd.DataFrame, n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            data: DataFrame with features and targets
            n_splits: Number of splits for time series CV
            
        Returns:
            Dict with cross-validation metrics
        """
        logger.info(f"Performing time series cross-validation with {n_splits} splits")
        
        # Initialize time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Sort data by timestamp
        data = data.sort_values('timestamp')
        
        # Prepare features and target
        X = data.copy()
        y = data['target'].values
        
        # Cross-validation results
        cv_results = []
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {i+1}/{n_splits}")
            
            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Initialize and train model
            model = EntryTimingModel(self.config)
            model.train(train_data, validation_split=0)
            
            # Evaluate on test data
            predictions = model.predict(test_data)
            
            # Merge predictions with test data
            merged = pd.merge(
                predictions, 
                test_data[['symbol', 'timestamp', 'target']], 
                on=['symbol', 'timestamp'],
                how='inner'
            )
            
            if len(merged) == 0:
                logger.warning(f"Fold {i+1}: No matching data for evaluation")
                continue
            
            # Calculate metrics
            metrics = {
                'fold': i+1,
                'accuracy': accuracy_score(merged['target'], merged['entry_signal']),
                'precision': precision_score(merged['target'], merged['entry_signal']),
                'recall': recall_score(merged['target'], merged['entry_signal']),
                'f1': f1_score(merged['target'], merged['entry_signal']),
                'auc': roc_auc_score(merged['target'], merged['entry_confidence'])
            }
            
            logger.info(f"Fold {i+1} metrics: {metrics}")
            cv_results.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            values = [result[metric] for result in cv_results]
            avg_metrics[metric] = sum(values) / len(values)
            avg_metrics[f"{metric}_std"] = np.std(values)
        
        logger.info(f"Average cross-validation metrics: {avg_metrics}")
        
        # Save cross-validation results
        results = {
            'folds': cv_results,
            'average': avg_metrics
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.results_dir, f"cv_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results


def run_training_pipeline(config_path: str, data_path: str, tracking_uri: Optional[str] = None) -> None:
    """
    Run the complete training pipeline.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file
        tracking_uri: Optional MLflow tracking URI
    """
    logger.info("Starting training pipeline")
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path} with {len(data)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Initialize trainer with MLflow tracking
    trainer = ModelTrainer(config_path, tracking_uri=tracking_uri, experiment_name="entry_timing")
    
    # Run the training pipeline using BaseTrainer's run method
    # This will handle MLflow tracking automatically
    trainer.run()
    
    # For backward compatibility, also run the old evaluation and visualization
    val_size = int(len(data) * 0.2)
    val_data = data.iloc[-val_size:]
    trainer.visualize_predictions(val_data)
    
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entry Timing Model Trainer')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--tracking-uri', type=str, help='MLflow tracking URI')
    args = parser.parse_args()
    
    run_training_pipeline(args.config, args.data, args.tracking_uri)
