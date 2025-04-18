#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Market Regime Trainer Module
------------------------------------
This module handles the training and evaluation of the enhanced market regime detection model,
integrating options flow data from Unusual Whales.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

from core.trainer import BaseTrainer
from feature_store.feature_store import FeatureStore
from models.explainability import ModelExplainer # Import ModelExplainer
from mlflow.tracking import get_tracker
from models.market_regime.features import MarketRegimeFeatures
from models.market_regime.model import EnhancedMarketRegimeModel

# Setup logging
from utils.logging import setup_logger
logger = setup_logger(__name__, category='models')


class MarketRegimeTrainer(BaseTrainer):
    """
    Trainer for the Enhanced Market Regime Detection Model.
    
    This class handles data loading, feature generation, model training,
    hyperparameter tuning, and evaluation for the enhanced market regime
    detection model that integrates options flow data.
    """
    
    def __init__(self, config: Dict[str, Any], tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary with training parameters
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        # Initialize BaseTrainer
        super().__init__(
            params=config,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )
        
        self.config = config
        
        # Model configuration
        self.model_config = config.get('model_config', {})
        
        # Feature configuration
        self.feature_config = config.get('feature_config', {})
        
        # Training parameters
        self.test_size = config.get('test_size', 0.2)
        self.cv_folds = config.get('cv_folds', 5)

        # Initialize Feature Store
        self.feature_store = FeatureStore(config.get('feature_store_config'))

        # Path settings
        self.model_dir = config.get('model_dir', 'models')
        self.plots_dir = os.path.join(self.model_dir, 'plots')
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize feature generator
        self.feature_generator = MarketRegimeFeatures(self.feature_config)
        
        # Initialize model
        self.model = EnhancedMarketRegimeModel(self.model_config)
        
        # Track metrics
        self.metrics = {}
        
        # Initialize MLflow tracker
        self.tracker = get_tracker('market_regime', config.get('version'), tracking_uri)
        
        logger.info("Enhanced Market Regime Trainer initialized")
    
    def load_data(self, market_data_path: str, options_data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load market data and options flow data.
        
        Args:
            market_data_path: Path to market data file
            options_data_path: Optional path to options flow data file
            
        Returns:
            DataFrame with combined market and options flow data
        """
        logger.info(f"Loading market data from {market_data_path}")
        
        # Load market data
        try:
            if market_data_path.endswith('.csv'):
                market_data = pd.read_csv(market_data_path)
            elif market_data_path.endswith('.parquet'):
                market_data = pd.read_parquet(market_data_path)
            else:
                raise ValueError(f"Unsupported file format: {market_data_path}")
            
            # Ensure timestamp column is datetime
            if 'timestamp' in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            else:
                raise ValueError("Market data must contain a 'timestamp' column")
            
            logger.info(f"Loaded {len(market_data)} rows of market data")
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return pd.DataFrame()
        
        # Load options flow data if provided
        if options_data_path:
            logger.info(f"Loading options flow data from {options_data_path}")
            
            try:
                if options_data_path.endswith('.csv'):
                    options_data = pd.read_csv(options_data_path)
                elif options_data_path.endswith('.parquet'):
                    options_data = pd.read_parquet(options_data_path)
                else:
                    raise ValueError(f"Unsupported file format: {options_data_path}")
                
                # Ensure timestamp column is datetime
                if 'timestamp' in options_data.columns:
                    options_data['timestamp'] = pd.to_datetime(options_data['timestamp'])
                else:
                    raise ValueError("Options data must contain a 'timestamp' column")
                
                logger.info(f"Loaded {len(options_data)} rows of options flow data")
                
                # Merge market data and options flow data
                combined_data = self._merge_data(market_data, options_data)
                
                logger.info(f"Combined data has {len(combined_data)} rows")
                return combined_data
            except Exception as e:
                logger.error(f"Error loading options flow data: {str(e)}")
                logger.warning("Proceeding with market data only")
                return market_data
        
        return market_data
    
    def _merge_data(self, market_data: pd.DataFrame, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge market data and options flow data.
        
        Args:
            market_data: DataFrame with market data
            options_data: DataFrame with options flow data
            
        Returns:
            DataFrame with combined data
        """
        # Ensure both DataFrames have timestamp column
        if 'timestamp' not in market_data.columns or 'timestamp' not in options_data.columns:
            raise ValueError("Both market data and options data must have 'timestamp' column")
        
        # Ensure timestamp columns are datetime
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        options_data['timestamp'] = pd.to_datetime(options_data['timestamp'])
        
        # Merge data on timestamp
        merged_data = pd.merge(
            market_data,
            options_data,
            on='timestamp',
            how='outer',
            suffixes=('', '_options')
        )
        
        # Sort by timestamp
        merged_data = merged_data.sort_values('timestamp')
        
        # Rename conflicting columns if any
        rename_cols = {}
        for col in merged_data.columns:
            if col.endswith('_options'):
                base_col = col.replace('_options', '')
                if base_col in merged_data.columns:
                    # Keep both columns but rename for clarity
                    rename_cols[col] = f"options_{base_col}"
        
        if rename_cols:
            merged_data = merged_data.rename(columns=rename_cols)
        
        # Forward fill options data for any missing dates
        options_cols = [col for col in options_data.columns if col != 'timestamp']
        merged_data[options_cols] = merged_data[options_cols].fillna(method='ffill')
        
        return merged_data
    
    def prepare_data(self, data: pd.DataFrame, feature_set_name: str = "market_regime_features", feature_set_version: str = "latest") -> pd.DataFrame:
        """
        Prepare data for training.

        Args:
            data: DataFrame with market and options flow data
            feature_set_name: Name of the feature set to retrieve/store
            feature_set_version: Version of the feature set

        Returns:
            DataFrame with prepared data
        """
        logger.info(f"Preparing data for training using feature set '{feature_set_name}' version '{feature_set_version}'")

        # Try to retrieve features from the feature store
        features_from_store = self.feature_store.get_feature_data(feature_set_name, feature_set_version)

        if features_from_store is not None and not features_from_store.empty:
            logger.info(f"Retrieved {len(features_from_store)} rows of features from feature store.")
            return features_from_store
        else:
            logger.info("Feature set not found in store or is empty. Generating features.")
        
        logger.info("Generating features from raw data")
        
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

    def select_features(self, features: pd.DataFrame, target: Any) -> pd.DataFrame:
        """
        Select and return a subset of features for model training based on feature importance.

        Args:
            features: Processed feature set
            target: Target values (can be None if not applicable)

        Returns:
            Selected features
        """
        num_selected_features = self.config.get('num_selected_features', 50) # Get from config

        logger.info(f"Performing feature selection, selecting top {num_selected_features} features")

        if features.empty:
            logger.warning("Features is empty, skipping feature selection")
            return features

        try:
            # The target for the Market Regime model is typically the regime labels themselves,
            # which are generated within the model or features. We don't have a separate
            # 'target' series here in the same way as a supervised model.
            # However, we can still use feature importance from a preliminary classifier
            # trained to predict regimes based on the features.

            # Identify potential target column (assuming 'regime' is present after feature generation)
            if 'regime' not in features.columns:
                 logger.warning("Target column 'regime' not found in features, skipping feature selection based on importance")
                 return features

            X = features.drop('regime', axis=1)
            y = features['regime']

            # Handle potential non-numeric features in X
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            X_numeric = X[numeric_cols]

            if X_numeric.empty:
                 logger.warning("No numeric features available for importance calculation, skipping feature selection")
                 return features

            # Train a simple classifier to get feature importances
            # Use a subset of data for faster training if needed
            sample_size = min(5000, len(X_numeric))
            X_sample = X_numeric.sample(sample_size, random_state=42)
            y_sample = y.loc[X_sample.index]

            # Initialize a temporary classifier (e.g., a simple RandomForestClassifier)
            from sklearn.ensemble import RandomForestClassifier
            temp_model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train the temporary model
            temp_model.fit(X_sample, y_sample)

            # Get feature importances
            importances = temp_model.feature_importances_
            feature_names = X_sample.columns

            # Create a DataFrame of feature importances
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

            # Select top N features
            selected_features_names = feature_importance_df['feature'].head(num_selected_features).tolist()

            logger.info(f"Selected {len(selected_features_names)} features: {selected_features_names}")

            # Return DataFrame with selected features and the target column
            return features[selected_features_names + ['regime']]

        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            logger.warning("Returning all features due to selection error")
            return features

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into feature set for model training.
        
        Args:
            data: Output from load_data
            
        Returns:
            Processed features
        """
        return self.prepare_data(data)
    
    def generate_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate and return target values/labels from raw data.
        
        Args:
            data: Output from load_data
            
        Returns:
            Target values for training
        """
        # For market regime model, the target is generated during feature preparation
        # and is included in the features DataFrame
        return data
    
    def train_model(self, features: pd.DataFrame, target: pd.DataFrame) -> EnhancedMarketRegimeModel:
        """
        Train and return the model.
        
        Args:
            features: Processed feature set
            target: Target values (same as features for this model)
            
        Returns:
            Trained model object
        """
        logger.info("Training market regime model")
        
        if features.empty:
            logger.error("Empty input data for training")
            return self.model
        
        # Check required columns
        required_columns = ['timestamp', 'close', 'returns']
        if not all(col in features.columns for col in required_columns):
            missing = [col for col in required_columns if col not in features.columns]
            logger.error(f"Missing required columns: {missing}")
            return self.model
        
        # If hyperparameter tuning is enabled, perform tuning
        if self.config.get('hyperparameter_tuning', False):
            logger.info("Performing hyperparameter tuning")
            best_params = self._hyperparameter_tuning(features)
            
            # Update model configuration with best parameters
            self.model_config.update(best_params)
            
            # Reinitialize model with best parameters
            self.model = EnhancedMarketRegimeModel(self.model_config)
            
            logger.info(f"Best parameters: {best_params}")
        
        # Train the model
        self.metrics = self.model.train(features)
        
        return self.model
    
    def evaluate(self, model: EnhancedMarketRegimeModel, features: pd.DataFrame, target: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the trained model and return performance metrics.

        Args:
            model: Trained model
            features: Feature set used for evaluation
            target: True target values (same as features for this model)

        Returns:
            Dictionary of metric names and values
        """
        # For market regime model, evaluation metrics are already calculated during training
        # Extract the metrics we want to log to MLflow
        metrics_to_log = {}

        if 'classifier_metrics' in self.metrics:
            classifier_metrics = self.metrics['classifier_metrics']

            # Extract scalar metrics
            for key in ['cv_accuracy_mean', 'test_accuracy']:
                if key in classifier_metrics:
                    metrics_to_log[key] = classifier_metrics[key]

            # Extract metrics from classification report if available
            if 'classification_report' in classifier_metrics and isinstance(classifier_metrics['classification_report'], dict):
                report = classifier_metrics['classification_report']

                # Extract overall metrics
                for key in ['accuracy', 'macro avg', 'weighted avg']:
                    if key in report:
                        if isinstance(report[key], dict):
                            for metric, value in report[key].items():
                                if isinstance(value, (int, float)):
                                    metrics_to_log[f"{key}_{metric}"] = value
                        elif isinstance(report[key], (int, float)):
                            metrics_to_log[key] = report[key]

        # Generate evaluation plots
        self._generate_evaluation_plots(features)

        # Analyze model explainability
        self._analyze_model_explainability(features)

        return metrics_to_log
    
    def save_model(self, model: EnhancedMarketRegimeModel) -> None:
        """
        Persist the trained model to storage.
        
        Args:
            model: Trained model
        """
        self._save_model_and_metrics()
    
    def train(self, data: pd.DataFrame, hyperparameter_tuning: bool = False) -> Dict:
        """
        Train the enhanced market regime model using the MLflow tracking pipeline.
        
        Args:
            data: DataFrame with prepared data
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dict with training metrics
        """
        logger.info("Starting model training with MLflow tracking")
        
        # Begin tracking
        run_name = f"MarketRegimeTrainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker.start_run(run_name=run_name)
        
        # Log parameters
        self.tracker.log_params({
            **self.model_config,
            **self.feature_config,
            'test_size': self.test_size,
            'cv_folds': self.cv_folds,
            'hyperparameter_tuning': hyperparameter_tuning
        })
        
        try:
            # Prepare data if needed
            if not isinstance(data, pd.DataFrame) or data.empty:
                data = self.load_data()
                
            prepared_data = self.prepare_data(data)
            
            # Train model
            if hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning")
                best_params = self._hyperparameter_tuning(prepared_data)
                
                # Update model configuration with best parameters
                self.model_config.update(best_params)
                
                # Reinitialize model with best parameters
                self.model = EnhancedMarketRegimeModel(self.model_config)
                
                # Log best parameters
                self.tracker.log_params(best_params)
                logger.info(f"Best parameters: {best_params}")
            
            # Train the model
            metrics = self.model.train(prepared_data)
            
            # Store metrics
            self.metrics = metrics
            
            # Log metrics to MLflow
            metrics_to_log = {}
            
            # Extract scalar metrics for logging
            if 'classifier_metrics' in metrics:
                classifier_metrics = metrics['classifier_metrics']
                
                # Extract scalar metrics
                for key in ['cv_accuracy_mean', 'test_accuracy']:
                    if key in classifier_metrics:
                        metrics_to_log[key] = classifier_metrics[key]
                
                # Extract metrics from classification report if available
                if 'classification_report' in classifier_metrics and isinstance(classifier_metrics['classification_report'], dict):
                    report = classifier_metrics['classification_report']
                    
                    # Extract overall metrics
                    for key in ['accuracy', 'macro avg', 'weighted avg']:
                        if key in report:
                            if isinstance(report[key], dict):
                                for metric, value in report[key].items():
                                    if isinstance(value, (int, float)):
                                        metrics_to_log[f"{key}_{metric}"] = value
                            elif isinstance(report[key], (int, float)):
                                metrics_to_log[key] = report[key]
            
            # Log metrics
            self.tracker.log_metrics(metrics_to_log)
            
            # Save model and metrics
            self._save_model_and_metrics()
            
            # Log model to MLflow
            self.tracker.log_model(self.model, artifact_path="market_regime_model")
            
            # Generate evaluation plots
            self._generate_evaluation_plots(prepared_data)
            
            # Log plots as artifacts
            for plot_file in os.listdir(self.plots_dir):
                if plot_file.endswith('.png'):
                    self.tracker.log_artifact(os.path.join(self.plots_dir, plot_file))
            
            return metrics
        finally:
            # Ensure the MLflow run is always closed
            self.tracker.end_run()
    
    def _hyperparameter_tuning(self, data: pd.DataFrame) -> Dict:
        """
        Perform hyperparameter tuning for the enhanced market regime model.
        
        Args:
            data: DataFrame with prepared data
            
        Returns:
            Dict with best hyperparameters
        """
        logger.info("Tuning hyperparameters for enhanced market regime model")
        
        # Define hyperparameter grid
        param_grid = {
            'n_regimes': [3, 4, 5],
            'hmm_n_iter': [50, 100, 200],
            'smooth_window': [3, 5, 7],
            'xgb_n_estimators': [50, 100, 200],
            'xgb_learning_rate': [0.01, 0.05, 0.1],
            'xgb_max_depth': [3, 5, 7]
        }
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Track best parameters and metrics
        best_score = -float('inf')
        best_params = {}
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Limit to maximum combinations
        max_combinations = 20
        if len(param_combinations) > max_combinations:
            logger.info(f"Limiting to {max_combinations} parameter combinations out of {len(param_combinations)}")
            np.random.seed(42)
            param_combinations = np.random.choice(param_combinations, max_combinations, replace=False)
        
        logger.info(f"Evaluating {len(param_combinations)} parameter combinations")
        
        # Evaluate each parameter combination
        for i, params in enumerate(param_combinations):
            logger.info(f"Evaluating parameters {i+1}/{len(param_combinations)}: {params}")
            
            # Initialize model with current parameters
            model = EnhancedMarketRegimeModel(params)
            
            # Track fold scores
            fold_scores = []
            
            # Perform cross-validation
            for train_idx, test_idx in tscv.split(data):
                # Split data
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                try:
                    # Train model on training data
                    model.train(train_data)
                    
                    # Evaluate on test data
                    test_metrics = self._evaluate_model(model, test_data)
                    
                    # Use accuracy as score
                    if 'classifier_metrics' in test_metrics and 'cv_accuracy_mean' in test_metrics['classifier_metrics']:
                        fold_scores.append(test_metrics['classifier_metrics']['cv_accuracy_mean'])
                except Exception as e:
                    logger.error(f"Error during cross-validation: {str(e)}")
                    fold_scores.append(0.0)
            
            # Calculate mean score across folds
            if fold_scores:
                mean_score = np.mean(fold_scores)
                logger.info(f"Mean score: {mean_score:.4f}")
                
                # Update best parameters if better
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    logger.info(f"New best parameters: {best_params}")
            else:
                logger.warning("No valid scores for this parameter combination")
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        Generate all parameter combinations from parameter grid.
        
        Args:
            param_grid: Dictionary of parameter names and values
            
        Returns:
            List of parameter combinations
        """
        param_combinations = [{}]
        
        for param, values in param_grid.items():
            new_combinations = []
            
            for combination in param_combinations:
                for value in values:
                    new_combination = combination.copy()
                    new_combination[param] = value
                    new_combinations.append(new_combination)
            
            param_combinations = new_combinations
        
        return param_combinations
    
    def _evaluate_model(self, model: EnhancedMarketRegimeModel, data: pd.DataFrame) -> Dict:
        """
        Evaluate the enhanced market regime model.
        
        Args:
            model: Enhanced market regime model
            data: DataFrame with test data
            
        Returns:
            Dict with evaluation metrics
        """
        # Get regime characteristics
        regime_stats = model.get_regime_characteristics()
        
        # Get classifier metrics if available
        if hasattr(model, 'metrics') and 'classifier_metrics' in model.metrics:
            classifier_metrics = model.metrics['classifier_metrics']
        else:
            classifier_metrics = {}
        
        # Predict regimes
        predictions = model.predict_regime(data)
        
        # Calculate accuracy if regime labels are available
        if 'regime' in data.columns and not predictions.empty and 'regime' in predictions.columns:
            # Get common timestamps
            common_timestamps = set(data['timestamp']) & set(predictions['timestamp'])
            
            if common_timestamps:
                # Filter data to common timestamps
                data_filtered = data[data['timestamp'].isin(common_timestamps)]
                predictions_filtered = predictions[predictions['timestamp'].isin(common_timestamps)]
                
                # Sort by timestamp
                data_filtered = data_filtered.sort_values('timestamp')
                predictions_filtered = predictions_filtered.sort_values('timestamp')
                
                # Ensure same order
                data_filtered = data_filtered.sort_values('timestamp')
                predictions_filtered = predictions_filtered.sort_values('timestamp')
                
                # Extract true and predicted regimes
                y_true = data_filtered['regime'].values
                y_pred = predictions_filtered['regime'].values
                
                # Calculate accuracy
                accuracy = accuracy_score(y_true, y_pred)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=list(regime_stats.keys()))
                
                # Calculate classification report
                report = classification_report(y_true, y_pred, labels=list(regime_stats.keys()), output_dict=True)
                
                # Add to classifier metrics
                classifier_metrics['test_accuracy'] = accuracy
                classifier_metrics['confusion_matrix'] = cm.tolist()
                classifier_metrics['classification_report'] = report
        
        return {
            'regime_stats': regime_stats,
            'classifier_metrics': classifier_metrics
        }
    
    def _save_model_and_metrics(self) -> None:
        """
        Save the trained model and metrics.
        
        Returns:
            None
        """
        # Save model
        model_path = os.path.join(self.model_dir, 'market_regime_model.pkl')
        self.model.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save feature metadata
        feature_metadata = self.feature_generator.get_feature_metadata()
        feature_metadata_path = os.path.join(self.model_dir, 'feature_metadata.json')
        with open(feature_metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=4)
        
        # Save configuration
        config_path = os.path.join(self.model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': self.model_config,
                'feature_config': self.feature_config,
                'training_config': {
                    'test_size': self.test_size,
                    'cv_folds': self.cv_folds
                }
            }, f, indent=4)
        
        logger.info(f"Model and metrics saved to {self.model_dir}")
    
    def _generate_evaluation_plots(self, data: pd.DataFrame) -> None:
        """
        Generate evaluation plots for the trained model.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            None
        """
        logger.info("Generating evaluation plots")
        
        # Plot market regimes
        if hasattr(self.model, 'plot_regimes'):
            try:
                fig = self.model.plot_regimes(data)
                
                if fig:
                    # Save regime plot
                    regime_plot_path = os.path.join(self.plots_dir, 'market_regimes.png')
                    fig.savefig(regime_plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Regime plot saved to {regime_plot_path}")
            except Exception as e:
                logger.error(f"Error generating regime plot: {str(e)}")
        
        # Plot regime transitions
        if hasattr(self.model, 'transition_matrix') and self.model.transition_matrix is not None:
            try:
                # Create transition matrix plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Get regime names
                regime_names = list(self.model.regime_mapping.values())
                
                # Plot heatmap
                sns.heatmap(
                    self.model.transition_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=regime_names,
                    yticklabels=regime_names,
                    ax=ax
                )
                
                ax.set_title('Regime Transition Probabilities')
                ax.set_xlabel('To Regime')
                ax.set_ylabel('From Regime')
                
                # Save transition matrix plot
                transition_plot_path = os.path.join(self.plots_dir, 'transition_matrix.png')
                fig.savefig(transition_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Transition matrix plot saved to {transition_plot_path}")
            except Exception as e:
                logger.error(f"Error generating transition matrix plot: {str(e)}")
        
        # Plot regime characteristics
        regime_stats = self.model.get_regime_characteristics()
        
        if regime_stats:
            try:
                # Create regime characteristics plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                # Extract data for plots
                regimes = list(regime_stats.keys())
                returns = [stats.get('mean_return', 0) for stats in regime_stats.values()]
                volatility = [stats.get('volatility', 0) for stats in regime_stats.values()]
                durations = [stats.get('avg_duration', 0) for stats in regime_stats.values()]
                
                # Additional options metrics if available
                options_metrics = []
                for metric in ['put_call_ratio', 'unusual_activity_score', 'implied_volatility']:
                    if all(metric in stats for stats in regime_stats.values()):
                        options_metrics.append(metric)
                
                # Plot returns
                axes[0].bar(regimes, returns)
                axes[0].set_title('Mean Return by Regime')
                axes[0].set_xlabel('Regime')
                axes[0].set_ylabel('Mean Return')
                
                # Plot volatility
                axes[1].bar(regimes, volatility)
                axes[1].set_title('Volatility by Regime')
                axes[1].set_xlabel('Regime')
                axes[1].set_ylabel('Volatility')
                
                # Plot duration
                axes[2].bar(regimes, durations)
                axes[2].set_title('Average Duration by Regime')
                axes[2].set_xlabel('Regime')
                axes[2].set_ylabel('Duration (days)')
                
                # Plot options metric if available
                if options_metrics:
                    metric = options_metrics[0]
                    metric_values = [stats.get(metric, 0) for stats in regime_stats.values()]
                    axes[3].bar(regimes, metric_values)
                    axes[3].set_title(f'{metric.replace("_", " ").title()} by Regime')
                    axes[3].set_xlabel('Regime')
                    axes[3].set_ylabel(metric.replace("_", " ").title())
                else:
                    # If no options metrics, hide the last subplot
                    axes[3].set_visible(False)
                
                # Add overall title
                plt.suptitle('Regime Characteristics')
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                
                # Save regime characteristics plot
                regime_char_plot_path = os.path.join(self.plots_dir, 'regime_characteristics.png')
                fig.savefig(regime_char_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Regime characteristics plot saved to {regime_char_plot_path}")
            except Exception as e:
                logger.error(f"Error generating regime characteristics plot: {str(e)}")
    
    def backtest(self, data: pd.DataFrame, regime_strategies: Dict[str, Dict] = None, initial_capital: float = 10000.0) -> Dict:
        """
        Perform a backtest of trading strategies based on identified market regimes.
        
        Args:
            data: DataFrame with market data and predicted regimes
            regime_strategies: Dictionary mapping regime names to trading strategies
            initial_capital: Initial capital for backtest
            
        Returns:
            Dict with backtest results
        """
        logger.info("Performing backtest of regime-based strategies")
        
        # Ensure data has required columns
        required_columns = ['timestamp', 'close', 'regime']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns for backtest: {missing}")
            return {}
        
        # Sort data by timestamp
        backtest_data = data.sort_values('timestamp')
        
        # If no strategies provided, use default strategies
        if not regime_strategies:
            # Get regime characteristics
            regime_stats = self.model.get_regime_characteristics()
            
            # Create default strategies based on regime characteristics
            regime_strategies = {}
            for regime, stats in regime_stats.items():
                if stats.get('mean_return', 0) > 0:
                    # Buy in positive-return regimes
                    regime_strategies[regime] = {
                        'action': 'buy',
                        'stop_loss_pct': 0.05,
                        'take_profit_pct': 0.1
                    }
                else:
                    # Do nothing in negative-return regimes
                    regime_strategies[regime] = {
                        'action': 'none'
                    }
            
            logger.info(f"Using default strategies based on regime characteristics: {regime_strategies}")
        
        # Initialize backtest variables
        capital = initial_capital
        position = None  # Current position
        position_size_pct = 0.5  # Percentage of capital to use for each position
        position_size = 0.0
        entry_price = 0.0
        entry_regime = None
        
        # Risk management parameters
        use_stop_loss = True
        use_take_profit = True
        
        # Track trades and daily equity
        trades = []
        daily_equity = []
        
        # Run backtest
        for i, row in backtest_data.iterrows():
            timestamp = row['timestamp']
            close = row['close']
            regime = row['regime']
            
            # Record daily equity
            current_equity = capital
            if position:
                # Calculate unrealized PnL
                pnl_pct = close / entry_price - 1
                pnl = position_size * pnl_pct
                current_equity = capital + position_size + pnl
            
            daily_equity.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'regime': regime
            })
            
            # Check for exit if in position
            if position:
                # Calculate unrealized PnL
                pnl_pct = close / entry_price - 1
                
                # Check stop loss
                stop_loss_pct = regime_strategies[entry_regime].get('stop_loss_pct', 0.05)
                take_profit_pct = regime_strategies[entry_regime].get('take_profit_pct', 0.1)
                
                if use_stop_loss and pnl_pct <= -stop_loss_pct:
                    # Exit position - stop loss hit
                    exit_price = close
                    pnl = position_size * pnl_pct
                    capital += position_size + pnl
                    
                    # Record trade
                    trades.append({
                        'entry_timestamp': position['entry_timestamp'],
                        'exit_timestamp': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'entry_regime': entry_regime,
                        'exit_regime': regime,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0.0
                    entry_price = 0.0
                    entry_regime = None
                    
                    continue
                
                # Check take profit
                if use_take_profit and pnl_pct >= take_profit_pct:
                    # Exit position - take profit hit
                    exit_price = close
                    pnl = position_size * pnl_pct
                    capital += position_size + pnl
                    
                    # Record trade
                    trades.append({
                        'entry_timestamp': position['entry_timestamp'],
                        'exit_timestamp': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'entry_regime': entry_regime,
                        'exit_regime': regime,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'take_profit'
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0.0
                    entry_price = 0.0
                    entry_regime = None
                    
                    continue
                
                # Check regime change
                if regime != entry_regime:
                    # Exit position - regime changed
                    exit_price = close
                    pnl = position_size * pnl_pct
                    capital += position_size + pnl
                    
                    # Record trade
                    trades.append({
                        'entry_timestamp': position['entry_timestamp'],
                        'exit_timestamp': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'entry_regime': entry_regime,
                        'exit_regime': regime,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'regime_change'
                    })
                    
                    # Reset position
                    position = None
                    position_size = 0.0
                    entry_price = 0.0
                    entry_regime = None
            
            # Check for entry if not in position
            if not position and regime in regime_strategies:
                strategy = regime_strategies[regime]
                
                # Check if strategy suggests entry
                if strategy.get('action') == 'buy':
                    # Enter position
                    entry_price = close
                    position_size = capital * position_size_pct
                    capital -= position_size
                    
                    # Record position
                    position = {
                        'entry_timestamp': timestamp,
                        'entry_price': entry_price,
                        'size': position_size
                    }
                    
                    entry_regime = regime
        
        # Close any open position at the end of backtest
        if position:
            # Calculate final P&L
            close = backtest_data['close'].iloc[-1]
            pnl_pct = close / entry_price - 1
            pnl = position_size * pnl_pct
            capital += position_size + pnl
            
            # Record trade
            trades.append({
                'entry_timestamp': position['entry_timestamp'],
                'exit_timestamp': backtest_data['timestamp'].iloc[-1],
                'entry_price': entry_price,
                'exit_price': close,
                'entry_regime': entry_regime,
                'exit_regime': backtest_data['regime'].iloc[-1],
                'position_size': position_size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_backtest'
            })
        
        # Calculate backtest metrics
        backtest_metrics = self._calculate_backtest_metrics(trades, daily_equity, initial_capital)
        
        # Prepare results
        backtest_results = {
            'trades': trades,
            'daily_equity': daily_equity,
            'metrics': backtest_metrics,
            'regime_strategies': regime_strategies
        }
        
        # Save backtest results
        self._save_backtest_results(backtest_results)
        
        return backtest_results
    
    def _calculate_backtest_metrics(
        self,
        trades: List[Dict],
        daily_equity: List[Dict],
        initial_capital: float
    ) -> Dict:
        """
        Calculate backtest performance metrics.
        
        Args:
            trades: List of trade records
            daily_equity: List of daily equity records
            initial_capital: Initial capital
            
        Returns:
            Dict with backtest metrics
        """
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(daily_equity)
        
        # Initialize metrics
        metrics = {}
        
        # Return metrics if no trades
        if trades_df.empty:
            metrics['total_trades'] = 0
            metrics['final_equity'] = initial_capital
            metrics['total_return'] = 0.0
            metrics['total_return_pct'] = 0.0
            return metrics
        
        # Basic metrics
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = len(trades_df[trades_df['pnl'] > 0])
        metrics['losing_trades'] = len(trades_df[trades_df['pnl'] <= 0])
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0.0
        
        # PnL metrics
        metrics['total_pnl'] = trades_df['pnl'].sum()
        metrics['avg_pnl'] = trades_df['pnl'].mean()
        metrics['avg_win'] = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if metrics['winning_trades'] > 0 else 0.0
        metrics['avg_loss'] = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if metrics['losing_trades'] > 0 else 0.0
        metrics['profit_factor'] = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if metrics['losing_trades'] > 0 and trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
        
        # Equity metrics
        metrics['final_equity'] = equity_df['equity'].iloc[-1]
        metrics['total_return'] = metrics['final_equity'] - initial_capital
        metrics['total_return_pct'] = metrics['total_return'] / initial_capital
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        metrics['max_drawdown'] = abs(equity_df['drawdown'].min())
        
        # Calculate regime-specific metrics
        regime_metrics = {}
        for regime in equity_df['regime'].unique():
            regime_data = equity_df[equity_df['regime'] == regime]
            regime_trades = trades_df[trades_df['entry_regime'] == regime]
            
            if not regime_data.empty:
                regime_metrics[regime] = {
                    'trades': len(regime_trades),
                    'winning_trades': len(regime_trades[regime_trades['pnl'] > 0]),
                    'win_rate': len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades) if len(regime_trades) > 0 else 0.0,
                    'total_pnl': regime_trades['pnl'].sum(),
                    'avg_pnl': regime_trades['pnl'].mean() if not regime_trades.empty else 0.0
                }
        
        metrics['regime_metrics'] = regime_metrics
        
        return metrics
    
    def _save_backtest_results(self, backtest_results: Dict) -> None:
        """
        Save backtest results.
        
        Args:
            backtest_results: Dict with backtest results
            
        Returns:
            None
        """
        # Save backtest results
        backtest_path = os.path.join(self.model_dir, 'backtest_results.json')
        
        # Convert DataFrames to lists for JSON serialization
        serializable_results = {
            'trades': backtest_results.get('trades', []),
            'daily_equity': backtest_results.get('daily_equity', []),
            'metrics': backtest_results.get('metrics', {}),
            'regime_strategies': backtest_results.get('regime_strategies', {})
        }
        
        with open(backtest_path, 'w') as f:
            json.dump(serializable_results, f, indent=4, default=str)
        
        logger.info(f"Backtest results saved to {backtest_path}")
        
        # Generate backtest plots
        self._generate_backtest_plots(backtest_results)
    
    def _generate_backtest_plots(self, backtest_results: Dict) -> None:
        """
        Generate backtest performance plots.
        
        Args:
            backtest_results: Dict with backtest results
            
        Returns:
            None
        """
        logger.info("Generating backtest plots")
        
        # Extract data
        trades = backtest_results.get('trades', [])
        daily_equity = backtest_results.get('daily_equity', [])
        metrics = backtest_results.get('metrics', {})
        
        if not daily_equity:
            logger.warning("No equity data for backtest plots")
            return
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(daily_equity)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Plot equity curve
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot equity curve
            ax.plot(equity_df['timestamp'], equity_df['equity'], label='Equity')
            
            # Add regime background colors
            if 'regime' in equity_df.columns:
                regimes = equity_df['regime'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
                
                for i, regime in enumerate(regimes):
                    regime_data = equity_df[equity_df['regime'] == regime]
                    ax.fill_between(
                        regime_data['timestamp'],
                        regime_data['equity'].min(),
                        regime_data['equity'].max(),
                        color=colors[i],
                        alpha=0.2,
                        label=f'Regime {regime}'
                    )
            
            # Add trade markers
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df['entry_timestamp'] = pd.to_datetime(trades_df['entry_timestamp'])
                trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'])
                
                # Plot entry points
                for _, trade in trades_df.iterrows():
                    entry_equity = equity_df[equity_df['timestamp'] == trade['entry_timestamp']]['equity'].values
                    if len(entry_equity) > 0:
                        ax.scatter(
                            trade['entry_timestamp'],
                            entry_equity[0],
                            color='green',
                            marker='^',
                            s=50
                        )
                
                # Plot exit points
                for _, trade in trades_df.iterrows():
                    exit_equity = equity_df[equity_df['timestamp'] == trade['exit_timestamp']]['equity'].values
                    if len(exit_equity) > 0:
                        ax.scatter(
                            trade['exit_timestamp'],
                            exit_equity[0],
                            color='red',
                            marker='v',
                            s=50
                        )
            
            # Add metrics as text
            metrics_text = (
                f"Total Return: {metrics.get('total_return_pct', 0)*100:.2f}%\n"
                f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%\n"
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%"
            )
            
            ax.text(
                0.02, 0.95,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            ax.set_title('Backtest Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.legend()
            ax.grid(True)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Save plot
            equity_plot_path = os.path.join(self.plots_dir, 'backtest_equity.png')
            fig.savefig(equity_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Equity curve plot saved to {equity_plot_path}")
        except Exception as e:
            logger.error(f"Error generating equity curve plot: {str(e)}")
        
        # Plot regime performance
        if 'regime_metrics' in metrics:
            try:
                regime_metrics = metrics['regime_metrics']
                
                if regime_metrics:
                    # Create regime performance plot
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    axes = axes.flatten()
                    
                    # Extract data for plots
                    regimes = list(regime_metrics.keys())
                    win_rates = [stats.get('win_rate', 0) * 100 for stats in regime_metrics.values()]
                    total_pnl = [stats.get('total_pnl', 0) for stats in regime_metrics.values()]
                    avg_pnl = [stats.get('avg_pnl', 0) for stats in regime_metrics.values()]
                    trade_counts = [stats.get('trades', 0) for stats in regime_metrics.values()]
                    
                    # Plot win rates
                    axes[0].bar(regimes, win_rates)
                    axes[0].set_title('Win Rate by Regime (%)')
                    axes[0].set_xticklabels(regimes, rotation=45, ha='right')
                    
                    # Plot total PnL
                    axes[1].bar(regimes, total_pnl)
                    axes[1].set_title('Total PnL by Regime')
                    axes[1].set_xticklabels(regimes, rotation=45, ha='right')
                    
                    # Plot average PnL
                    axes[2].bar(regimes, avg_pnl)
                    axes[2].set_title('Average PnL by Regime')
                    axes[2].set_xticklabels(regimes, rotation=45, ha='right')
                    
                    # Plot trade counts
                    axes[3].bar(regimes, trade_counts)
                    axes[3].set_title('Number of Trades by Regime')
                    axes[3].set_xticklabels(regimes, rotation=45, ha='right')
                    
                    # Add overall title
                    plt.suptitle('Regime Performance Metrics')
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9)
                    
                    # Save regime performance plot
                    regime_plot_path = os.path.join(self.plots_dir, 'regime_performance.png')
                    fig.savefig(regime_plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Regime performance plot saved to {regime_plot_path}")
            except Exception as e:
                logger.error(f"Error generating regime performance plot: {str(e)}")

    def _analyze_model_explainability(self, data: pd.DataFrame):
        """
        Analyze model explainability using SHAP and generate plots.

        Args:
            data: DataFrame with prepared data (features and target)
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return
            
        # Check if model has xgb_model attribute (which is used in the model class)
        if not hasattr(self.model, 'xgb_model'):
            logger.warning("Model doesn't have xgb_model attribute, skipping explainability analysis")
            return
            
        # Use xgb_model as classifier_model for SHAP analysis
        classifier_model = self.model.xgb_model

        # Check if the classifier model is tree-based and compatible with SHAP
        if not isinstance(self.model.classifier_model, (xgb.Booster, xgb.XGBClassifier, RandomForestClassifier)):
             logger.warning(f"Model type {type(self.model.classifier_model)} not supported for SHAP analysis, skipping explainability.")
             return

        logger.info("Analyzing model explainability using SHAP")

        # Create plots directory
        plots_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Extract features and target
        if 'regime' not in data.columns:
             logger.warning("Target column 'regime' not found in data, cannot perform explainability analysis.")
             return

        X = data.drop('regime', axis=1)
        y = data['regime']

        # Handle potential non-numeric features in X
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        X_numeric = X[numeric_cols]

        if X_numeric.empty:
             logger.warning("No numeric features available for explainability analysis.")
             return

        # Sample data for SHAP analysis (for performance)
        sample_size = min(1000, len(X_numeric))
        X_sample = X_numeric.sample(sample_size, random_state=42)

        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model.classifier_model)

            # Calculate SHAP values
            # For multi-class models, shap_values will be a list of arrays
            shap_values = explainer.shap_values(X_sample)

            # If multi-class, calculate mean absolute SHAP values across classes for summary plots
            if isinstance(shap_values, list):
                 mean_abs_shap_values = np.abs(np.array(shap_values)).mean(0).mean(0) # Mean across classes and samples
            else:
                 mean_abs_shap_values = np.abs(shap_values).mean(0) # Mean across samples

            # Summary plot (Feature Importance)
            plt.figure(figsize=(12, 8))
            # For multi-class, plot against the SHAP values for the first class or average
            if isinstance(shap_values, list):
                 shap.summary_plot(shap_values[0], X_sample, feature_names=X_sample.columns, show=False) # Plot for first class
            else:
                 shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'shap_summary.png'))
            plt.close()

            # Bar plot (Mean Absolute SHAP Value)
            plt.figure(figsize=(12, 8))
            # For multi-class, plot against the mean absolute SHAP values
            if isinstance(shap_values, list):
                 shap.summary_plot(np.abs(np.array(shap_values)).mean(0), X_sample, feature_names=X_sample.columns, plot_type='bar', show=False)
            else:
                 shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, plot_type='bar', show=False)

            plt.title('SHAP Mean Absolute Value')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'shap_bar.png'))
            plt.close()

            # Feature Interaction Analysis (SHAP Dependence Plots for top features)
            logger.info("Generating SHAP dependence plots for top features")
            # Get top features based on mean absolute SHAP values
            top_feature_indices = np.argsort(mean_abs_shap_values)[::-1][:5] # Top 5 features

            for i in top_feature_indices:
                feature_name = X_sample.columns[i]
                plt.figure(figsize=(10, 6))
                # Plot dependence plot for the top feature
                # For multi-class, plot for the first class or a representative class
                if isinstance(shap_values, list):
                     shap.dependence_plot(feature_name, shap_values[0], X_sample, feature_names=X_sample.columns, show=False)
                else:
                     shap.dependence_plot(feature_name, shap_values, X_sample, feature_names=X_sample.columns, show=False)

                plt.title(f'SHAP Dependence Plot for {feature_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'shap_dependence_{feature_name}.png'))
                plt.close()

            # Partial Dependence Plots (for top features)
            logger.info("Generating Partial Dependence Plots for top features")
            from sklearn.inspection import PartialDependenceDisplay
            from sklearn.ensemble import RandomForestClassifier # Using a compatible model type for plotting

            # Need a trained model object that plot_partial_dependence can use
            # Assuming self.model.classifier_model is a compatible classifier

            # Check if the model is compatible with plot_partial_dependence
            if hasattr(self.model.classifier_model, 'predict_proba'): # Common for classifiers
                 # Select top features for PDP
                 top_pdp_features = X_sample.columns[top_feature_indices].tolist()

                 # Generate PDPs
                 # PartialDependenceDisplay requires a fitted estimator and feature names/indices
                 # It also works best with a subset of data for performance
                 display = PartialDependenceDisplay.from_estimator(
                     self.model.classifier_model, # Fitted estimator
                     X_sample, # Data used for plotting
                     top_pdp_features, # Features to plot
                     feature_names=X_sample.columns, # All feature names
                     n_jobs=-1, # Use all cores
                     grid_resolution=20 # Number of points in the grid
                 )
                 fig = display.figure_

                 plt.suptitle('Partial Dependence Plots for Top Features')
                 plt.tight_layout()
                 plt.savefig(os.path.join(plots_dir, 'partial_dependence_plots.png'))
                 plt.close(fig)
            else:
                 logger.warning("Model type not compatible with scikit-learn partial dependence plots, skipping PDP generation.")

            # Store feature importance (using mean absolute SHAP values)
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': mean_abs_shap_values
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)

            # Save feature importance
            feature_importance.to_csv(os.path.join(self.model_dir, 'feature_importance_shap.csv'), index=False)

            # Log top features
            top_features_list = feature_importance.head(20)
            logger.info("Top 20 most important features (SHAP):")
            for i, (feature, importance) in enumerate(zip(top_features_list['feature'], top_features_list['importance'])):
                logger.info(f"{i+1}. {feature}: {importance:.6f}")

        except Exception as e:
            logger.error(f"Error during SHAP analysis or plot generation: {str(e)}")
