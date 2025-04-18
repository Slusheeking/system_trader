#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Optimization Trainer
-----------------------------
This module handles training and evaluation of portfolio optimization models.
"""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.trainer import BaseTrainer
from mlflow.tracking import get_tracker
from models.optimization.model import PortfolioOptimizationModel
from models.optimization.features import OptimizationFeatures
from models.optimization.gh200_optimizer import GH200Optimizer
from models.optimization.onnx_converter import ONNXConverter

# Setup logging
logger = logging.getLogger(__name__)


class PortfolioOptimizationTrainer(BaseTrainer):
    """
    Trainer for portfolio optimization models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        # Initialize BaseTrainer
        super().__init__(
            params=config,
            tracking_uri=config.get('tracking_uri'),
            experiment_name=config.get('experiment_name')
        )
        
        # Initialize MLflow tracker
        self.tracker = get_tracker('portfolio_optimization', config.get('version'))
        
        self.config = config
        
        # Training parameters
        self.test_size = config.get('test_size', 0.2)
        self.cv_folds = config.get('cv_folds', 5)
        self.rebalance_frequency = config.get('rebalance_frequency', 'monthly')  # 'daily', 'weekly', 'monthly', 'quarterly'
        self.backtest_period = config.get('backtest_period', 365)  # days
        
        # Optimization parameters
        self.optimization_method = config.get('optimization_method', 'efficient_frontier')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.target_return = config.get('target_return', 0.15)
        
        # Path settings
        self.model_dir = config.get('model_dir', 'models/optimization')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize feature generator
        self.feature_generator = OptimizationFeatures(config.get('feature_config', {}))
        
        # Initialize model
        self.model = PortfolioOptimizationModel(config)
        
        # Performance metrics
        self.metrics = {}
        
        # Training history
        self.history = {}
        
        # Hardware optimization
        self.use_hardware_optimization = config.get('use_hardware_optimization', False)
        self.target_hardware = config.get('target_hardware', 'gh200')
        
        logger.info("Portfolio Optimization Trainer initialized")
    
    # Implement abstract methods from BaseTrainer
    def load_data(self) -> Any:
        """
        Load and return raw data required for training.
        """
        data_path = self.config.get('data_path')
        if not data_path:
            raise ValueError("data_path not specified in config")
        
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        try:
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            logger.info(f"Loaded {len(data)} rows of data")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, data: Any) -> Any:
        """
        Transform raw data into feature set for model training.
        """
        logger.info("Generating features")
        feature_data = self.feature_generator.generate_features(data)
        logger.info(f"Generated features, resulting in {len(feature_data)} rows")
        return feature_data
    
    def generate_target(self, data: Any) -> Any:
        """
        Generate and return target values/labels from raw data.
        For portfolio optimization, this is typically not needed as we're
        optimizing based on returns and covariance, not predicting a target.
        """
        logger.info("Portfolio optimization doesn't require target generation")
        return data
    
    def select_features(self, features: pd.DataFrame, target: Any) -> pd.DataFrame:
        """
        Select and return a subset of features for model training based on correlation with asset returns.

        Args:
            features: Processed feature set
            target: Target values (not directly used for correlation in this case)

        Returns:
            Selected features
        """
        num_selected_features = self.config.get('num_selected_features', 50) # Get from config
        correlation_threshold = self.config.get('correlation_threshold', 0.1) # Get from config

        logger.info(f"Performing feature selection based on correlation with asset returns, selecting top {num_selected_features} features with average absolute correlation > {correlation_threshold}")

        if features.empty:
            logger.warning("Features is empty, skipping feature selection")
            return features

        try:
            # To calculate correlation with asset returns, we need the 'daily_return'
            # column which should be present in the features DataFrame after generation.
            if 'daily_return' not in features.columns or 'symbol' not in features.columns or 'timestamp' not in features.columns:
                 logger.warning("Required columns ('daily_return', 'symbol', 'timestamp') not found in features, skipping correlation-based feature selection")
                 return features

            # Pivot features to get features by asset over time
            # This might be memory intensive for large datasets
            # Consider sampling or alternative approaches for very large data
            feature_values = features.pivot_table(index='timestamp', columns='symbol')

            # Extract daily returns for correlation calculation
            daily_returns = feature_values[('daily_return', slice(None))]
            daily_returns.columns = daily_returns.columns.get_level_values(1) # Flatten multi-index

            # Drop the daily_return column from feature_values for correlation calculation
            feature_values = feature_values.drop('daily_return', axis=1, level=0)

            # Calculate correlation of each feature with each asset's daily return
            correlations = {}
            for feature_col in feature_values.columns.get_level_values(0).unique():
                feature_data = feature_values[feature_col]
                # Calculate correlation with each asset's return
                feature_correlations = feature_data.corrwith(daily_returns).abs().mean() # Average absolute correlation across assets
                correlations[feature_col] = feature_correlations

            # Convert correlations to a Series and sort
            correlations_series = pd.Series(correlations)
            sorted_correlations = correlations_series.sort_values(ascending=False)

            # Filter features based on correlation threshold
            high_correlation_features = sorted_correlations[sorted_correlations > correlation_threshold].index.tolist()

            # Select top N features from the high correlation features
            # If fewer high correlation features than num_selected_features, select all
            selected_features_names = sorted_correlations.loc[high_correlation_features].head(num_selected_features).index.tolist()

            logger.info(f"Selected {len(selected_features_names)} features: {selected_features_names}")

            # Return DataFrame with selected features and the necessary columns for optimization
            # Include 'symbol', 'timestamp', and 'daily_return' as they are needed later
            return features[selected_features_names + ['symbol', 'timestamp', 'daily_return']]

        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            logger.warning("Returning all features due to selection error")
            return features

    def train_model(self, features: Any, target: Any) -> Any:
        """
        Train and return the model.
        """
        logger.info("Training portfolio optimization model")
        
        # For portfolio optimization, we need to prepare returns data
        returns_data = self._prepare_returns_data(features)
        
        # Get sector data if available
        sector_data = self._prepare_sector_data(features)
        
        # Run optimization
        optimization_results = self.model.optimize(returns_data, sector_data)
        
        # Store metrics
        if 'error' not in optimization_results:
            self.metrics['train'] = {
                'sharpe_ratio': optimization_results.get('sharpe_ratio', 0),
                'expected_annual_return': optimization_results.get('expected_annual_return', 0),
                'expected_annual_volatility': optimization_results.get('expected_annual_volatility', 0)
            }
        
        return self.model
    
    def evaluate(self, model: Any, features: Any, target: Any) -> Dict[str, float]:
        """
        Evaluate the trained model and return performance metrics.
        """
        logger.info("Evaluating portfolio optimization model")
        
        # Run backtest
        backtest_results = self._run_backtest(features)
        
        # Store metrics
        self.metrics['test'] = backtest_results
        
        # Generate evaluation plots
        self._generate_evaluation_plots(backtest_results)
        
        return backtest_results
    
    def save_model(self, model: Any) -> None:
        """
        Persist the trained model to storage.
        """
        self._save_model()
        
        # If hardware optimization is enabled, convert to optimized format
        if self.use_hardware_optimization:
            self._optimize_for_hardware()
    
    def _prepare_returns_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare returns data for optimization.
        
        Args:
            data: DataFrame with price data
        """
        # Check if returns are already calculated
        if 'daily_return' in data.columns:
            # Pivot the data to get returns by asset
            returns_data = data.pivot_table(
                index='timestamp',
                columns='symbol',
                values='daily_return'
            )
            return returns_data
        
        # Calculate returns from price data
        if all(col in data.columns for col in ['symbol', 'timestamp', 'close']):
            # Group by symbol and calculate returns
            returns = []
            
            for symbol, group in data.groupby('symbol'):
                group = group.sort_values('timestamp')
                group['daily_return'] = group['close'].pct_change()
                returns.append(group[['timestamp', 'symbol', 'daily_return']].dropna())
            
            # Combine all returns
            all_returns = pd.concat(returns, ignore_index=True)
            
            # Pivot to get returns by asset
            returns_data = all_returns.pivot_table(
                index='timestamp',
                columns='symbol',
                values='daily_return'
            )
            
            return returns_data
        
        raise ValueError("Data format not suitable for returns calculation")
    
    def _prepare_sector_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare sector data for optimization.
        
        Args:
            data: DataFrame with asset data
            
        Returns:
            DataFrame with sector mappings or None if not available
        """
        if 'sector' in data.columns:
            # Get unique symbol-sector mappings
            sector_data = data[['symbol', 'sector']].drop_duplicates()
            sector_data = sector_data.set_index('symbol')
            return sector_data
        
        return None
    
    def _run_backtest(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with backtest metrics
        """
        logger.info("Running backtest")
        
        # Prepare price data
        price_data = data.copy()
        
        # Ensure we have required columns
        required_columns = ['symbol', 'timestamp', 'close']
        if not all(col in price_data.columns for col in required_columns):
            logger.error("Price data missing required columns")
            return {'error': 'Price data missing required columns'}
        
        # Sort by timestamp
        price_data = price_data.sort_values('timestamp')
        
        # Get unique timestamps for rebalancing
        timestamps = price_data['timestamp'].unique()
        timestamps.sort()
        
        # Determine rebalancing dates
        rebalance_dates = self._get_rebalance_dates(timestamps)
        
        # Initialize portfolio
        initial_capital = 1000000  # $1M
        current_capital = initial_capital
        portfolio = {}  # {symbol: shares}
        portfolio_values = []  # [(timestamp, value)]
        
        # Run backtest
        for i, rebalance_date in enumerate(rebalance_dates):
            # Get data up to rebalance date
            historical_data = price_data[price_data['timestamp'] <= rebalance_date]
            
            # Prepare returns data for optimization
            returns_data = self._prepare_returns_data(historical_data)
            
            # Get sector data if available
            sector_data = self._prepare_sector_data(historical_data)
            
            # Run optimization
            optimization_results = self.model.optimize(returns_data, sector_data)
            
            if 'error' in optimization_results:
                logger.error(f"Optimization failed at {rebalance_date}: {optimization_results['error']}")
                continue
            
            # Get optimal weights
            weights = optimization_results['weights']
            
            # Get current prices
            current_prices = {}
            for symbol in weights.keys():
                symbol_data = price_data[(price_data['symbol'] == symbol) & 
                                        (price_data['timestamp'] <= rebalance_date)]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data.iloc[-1]['close']
            
            # Calculate current portfolio value
            current_value = 0
            for symbol, shares in portfolio.items():
                if symbol in current_prices:
                    current_value += shares * current_prices[symbol]
                else:
                    logger.warning(f"No price data for {symbol} at {rebalance_date}")
            
            # Add cash
            current_value += (current_capital - current_value)
            
            # Record portfolio value
            portfolio_values.append((rebalance_date, current_value))
            
            # Rebalance portfolio
            new_portfolio = {}
            for symbol, weight in weights.items():
                if symbol in current_prices:
                    target_value = current_value * weight
                    new_portfolio[symbol] = target_value / current_prices[symbol]
                else:
                    logger.warning(f"Cannot allocate to {symbol} at {rebalance_date} - no price data")
            
            # Update portfolio
            portfolio = new_portfolio
            
            # If not the last rebalance date, calculate portfolio value at next rebalance
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                next_prices = {}
                
                for symbol in portfolio.keys():
                    symbol_data = price_data[(price_data['symbol'] == symbol) & 
                                           (price_data['timestamp'] <= next_date)]
                    if not symbol_data.empty:
                        next_prices[symbol] = symbol_data.iloc[-1]['close']
                
                # Calculate portfolio value at next rebalance date
                next_value = 0
                for symbol, shares in portfolio.items():
                    if symbol in next_prices:
                        next_value += shares * next_prices[symbol]
                    else:
                        logger.warning(f"No price data for {symbol} at {next_date}")
                
                # Record portfolio value
                portfolio_values.append((next_date, next_value))
        
        # Calculate backtest metrics
        if len(portfolio_values) < 2:
            logger.error("Insufficient data for backtest")
            return {'error': 'Insufficient data for backtest'}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values, columns=['timestamp', 'value'])
        
        # Calculate returns
        portfolio_df['return'] = portfolio_df['value'].pct_change()
        
        # Calculate metrics
        total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        volatility = portfolio_df['return'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        
        # Calculate drawdown
        portfolio_df['peak'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] / portfolio_df['peak']) - 1
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calculate Sortino ratio
        downside_returns = portfolio_df['return'].copy()
        downside_returns[downside_returns > 0] = 0
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Store backtest data
        self.history['backtest'] = portfolio_df
        
        # Return metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio
        }
        
        logger.info(f"Backtest results: {metrics}")
        return metrics
    
    def _get_rebalance_dates(self, timestamps: np.ndarray) -> List[np.datetime64]:
        """
        Determine rebalancing dates based on frequency.
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            List of rebalancing dates
        """
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)
        
        # Sort timestamps
        timestamps = np.sort(timestamps)
        
        # Get start and end dates
        start_date = timestamps[0]
        end_date = timestamps[-1]
        
        # Generate rebalancing dates
        if self.rebalance_frequency == 'daily':
            # Daily rebalancing
            rebalance_dates = timestamps
        elif self.rebalance_frequency == 'weekly':
            # Weekly rebalancing (every Monday)
            date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')
            rebalance_dates = self._find_closest_timestamps(timestamps, date_range)
        elif self.rebalance_frequency == 'monthly':
            # Monthly rebalancing (first day of month)
            date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
            rebalance_dates = self._find_closest_timestamps(timestamps, date_range)
        elif self.rebalance_frequency == 'quarterly':
            # Quarterly rebalancing (first day of quarter)
            date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
            rebalance_dates = self._find_closest_timestamps(timestamps, date_range)
        else:
            # Default to monthly
            date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
            rebalance_dates = self._find_closest_timestamps(timestamps, date_range)
        
        return rebalance_dates
    
    def _find_closest_timestamps(self, timestamps: np.ndarray, target_dates: pd.DatetimeIndex) -> List[np.datetime64]:
        """
        Find closest actual timestamps to target dates.
        
        Args:
            timestamps: Array of available timestamps
            target_dates: Target rebalancing dates
            
        Returns:
            List of closest actual timestamps
        """
        result = []
        timestamps_series = pd.Series(timestamps)
        
        for target in target_dates:
            # Find closest timestamp not before target
            future_timestamps = timestamps_series[timestamps_series >= target]
            
            if not future_timestamps.empty:
                # Get the first timestamp not before target
                closest = future_timestamps.iloc[0]
                result.append(closest)
            else:
                # If no future timestamps, use the last available
                result.append(timestamps_series.iloc[-1])
        
        return result
    
    def _generate_evaluation_plots(self, backtest_results: Dict[str, float]) -> None:
        """
        Generate evaluation plots.
        
        Args:
            backtest_results: Dictionary with backtest metrics
        """
        # Create plots directory
        plots_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot portfolio value over time
        if 'backtest' in self.history:
            portfolio_df = self.history['backtest']
            
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_df['timestamp'], portfolio_df['value'])
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'portfolio_value.png'))
            plt.close()
            
            # Plot drawdown
            plt.figure(figsize=(12, 6))
            plt.fill_between(portfolio_df['timestamp'], portfolio_df['drawdown'], 0, color='red', alpha=0.3)
            plt.plot(portfolio_df['timestamp'], portfolio_df['drawdown'], color='red')
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'drawdown.png'))
            plt.close()
        
        # Plot optimal portfolio weights
        if hasattr(self.model, 'weights') and self.model.weights is not None:
            weights = {asset: weight for asset, weight in zip(self.model.asset_names, self.model.weights) if weight > 0.005}
            
            # Sort by weight
            weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
            
            plt.figure(figsize=(12, 6))
            plt.bar(weights.keys(), weights.values())
            plt.title('Optimal Portfolio Weights')
            plt.xlabel('Asset')
            plt.ylabel('Weight')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'portfolio_weights.png'))
            plt.close()
        
        # Plot efficient frontier if available
        if hasattr(self.model, 'efficient_frontier') and self.model.efficient_frontier is not None:
            plt.figure(figsize=(10, 6))
            
            # Plot efficient frontier
            plt.plot(
                self.model.efficient_frontier['risks'],
                self.model.efficient_frontier['returns'],
                'b-', linewidth=2, label='Efficient Frontier'
            )
            
            # Plot optimal portfolio
            if hasattr(self.model, 'expected_volatility') and hasattr(self.model, 'expected_return'):
                plt.scatter(
                    self.model.expected_volatility,
                    self.model.expected_return,
                    marker='*', s=100, color='r', label='Optimal Portfolio'
                )
            
            # Add labels and title
            plt.xlabel('Expected Volatility (Annualized)')
            plt.ylabel('Expected Return (Annualized)')
            plt.title('Efficient Frontier')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'efficient_frontier.png'))
            plt.close()
    
    def _save_model(self) -> None:
        """
        Save the trained model and metadata.
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        # Save model
        model_path = os.path.join(self.model_dir, 'portfolio_optimization_model.pkl')
        self.model.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'feature_metadata': self.feature_generator.get_feature_metadata()
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Log metadata as artifact
        self.tracker.log_artifact(metadata_path)
        
        logger.info(f"Model and metadata saved to {self.model_dir}")
    
    def _optimize_for_hardware(self) -> None:
        """
        Optimize the model for specific hardware.
        """
        if not self.use_hardware_optimization:
            return
        
        logger.info(f"Optimizing model for {self.target_hardware} hardware")
        
        try:
            # Create optimizer
            if self.target_hardware == 'gh200':
                optimizer = GH200Optimizer()
                
                # Get model path
                model_path = os.path.join(self.model_dir, 'portfolio_optimization_model.pkl')
                
                # Create output path
                output_path = os.path.join(self.model_dir, 'portfolio_optimization_model_optimized.onnx')
                
                # Convert to ONNX and optimize
                optimizer.optimize_onnx_for_gh200(model_path, output_path)
                
                logger.info(f"Model optimized for {self.target_hardware} and saved to {output_path}")
            else:
                logger.warning(f"Hardware optimization not supported for {self.target_hardware}")
        except Exception as e:
            logger.error(f"Error during hardware optimization: {str(e)}")
    
    def load_model(self) -> bool:
        """
        Load a trained model.
        """
        model_path = os.path.join(self.model_dir, 'portfolio_optimization_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Load model
        success = self.model.load(model_path)
        
        if success:
            logger.info(f"Model loaded from {model_path}")
            
            # Load metrics
            metrics_path = os.path.join(self.model_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
        
        return success
    
    def train(self, data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the portfolio optimization model.
        
        Args:
            data: Training data with price history
            test_data: Optional test data for evaluation
            
        Returns:
            Dict with training metrics
        """
        # Start MLflow run
        run_name = f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker.start_run(run_name=run_name)
        
        # Log parameters
        self.tracker.log_params(self.config)
        
        try:
            logger.info("Starting model training")
            
            # Generate features
            feature_data = self.prepare_features(data)
            
            # Train model
            self.train_model(feature_data, None)  # No target needed for portfolio optimization
            
            # Log training metrics
            if 'train' in self.metrics:
                self.tracker.log_metrics(self.metrics['train'])
            
            # Evaluate on test data if provided
            if test_data is not None and not test_data.empty:
                logger.info("Evaluating model on test data")
                test_features = self.prepare_features(test_data)
                test_metrics = self.evaluate(self.model, test_features, None)
                
                # Log test metrics
                if 'error' not in test_metrics:
                    self.tracker.log_metrics(test_metrics)
            
            # Save model
            self._save_model()
            
            # Log model to MLflow
            self.tracker.log_model(self.model, artifact_path="model")
            
            return self.metrics
        finally:
            # Ensure the MLflow run is always closed
            self.tracker.end_run()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'model_dir': 'models/optimization',
        'optimization_method': 'efficient_frontier',
        'risk_free_rate': 0.02,
        'target_return': 0.15,
        'max_position_size': 0.2,
        'min_position_size': 0.01,
        'max_sector_allocation': 0.3,
        'rebalance_frequency': 'monthly',
        'use_hardware_optimization': False
    }
    
    # Create trainer
    trainer = PortfolioOptimizationTrainer(config)
    
    # Example usage
    # data = pd.read_csv('data/stock_prices.csv')
    # trainer.train(data)