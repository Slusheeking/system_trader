#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Management Model
-------------------
This module implements an XGBoost-based risk management model for determining
optimal position sizes based on stock characteristics and market conditions.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

# Setup logging
logger = logging.getLogger(__name__)


class RiskManagementModel:
    """
    Risk Management Model using XGBoost to determine optimal position sizes
    based on stock characteristics, prediction confidence, and market conditions.
    """

    def __init__(self, config: Dict):
        """
        Initialize the risk management model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model hyperparameters
        self.n_estimators = config.get('n_estimators', 500)
        self.learning_rate = config.get('learning_rate', 0.05)
        self.max_depth = config.get('max_depth', 5)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.gamma = config.get('gamma', 0.1)
        self.min_child_weight = config.get('min_child_weight', 3)
        self.random_state = config.get('random_state', 42)
        
        # Risk parameters
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of capital
        self.risk_aversion = config.get('risk_aversion', 1.0)  # Risk aversion factor
        self.max_daily_risk = config.get('max_daily_risk', 0.02)  # 2% max daily risk
        
        # Feature parameters
        self.feature_groups = config.get('feature_groups', {
            'volatility': True,
            'liquidity': True,
            'prediction': True,
            'market': True,
            'historical': True
        })
        
        # Initialize model
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        
        # Model performance metrics
        self.metrics = {}
        
        logger.info("Risk Management Model initialized")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the risk management model.
        
        Args:
            data: DataFrame with stock data and prediction confidences
            
        Returns:
            DataFrame with features for risk management
        """
        df = data.copy()
        
        # Ensure required columns are present
        required_columns = ['symbol', 'timestamp', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Add volatility features
        if self.feature_groups.get('volatility', True):
            # Daily volatility (if not already present)
            if 'volatility_20d' not in df.columns:
                # Calculate daily returns
                df['daily_return'] = df.groupby('symbol')['close'].pct_change()
                
                # Calculate rolling volatility for 10 and 20 days
                df['volatility_10d'] = df.groupby('symbol')['daily_return'].transform(
                    lambda x: x.rolling(10).std() * np.sqrt(252)
                )
                df['volatility_20d'] = df.groupby('symbol')['daily_return'].transform(
                    lambda x: x.rolling(20).std() * np.sqrt(252)
                )
            
            # Volatility ratio
            df['volatility_ratio'] = df['volatility_10d'] / df['volatility_20d']
            
            # High-Low range
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            df['avg_range_5d'] = df.groupby('symbol')['daily_range'].transform(
                lambda x: x.rolling(5).mean()
            )
        
        # Add liquidity features
        if self.feature_groups.get('liquidity', True):
            # Average daily dollar volume
            df['dollar_volume'] = df['close'] * df['volume']
            df['avg_dollar_volume_10d'] = df.groupby('symbol')['dollar_volume'].transform(
                lambda x: x.rolling(10).mean()
            )
            
            # Volume volatility
            df['volume_volatility'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(10).std() / x.rolling(10).mean()
            )
            
            # Bid-ask spread (if available)
            if 'bid_price' in df.columns and 'ask_price' in df.columns:
                df['spread_pct'] = (df['ask_price'] - df['bid_price']) / df['close']
            else:
                # Use a proxy based on volatility if spread not available
                df['spread_pct'] = df['volatility_20d'] / np.sqrt(252) * 0.1
        
        # Add prediction confidence features
        if self.feature_groups.get('prediction', True):
            # Check if prediction confidences are available
            prediction_cols = ['selection_confidence', 'entry_confidence']
            available_cols = [col for col in prediction_cols if col in df.columns]
            
            if available_cols:
                # Use available prediction confidences
                for col in available_cols:
                    # Create confidence buckets
                    df[f'{col}_bucket'] = pd.cut(
                        df[col], 
                        bins=[-0.001, 0.25, 0.5, 0.75, 1.001], 
                        labels=['very_low', 'low', 'medium', 'high']
                    ).astype(str)
            else:
                # Use dummy values if no confidences available
                logger.warning("No prediction confidence columns found")
                df['prediction_confidence'] = 0.5
                df['prediction_confidence_bucket'] = 'medium'
        
        # Add market context features
        if self.feature_groups.get('market', True) and 'market_volatility' in df.columns:
            # Market volatility ratio
            df['vol_ratio_to_market'] = df['volatility_20d'] / df['market_volatility']
            
            # Beta (if market_return is available)
            if 'market_return' in df.columns:
                # Calculate rolling covariance and variance
                df['rolling_cov'] = df.groupby('symbol').apply(
                    lambda group: group['daily_return'].rolling(60).cov(group['market_return'])
                ).reset_index(level=0, drop=True)
                
                df['rolling_market_var'] = df.groupby('symbol')['market_return'].transform(
                    lambda x: x.rolling(60).var()
                )
                
                # Calculate beta
                df['beta'] = df['rolling_cov'] / df['rolling_market_var']
                
                # Clean up intermediate columns
                df = df.drop(['rolling_cov', 'rolling_market_var'], axis=1)
        
        # Add historical performance features
        if self.feature_groups.get('historical', True):
            # Monthly performance
            df['return_1m'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change(20)
            )
            
            # Return to volatility ratio
            df['return_vol_ratio'] = df['return_1m'] / df['volatility_20d']
            
            # Momentum signals
            df['momentum_3m'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change(60)
            )
        
        # Drop rows with missing values
        df = df.dropna()
        
        return df
    
    def _calculate_optimal_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate optimal position sizes using the Kelly criterion.
        
        Args:
            data: DataFrame with trading features
            
        Returns:
            DataFrame with optimal position sizes
        """
        df = data.copy()
        
        # Ensure required columns are present
        if 'volatility_20d' not in df.columns:
            logger.error("Volatility column missing")
            return pd.DataFrame()
        
        # Estimate win probability from prediction confidence if available
        if 'selection_confidence' in df.columns:
            df['win_probability'] = df['selection_confidence']
        else:
            # Use historical win rate as a default
            df['win_probability'] = 0.55  # Default win probability
        
        # Estimate win/loss ratio
        # Expected gain on winning trades (default to 1.5x risk)
        df['reward_risk_ratio'] = 1.5
        
        # Calculate baseline Kelly fraction
        df['kelly_fraction'] = (
            df['win_probability'] - 
            (1 - df['win_probability']) / df['reward_risk_ratio']
        ) / df['reward_risk_ratio']
        
        # Adjust for volatility
        df['volatility_adjustment'] = 1.0 / (1.0 + df['volatility_20d'])
        
        # Adjust for liquidity
        if 'avg_dollar_volume_10d' in df.columns:
            # Normalize dollar volume (higher is better)
            df['liquidity_score'] = df['avg_dollar_volume_10d'] / df['avg_dollar_volume_10d'].max()
            df['liquidity_adjustment'] = df['liquidity_score'].clip(0.1, 1.0)
        else:
            df['liquidity_adjustment'] = 1.0
            
        # Apply risk aversion factor and adjustments
        df['adjusted_kelly'] = (
            df['kelly_fraction'] * 
            df['volatility_adjustment'] * 
            df['liquidity_adjustment'] / 
            self.risk_aversion
        )
        
        # Clip negative values to zero and cap at maximum position size
        df['position_size'] = df['adjusted_kelly'].clip(0, self.max_position_size)
        
        return df
    
    def train(self, data: pd.DataFrame, historical_performance: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the risk management model.
        
        Args:
            data: DataFrame with stock data and prediction confidences
            historical_performance: Optional DataFrame with historical trade performance
            
        Returns:
            Dict with training metrics
        """
        logger.info("Preparing features for risk management model")
        
        # Prepare features
        feature_df = self._prepare_features(data)
        
        if feature_df.empty:
            logger.error("Failed to prepare features")
            return {}
        
        # Calculate baseline position sizes using Kelly criterion
        feature_df = self._calculate_optimal_position(feature_df)
        
        # If we have historical performance data, use it to optimize the model
        if historical_performance is not None and not historical_performance.empty:
            logger.info("Using historical performance data for model training")
            
            # Join historical performance with features
            training_data = pd.merge(
                feature_df,
                historical_performance,
                on=['symbol', 'timestamp'],
                how='inner'
            )
            
            if training_data.empty:
                logger.error("No matching data between features and historical performance")
                return {}
            
            # Use actual performance as target
            y = training_data['actual_profit_pct']
            
            # Select features for modeling
            exclude_cols = [
                'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'position_size', 'kelly_fraction', 'adjusted_kelly', 'actual_profit_pct',
                'win_probability', 'reward_risk_ratio', 'daily_return'
            ]
            
            feature_cols = [col for col in training_data.columns if col not in exclude_cols]
            X = training_data[feature_cols]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            logger.info(f"Training XGBoost model with {len(feature_cols)} features")
            
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state
            )
            
            # Perform time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                self.model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                cv_scores.append(mse)
            
            # Final fit on all data
            self.model.fit(X_scaled, y)
            
            # Get feature importance
            self.feature_columns = feature_cols
            self.feature_importance = dict(zip(
                self.feature_columns, 
                self.model.feature_importances_
            ))
            
            # Calculate metrics
            y_pred = self.model.predict(X_scaled)
            
            self.metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred),
                'cv_mse_mean': np.mean(cv_scores),
                'cv_mse_std': np.std(cv_scores)
            }
            
            logger.info(f"Model training metrics: {self.metrics}")
            
            return self.metrics
        
        else:
            # If no historical data, just use Kelly criterion
            logger.info("No historical performance data provided. Using Kelly criterion directly.")
            
            # Select features that would be used in a model
            exclude_cols = [
                'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'position_size', 'kelly_fraction', 'adjusted_kelly', 'daily_return'
            ]
            
            self.feature_columns = [col for col in feature_df.columns if col not in exclude_cols]
            
            # No model is trained, but we store features for future use
            self.model = None
            
            return {'message': 'Using Kelly criterion without ML model'}
    
    def predict_position_size(self, data: pd.DataFrame, prediction_confidences: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict optimal position sizes for new data.
        
        Args:
            data: DataFrame with stock data
            prediction_confidences: Optional DataFrame with prediction confidences
            
        Returns:
            DataFrame with predicted position sizes
        """
        # Merge prediction confidences if provided
        if prediction_confidences is not None and not prediction_confidences.empty:
            merged_data = pd.merge(
                data,
                prediction_confidences,
                on=['symbol', 'timestamp'],
                how='inner'
            )
        else:
            merged_data = data.copy()
        
        # Prepare features
        feature_df = self._prepare_features(merged_data)
        
        if feature_df.empty:
            logger.error("Failed to prepare features")
            return pd.DataFrame()
        
        # Calculate baseline position sizes using Kelly criterion
        position_df = self._calculate_optimal_position(feature_df)
        
        # If we have a trained model, use it to refine position sizes
        if self.model is not None:
            logger.info("Using trained model to refine position sizes")
            
            # Select features for prediction
            X = position_df[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict expected returns
            expected_returns = self.model.predict(X_scaled)
            
            # Adjust position sizes based on expected returns
            # Higher expected return = larger position
            position_df['expected_return'] = expected_returns
            
            # Normalize expected returns to 0-1 range
            min_return = position_df['expected_return'].min()
            max_return = position_df['expected_return'].max()
            
            if max_return > min_return:
                position_df['return_score'] = (position_df['expected_return'] - min_return) / (max_return - min_return)
            else:
                position_df['return_score'] = 0.5  # Default if all expected returns are the same
            
            # Adjust position size by expected return score
            position_df['model_position_size'] = (
                position_df['position_size'] * 
                (0.5 + 0.5 * position_df['return_score'])
            ).clip(0, self.max_position_size)
            
            # Use model-adjusted position size
            position_size_col = 'model_position_size'
        else:
            # Use Kelly criterion position size
            position_size_col = 'position_size'
        
        # Create result DataFrame
        result_df = position_df[['symbol', 'timestamp', position_size_col]].copy()
        result_df.rename(columns={position_size_col: 'position_size'}, inplace=True)
        
        # Add additional useful information
        result_df['volatility'] = position_df['volatility_20d']
        
        if 'selection_confidence' in position_df.columns:
            result_df['prediction_confidence'] = position_df['selection_confidence']
        
        if 'dollar_volume' in position_df.columns:
            result_df['dollar_volume'] = position_df['dollar_volume']
        
        return result_df
    
    def allocate_capital(self, position_sizes: pd.DataFrame, total_capital: float, max_positions: int = 10) -> pd.DataFrame:
        """
        Allocate capital across positions based on predicted position sizes.
        
        Args:
            position_sizes: DataFrame with predicted position sizes
            total_capital: Total capital available for allocation
            max_positions: Maximum number of positions to allocate
            
        Returns:
            DataFrame with capital allocation
        """
        # Sort by position size (descending)
        sorted_positions = position_sizes.sort_values('position_size', ascending=False)
        
        # Take top positions
        top_positions = sorted_positions.head(max_positions).copy()
        
        # Calculate proportion of each position
        total_size = top_positions['position_size'].sum()
        
        if total_size > 0:
            top_positions['allocation_pct'] = top_positions['position_size'] / total_size
        else:
            top_positions['allocation_pct'] = 0.0
        
        # Calculate dollar allocation
        top_positions['capital_allocation'] = top_positions['allocation_pct'] * total_capital
        
        # Ensure we don't exceed max daily risk
        total_risk = (
            (top_positions['capital_allocation'] * top_positions['volatility'] / np.sqrt(252)).sum() / 
            total_capital
        )
        
        if total_risk > self.max_daily_risk:
            # Scale down allocations to meet risk limit
            risk_scale_factor = self.max_daily_risk / total_risk
            top_positions['capital_allocation'] *= risk_scale_factor
            logger.info(f"Scaling allocations by {risk_scale_factor:.2f} to meet risk limit")
        
        return top_positions
    
    def calculate_stop_loss(self, data: pd.DataFrame, risk_per_trade: float = 0.01) -> pd.DataFrame:
        """
        Calculate stop-loss prices based on volatility and risk per trade.
        
        Args:
            data: DataFrame with stock data
            risk_per_trade: Maximum risk per trade as percentage of account
            
        Returns:
            DataFrame with stop-loss prices
        """
        result_df = data.copy()
        
        # Ensure required columns are present
        required_columns = ['symbol', 'timestamp', 'close', 'volatility_20d']
        if not all(col in result_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in result_df.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Calculate daily ATR if not present
        if 'atr_14' not in result_df.columns:
            # Use volatility as a proxy for ATR
            result_df['atr_proxy'] = result_df['close'] * result_df['volatility_20d'] / np.sqrt(252)
        else:
            result_df['atr_proxy'] = result_df['atr_14']
        
        # Calculate stop distance based on ATR
        result_df['stop_atr_multiplier'] = 1.5  # Default ATR multiplier
        
        # Adjust multiplier based on volatility
        result_df.loc[result_df['volatility_20d'] > 0.5, 'stop_atr_multiplier'] = 2.0  # More volatile stocks get wider stops
        result_df.loc[result_df['volatility_20d'] < 0.2, 'stop_atr_multiplier'] = 1.0  # Less volatile stocks get tighter stops
        
        # Calculate stop distance
        result_df['stop_distance'] = result_df['atr_proxy'] * result_df['stop_atr_multiplier']
        
        # Calculate stop price
        result_df['stop_loss_price'] = result_df['close'] - result_df['stop_distance']
        
        # Calculate position size based on risk
        result_df['risk_based_position'] = risk_per_trade / (result_df['stop_distance'] / result_df['close'])
        
        # Combine with model position size (if available)
        if 'position_size' in result_df.columns:
            result_df['final_position_size'] = np.minimum(
                result_df['position_size'],
                result_df['risk_based_position']
            )
        else:
            result_df['final_position_size'] = result_df['risk_based_position']
        
        # Clip to max position size
        result_df['final_position_size'] = result_df['final_position_size'].clip(0, self.max_position_size)
        
        # Select columns for result
        result_columns = [
            'symbol', 'timestamp', 'close', 'stop_loss_price', 
            'final_position_size', 'stop_distance'
        ]
        
        return result_df[result_columns]
    
    def save(self, path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Boolean indicating success
        """
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'scaler': self.scaler,
                'metrics': self.metrics,
                'config': self.config
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
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
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.scaler = model_data['scaler']
            self.metrics = model_data['metrics']
            self.config = model_data['config']
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False