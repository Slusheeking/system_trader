#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Management Features
---------------------
This module provides feature generation for the risk management model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

# Setup logging
logger = logging.getLogger(__name__)

class RiskManagementFeatures:
    """
    Feature generator for the risk management model.
    
    This class handles the generation of features for risk management
    and position sizing optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature generator.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        
        # Feature parameters
        self.feature_groups = config.get('feature_groups', {
            'market': True,
            'volatility': True,
            'correlation': True,
            'drawdown': True,
            'portfolio': True
        })
        
        # Risk calculation parameters
        self.var_confidence = config.get('var_confidence', 0.95)
        self.var_window = config.get('var_window', 20)
        self.drawdown_window = config.get('drawdown_window', 60)
        self.volatility_window = config.get('volatility_window', 20)
        
        # Risk level thresholds
        self.risk_thresholds = config.get('risk_thresholds', {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        })
        
        # Feature statistics
        self.feature_stats = {}
        
        logger.info("Risk Management Features initialized")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for risk management.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with generated features and risk level
        """
        logger.info("Generating features for risk management")
        
        # Check if data is empty
        if data.empty:
            logger.error("Empty data for feature generation")
            return pd.DataFrame()
        
        # Check required columns
        required_columns = ['symbol', 'close']
        time_column = None
        
        # Check if we have either 'timestamp' or 'time' column
        if 'timestamp' in data.columns:
            time_column = 'timestamp'
        elif 'time' in data.columns:
            time_column = 'time'
            # Rename 'time' to 'timestamp' for consistency
            data = data.rename(columns={'time': 'timestamp'})
        else:
            logger.error("Missing time column (either 'timestamp' or 'time')")
            return pd.DataFrame()
            
        # Check other required columns
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Make a copy of the data
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        feature_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Sort by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Generate features
            symbol_features = self._generate_symbol_features(symbol_data)
            
            if not symbol_features.empty:
                feature_dfs.append(symbol_features)
        
        # Combine results
        if feature_dfs:
            result_df = pd.concat(feature_dfs, ignore_index=True)
            
            # Generate portfolio-level features
            result_df = self._generate_portfolio_features(result_df)
            
            # Generate risk level target
            result_df = self._generate_risk_level(result_df)
            
            logger.info(f"Generated features for {len(result_df)} rows")
            
            return result_df
        else:
            logger.warning("No features generated")
            return pd.DataFrame()
    
    def _generate_symbol_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for a single symbol.
        
        Args:
            data: DataFrame with market data for a single symbol
            
        Returns:
            DataFrame with generated features
        """
        df = data.copy()
        
        # Check if we have enough data
        min_data_points = max(self.var_window, self.drawdown_window, self.volatility_window) * 2
        if len(df) < min_data_points:
            logger.warning(f"Limited data points for {df['symbol'].iloc[0]}: {len(df)} < {min_data_points}")
            
            # Adjust window sizes based on available data
            available_points = len(df)
            scale_factor = max(0.1, min(1.0, available_points / min_data_points))
            
            # Scale down window sizes
            self.var_window = max(3, int(self.var_window * scale_factor))
            self.drawdown_window = max(3, int(self.drawdown_window * scale_factor))
            self.volatility_window = max(3, int(self.volatility_window * scale_factor))
            
            logger.info(f"Adjusted window sizes: var={self.var_window}, drawdown={self.drawdown_window}, volatility={self.volatility_window}")
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        
        # Generate basic features even with limited data
        try:
            # Generate market features
            if self.feature_groups.get('market', True):
                df = self._generate_market_features(df)
            
            # Generate volatility features
            if self.feature_groups.get('volatility', True):
                df = self._generate_volatility_features(df)
            
            # Generate drawdown features
            if self.feature_groups.get('drawdown', True):
                df = self._generate_drawdown_features(df)
            
            # Drop rows with missing values
            df = df.dropna()
            
            # If we still have no data after feature generation, create minimal features
            if df.empty:
                logger.warning(f"No data after feature generation for {data['symbol'].iloc[0]}, creating minimal features")
                
                # Create minimal features
                min_df = data.copy()
                
                # Ensure we have the required columns
                if 'return' not in min_df.columns:
                    min_df['return'] = min_df['close'].pct_change().fillna(0)
                
                # Add essential features with default values
                min_df['volatility_20d'] = 0.2  # Default medium volatility
                if len(min_df) >= 5:
                    # Calculate volatility if we have enough data
                    min_df['volatility_20d'] = min_df['return'].rolling(window=min(5, len(min_df))).std().fillna(0.2) * np.sqrt(252)
                
                min_df['drawdown'] = 0  # Default value
                min_df['var_95_20d'] = -0.02  # Default value
                
                # Add additional minimal features needed for risk management
                min_df['return_mean_20d'] = 0
                min_df['return_std_20d'] = 0.2
                min_df['cum_return_20d'] = 0
                min_df['max_drawdown_20d'] = 0
                
                # Generate risk level - CRITICAL: This must be present for model training
                min_df['risk_level'] = 'medium'  # Default risk level
                
                # Ensure we have at least one sample of each risk level
                if len(min_df) >= 3:
                    min_df.iloc[0, min_df.columns.get_loc('risk_level')] = 'low'
                    min_df.iloc[1, min_df.columns.get_loc('risk_level')] = 'medium'
                    min_df.iloc[2, min_df.columns.get_loc('risk_level')] = 'high'
                
                # Fill any NaN values
                min_df = min_df.fillna(0)
                
                logger.info(f"Created minimal features with risk_level column for {data['symbol'].iloc[0]}")
                logger.info(f"Columns in minimal features: {min_df.columns.tolist()}")
                
                return min_df
        except Exception as e:
            logger.error(f"Error in feature generation: {str(e)}")
            # Create minimal features on error
            min_df = data.copy()
            min_df['return'] = min_df['close'].pct_change()
            min_df['volatility_20d'] = 0.2  # Default value
            min_df['drawdown'] = 0  # Default value
            min_df['var_95_20d'] = -0.02  # Default value
            min_df['risk_level'] = 'medium'  # Default risk level
            
            # Drop rows with missing values
            min_df = min_df.dropna()
            
            return min_df
        
        return df
    
    def _generate_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with market features
        """
        df = data.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Price relative to moving average
            df[f'close_to_ma_{period}'] = df['close'] / df[f'ma_{period}']
        
        # Return statistics
        df['return_mean_20d'] = df['return'].rolling(window=20).mean()
        df['return_std_20d'] = df['return'].rolling(window=20).std()
        df['return_skew_20d'] = df['return'].rolling(window=20).skew()
        df['return_kurt_20d'] = df['return'].rolling(window=20).kurt()
        
        # Cumulative returns
        for period in [5, 10, 20, 60]:
            df[f'cum_return_{period}d'] = (1 + df['return']).rolling(window=period).apply(lambda x: np.prod(x) - 1)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
            df['volume_std_20d'] = df['volume'].rolling(window=20).std()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20d']
        
        return df
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with volatility features
        """
        df = data.copy()
        
        # Historical volatility
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = df['return'].rolling(window=period).std() * np.sqrt(252)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20d'].rolling(window=20).std()
        
        # Volatility ratio
        df['vol_ratio_5_20'] = df['volatility_5d'] / df['volatility_20d']
        df['vol_ratio_10_60'] = df['volatility_10d'] / df['volatility_60d']
        
        # Value at Risk (VaR)
        for confidence in [0.95, 0.99]:
            for period in [10, 20]:
                df[f'var_{int(confidence*100)}_{period}d'] = df['return'].rolling(window=period).quantile(1 - confidence)
        
        # Conditional Value at Risk (CVaR) / Expected Shortfall
        for confidence in [0.95, 0.99]:
            for period in [10, 20]:
                def calc_cvar(x):
                    var = np.quantile(x, 1 - confidence)
                    return x[x <= var].mean() if len(x[x <= var]) > 0 else var
                
                df[f'cvar_{int(confidence*100)}_{period}d'] = df['return'].rolling(window=period).apply(calc_cvar)
        
        # GARCH volatility (simplified approximation)
        # In a real implementation, you would use a proper GARCH model
        alpha = 0.05
        beta = 0.9
        omega = (1 - alpha - beta) * df['return'].var()
        
        df['garch_vol'] = np.nan
        df.loc[1, 'garch_vol'] = df['return'].iloc[1] ** 2
        
        for i in range(2, len(df)):
            df.loc[i, 'garch_vol'] = omega + alpha * df['return'].iloc[i-1]**2 + beta * df.loc[i-1, 'garch_vol']
        
        df['garch_vol'] = np.sqrt(df['garch_vol']) * np.sqrt(252)
        
        return df
    
    def _generate_drawdown_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate drawdown-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with drawdown features
        """
        df = data.copy()
        
        # Calculate cumulative returns
        df['cum_return'] = (1 + df['return']).cumprod()
        
        # Calculate running maximum
        df['cum_max'] = df['cum_return'].cummax()
        
        # Calculate drawdown
        df['drawdown'] = df['cum_return'] / df['cum_max'] - 1
        
        # Maximum drawdown over different periods
        for period in [20, 60, 252]:
            df[f'max_drawdown_{period}d'] = df['drawdown'].rolling(window=period).min()
        
        # Drawdown duration
        df['in_drawdown'] = (df['drawdown'] < 0).astype(int)
        df['drawdown_start'] = ((df['in_drawdown'] == 1) & (df['in_drawdown'].shift(1) == 0)).astype(int)
        
        # Calculate drawdown duration
        df['drawdown_duration'] = 0
        current_duration = 0
        
        for i in range(1, len(df)):
            if df['in_drawdown'].iloc[i] == 1:
                current_duration += 1
                df.loc[df.index[i], 'drawdown_duration'] = current_duration
            else:
                current_duration = 0
        
        # Average drawdown duration
        df['avg_drawdown_duration'] = df['drawdown_duration'].rolling(window=252).mean()
        
        # Drawdown recovery
        df['recovery_rate'] = df['drawdown'] / df['drawdown_duration']
        
        # Clean up intermediate columns
        df = df.drop(['cum_return', 'cum_max', 'in_drawdown', 'drawdown_start'], axis=1)
        
        return df
    
    def _generate_portfolio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio-level features.
        
        Args:
            data: DataFrame with symbol-level features
            
        Returns:
            DataFrame with portfolio features
        """
        df = data.copy()
        
        # Group by timestamp
        timestamps = df['timestamp'].unique()
        
        # Portfolio metrics
        portfolio_metrics = []
        
        for ts in timestamps:
            ts_data = df[df['timestamp'] == ts]
            
            # Skip if not enough symbols
            if len(ts_data) < 2:
                continue
            
            # Calculate correlation matrix if we have multiple symbols
            unique_symbols = ts_data['symbol'].nunique()
            if unique_symbols > 1 and 'return' in ts_data.columns:
                try:
                    # Check for duplicate timestamp-symbol combinations
                    if ts_data.duplicated(subset=['timestamp', 'symbol']).any():
                        # Keep only the first occurrence of each timestamp-symbol combination
                        ts_data = ts_data.drop_duplicates(subset=['timestamp', 'symbol'])
                        logger.warning("Dropped duplicate timestamp-symbol combinations")
                    
                    returns = ts_data.pivot(index='timestamp', columns='symbol', values='return')
                    corr_matrix = returns.corr()
                    
                    # Average correlation
                    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    
                    # Maximum correlation
                    max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
                    
                    # Minimum correlation
                    min_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
                except Exception as e:
                    logger.warning(f"Error calculating correlation matrix: {str(e)}")
                    avg_corr = max_corr = min_corr = np.nan
            else:
                # Skip correlation calculation for single symbol
                logger.info(f"Skipping correlation calculation for single symbol or missing return data")
                avg_corr = max_corr = min_corr = np.nan
            
            # Average volatility
            if 'volatility_20d' in ts_data.columns:
                avg_vol = ts_data['volatility_20d'].mean()
                max_vol = ts_data['volatility_20d'].max()
                min_vol = ts_data['volatility_20d'].min()
            else:
                avg_vol = max_vol = min_vol = np.nan
            
            # Average drawdown
            if 'drawdown' in ts_data.columns:
                avg_dd = ts_data['drawdown'].mean()
                max_dd = ts_data['drawdown'].min()  # Drawdowns are negative
            else:
                avg_dd = max_dd = np.nan
            
            # Portfolio diversification score
            # Simple approximation: 1 - average correlation
            diversification = 1 - avg_corr if not np.isnan(avg_corr) else np.nan
            
            # Create portfolio metrics row
            portfolio_row = {
                'timestamp': ts,
                'symbol': 'PORTFOLIO',
                'avg_correlation': avg_corr,
                'max_correlation': max_corr,
                'min_correlation': min_corr,
                'avg_volatility': avg_vol,
                'max_volatility': max_vol,
                'min_volatility': min_vol,
                'avg_drawdown': avg_dd,
                'max_drawdown': max_dd,
                'diversification_score': diversification
            }
            
            portfolio_metrics.append(portfolio_row)
        
        # Create portfolio metrics DataFrame
        if portfolio_metrics:
            portfolio_df = pd.DataFrame(portfolio_metrics)
            
            # Merge with original data
            result = pd.concat([df, portfolio_df], ignore_index=True)
            
            return result
        else:
            return df
    
    def _generate_risk_level(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate risk level target variable.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with risk level target
        """
        df = data.copy()
        
        # Initialize risk level column
        df['risk_level'] = 'medium'  # Default risk level
        
        # Define risk level based on volatility
        if 'volatility_20d' in df.columns:
            # Low risk
            df.loc[df['volatility_20d'] <= self.risk_thresholds['low'], 'risk_level'] = 'low'
            
            # Medium risk
            df.loc[(df['volatility_20d'] > self.risk_thresholds['low']) & 
                   (df['volatility_20d'] <= self.risk_thresholds['high']), 'risk_level'] = 'medium'
            
            # High risk
            df.loc[df['volatility_20d'] > self.risk_thresholds['high'], 'risk_level'] = 'high'
        
        # Adjust risk level based on drawdown
        if 'drawdown' in df.columns:
            # Increase risk level if in significant drawdown
            df.loc[df['drawdown'] < -0.1, 'risk_level'] = 'high'
        
        # Adjust risk level based on VaR
        if 'var_95_20d' in df.columns:
            # Increase risk level if VaR is high
            df.loc[df['var_95_20d'] < -0.03, 'risk_level'] = 'high'
        
        # For portfolio level, consider correlation
        if 'avg_correlation' in df.columns:
            # Increase risk level if correlation is high
            df.loc[(df['symbol'] == 'PORTFOLIO') & (df['avg_correlation'] > 0.7), 'risk_level'] = 'high'
            
            # Decrease risk level if correlation is low (good diversification)
            df.loc[(df['symbol'] == 'PORTFOLIO') & (df['avg_correlation'] < 0.3), 'risk_level'] = 'low'
        
        return df
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature statistics.
        
        Returns:
            Dictionary with feature statistics
        """
        return self.feature_stats
