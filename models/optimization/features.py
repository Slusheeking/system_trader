#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Optimization Features Module
-------------------------------------
This module handles feature engineering for the portfolio optimization model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('optimization_features')

class OptimizationFeatures:
    """
    Feature generator for portfolio optimization model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature generator.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        
        # Feature groups to generate
        self.feature_groups = config.get('feature_groups', {
            'returns': True,
            'volatility': True,
            'correlation': True,
            'risk_metrics': True,
            'economic_indicators': True,
            'sector_exposure': True
        })
        
        # Lookback periods for various indicators
        self.lookback_periods = config.get('lookback_periods', {
            'returns': [30, 60, 90, 180, 365],  # Days
            'volatility': [30, 60, 90],
            'correlation': 60,
            'risk_metrics': 90
        })
        
        # Maximum feature lookback (for determining data requirements)
        self.max_lookback = config.get('max_lookback', 365)
        
        logger.info("Portfolio Optimization Features initialized")
    
    def generate_features(self, market_data: pd.DataFrame, economic_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate features for portfolio optimization from market data.
        
        Args:
            market_data: DataFrame with OHLCV data for multiple assets
            economic_data: Optional DataFrame with economic indicators
            
        Returns:
            DataFrame with generated features
        """
        # Copy input data to avoid modifying the original
        data = market_data.copy()
        
        # Check required columns
        required_columns = ['symbol', 'timestamp', 'close']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns for feature generation: {missing}")
            return pd.DataFrame()
        
        # Apply feature generation functions for each enabled group
        if self.feature_groups.get('returns', True):
            data = self._generate_return_features(data)
        
        if self.feature_groups.get('volatility', True):
            data = self._generate_volatility_features(data)
        
        if self.feature_groups.get('correlation', True):
            data = self._generate_correlation_features(data)
        
        if self.feature_groups.get('risk_metrics', True):
            data = self._generate_risk_features(data)
        
        if self.feature_groups.get('economic_indicators', True) and economic_data is not None:
            data = self._generate_economic_features(data, economic_data)
        
        if self.feature_groups.get('sector_exposure', True):
            data = self._generate_sector_features(data)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Get list of generated features
        feature_columns = [col for col in data.columns if col not in required_columns]
        logger.info(f"Generated {len(feature_columns)} features")
        
        return data
    
    def _generate_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate return-related features.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with return features added
        """
        logger.info("Generating return features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate daily returns
            symbol_data['daily_return'] = symbol_data['close'].pct_change()
            
            # Calculate returns for different lookback periods
            for period in self.lookback_periods.get('returns', [30, 60, 90, 180, 365]):
                # Total return over period
                symbol_data[f'return_{period}d'] = (
                    symbol_data['close'] / symbol_data['close'].shift(period) - 1
                )
                
                # Annualized return
                symbol_data[f'return_{period}d_annualized'] = (
                    (1 + symbol_data[f'return_{period}d']) ** (365 / period) - 1
                )
                
                # Rolling average daily return
                symbol_data[f'avg_daily_return_{period}d'] = (
                    symbol_data['daily_return'].rolling(period).mean()
                )
            
            # Calculate cumulative returns
            symbol_data['cum_return_ytd'] = self._calculate_ytd_return(symbol_data)
            
            # Calculate momentum indicators
            symbol_data['momentum_1m_3m'] = (
                symbol_data['return_30d'] / symbol_data['return_90d']
                if 'return_30d' in symbol_data.columns and 'return_90d' in symbol_data.columns
                else np.nan
            )
            
            symbol_data['momentum_3m_6m'] = (
                symbol_data['return_90d'] / symbol_data['return_180d']
                if 'return_90d' in symbol_data.columns and 'return_180d' in symbol_data.columns
                else np.nan
            )
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _calculate_ytd_return(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate year-to-date return for each row.
        
        Args:
            data: DataFrame with price data for a single symbol
            
        Returns:
            Series with YTD returns
        """
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            timestamps = data['timestamp']
        
        # Initialize result series
        ytd_returns = pd.Series(index=data.index, dtype=float)
        
        # Group by year
        for year, year_group in data.groupby(timestamps.dt.year):
            # Get first close price of the year
            first_close = year_group['close'].iloc[0]
            
            # Calculate YTD return for each row in this year
            ytd_returns.loc[year_group.index] = year_group['close'] / first_close - 1
        
        return ytd_returns
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility-related features.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with volatility features added
        """
        logger.info("Generating volatility features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate daily returns if not already present
            if 'daily_return' not in symbol_data.columns:
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
            
            # Calculate volatility for different lookback periods
            for period in self.lookback_periods.get('volatility', [30, 60, 90]):
                # Standard deviation of returns (volatility)
                symbol_data[f'volatility_{period}d'] = (
                    symbol_data['daily_return'].rolling(period).std() * np.sqrt(252)  # Annualized
                )
                
                # Downside deviation (only negative returns)
                downside_returns = symbol_data['daily_return'].copy()
                downside_returns[downside_returns > 0] = 0
                symbol_data[f'downside_deviation_{period}d'] = (
                    downside_returns.rolling(period).std() * np.sqrt(252)  # Annualized
                )
                
                # Volatility of volatility
                if f'volatility_{period}d' in symbol_data.columns:
                    symbol_data[f'vol_of_vol_{period}d'] = (
                        symbol_data[f'volatility_{period}d'].rolling(period).std()
                    )
            
            # Calculate volatility ratios
            if all(f'volatility_{p}d' in symbol_data.columns for p in [30, 90]):
                symbol_data['volatility_ratio_1m_3m'] = (
                    symbol_data['volatility_30d'] / symbol_data['volatility_90d']
                )
            
            # Calculate maximum drawdown
            rolling_max = symbol_data['close'].rolling(252, min_periods=1).max()
            drawdown = symbol_data['close'] / rolling_max - 1
            symbol_data['max_drawdown_1y'] = drawdown.rolling(252, min_periods=1).min()
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate correlation-related features.
        
        Args:
            data: DataFrame with price data for multiple assets
            
        Returns:
            DataFrame with correlation features added
        """
        logger.info("Generating correlation features")
        df = data.copy()
        
        # Ensure we have daily returns
        if 'daily_return' not in df.columns:
            # Group by timestamp and symbol to ensure we have one row per day per symbol
            df = df.sort_values(['timestamp', 'symbol'])
            df['daily_return'] = df.groupby('symbol')['close'].pct_change()
        
        # Pivot the data to get returns by symbol
        returns_pivot = df.pivot_table(
            index='timestamp', 
            columns='symbol', 
            values='daily_return'
        )
        
        # Calculate correlation matrix
        correlation_period = self.lookback_periods.get('correlation', 60)
        rolling_corr = returns_pivot.rolling(correlation_period).corr()
        
        # Get market index if available (assuming first symbol is market index)
        if len(returns_pivot.columns) > 0:
            market_symbol = returns_pivot.columns[0]
            
            # Calculate correlation with market for each symbol
            for symbol in returns_pivot.columns:
                if symbol != market_symbol:
                    # Get correlation between this symbol and market
                    symbol_market_corr = rolling_corr.loc[(slice(None), symbol), market_symbol]
                    symbol_market_corr.index = symbol_market_corr.index.get_level_values(0)
                    
                    # Add to original dataframe
                    for date, corr_value in symbol_market_corr.items():
                        mask = (df['timestamp'] == date) & (df['symbol'] == symbol)
                        df.loc[mask, f'correlation_with_market_{correlation_period}d'] = corr_value
        
        # Calculate average correlation with all other assets
        for symbol in returns_pivot.columns:
            # Get correlations between this symbol and all others
            symbol_corrs = rolling_corr.loc[(slice(None), symbol), :]
            symbol_corrs = symbol_corrs.drop(symbol, axis=1, errors='ignore')
            
            # Calculate average correlation
            avg_corr = symbol_corrs.mean(axis=1)
            avg_corr.index = avg_corr.index.get_level_values(0)
            
            # Add to original dataframe
            for date, corr_value in avg_corr.items():
                mask = (df['timestamp'] == date) & (df['symbol'] == symbol)
                df.loc[mask, f'avg_correlation_{correlation_period}d'] = corr_value
        
        return df
    
    def _generate_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate risk-related features.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with risk features added
        """
        logger.info("Generating risk features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate daily returns if not already present
            if 'daily_return' not in symbol_data.columns:
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
            
            # Calculate risk metrics
            risk_period = self.lookback_periods.get('risk_metrics', 90)
            
            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            if f'avg_daily_return_{risk_period}d' in symbol_data.columns and f'volatility_{risk_period}d' in symbol_data.columns:
                symbol_data[f'sharpe_ratio_{risk_period}d'] = (
                    symbol_data[f'avg_daily_return_{risk_period}d'] * 252 / 
                    symbol_data[f'volatility_{risk_period}d']
                )
            
            # Sortino ratio (using downside deviation)
            if f'avg_daily_return_{risk_period}d' in symbol_data.columns and f'downside_deviation_{risk_period}d' in symbol_data.columns:
                symbol_data[f'sortino_ratio_{risk_period}d'] = (
                    symbol_data[f'avg_daily_return_{risk_period}d'] * 252 / 
                    symbol_data[f'downside_deviation_{risk_period}d']
                )
            
            # Calmar ratio (return / max drawdown)
            if f'return_{risk_period}d_annualized' in symbol_data.columns and 'max_drawdown_1y' in symbol_data.columns:
                symbol_data['calmar_ratio'] = (
                    symbol_data[f'return_{risk_period}d_annualized'] / 
                    symbol_data['max_drawdown_1y'].abs()
                )
            
            # Value at Risk (VaR) - 95% confidence
            symbol_data[f'var_95_{risk_period}d'] = (
                symbol_data['daily_return'].rolling(risk_period).quantile(0.05) * np.sqrt(252)
            )
            
            # Conditional VaR (CVaR) / Expected Shortfall
            def calculate_cvar(returns, confidence=0.05):
                var = np.percentile(returns, confidence * 100)
                return returns[returns <= var].mean()
            
            symbol_data[f'cvar_95_{risk_period}d'] = (
                symbol_data['daily_return'].rolling(risk_period).apply(
                    lambda x: calculate_cvar(x.dropna()), raw=False
                ) * np.sqrt(252)
            )
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_economic_features(self, data: pd.DataFrame, economic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features based on economic indicators.
        
        Args:
            data: DataFrame with price data
            economic_data: DataFrame with economic indicators
            
        Returns:
            DataFrame with economic features added
        """
        logger.info("Generating economic indicator features")
        df = data.copy()
        
        # Ensure economic data has a timestamp column
        if 'timestamp' not in economic_data.columns:
            logger.error("Economic data must have a timestamp column")
            return df
        
        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if not pd.api.types.is_datetime64_any_dtype(economic_data['timestamp']):
            economic_data['timestamp'] = pd.to_datetime(economic_data['timestamp'])
        
        # Merge economic data with price data
        # First, ensure economic data has daily values (forward fill missing dates)
        economic_data = economic_data.sort_values('timestamp')
        
        # Get all dates in the price data
        all_dates = pd.DataFrame({'timestamp': df['timestamp'].unique()})
        all_dates = all_dates.sort_values('timestamp')
        
        # Merge economic data with all dates and forward fill
        merged_economic = pd.merge_asof(
            all_dates, economic_data, on='timestamp', direction='backward'
        )
        
        # Forward fill missing values
        merged_economic = merged_economic.ffill()
        
        # Now merge with the original price data
        df = pd.merge(df, merged_economic, on='timestamp', how='left')
        
        # Calculate rate of change for economic indicators
        for col in merged_economic.columns:
            if col != 'timestamp' and pd.api.types.is_numeric_dtype(merged_economic[col]):
                # 1-month rate of change
                df[f'{col}_1m_change'] = df.groupby('symbol')[col].pct_change(30)
                
                # 3-month rate of change
                df[f'{col}_3m_change'] = df.groupby('symbol')[col].pct_change(90)
        
        return df
    
    def _generate_sector_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate sector exposure features.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with sector features added
        """
        logger.info("Generating sector exposure features")
        df = data.copy()
        
        # Check if sector information is available
        if 'sector' not in df.columns:
            logger.warning("Sector information not available, skipping sector features")
            return df
        
        # Calculate sector performance
        sector_returns = df.groupby(['timestamp', 'sector'])['daily_return'].mean().reset_index()
        sector_returns = sector_returns.rename(columns={'daily_return': 'sector_return'})
        
        # Merge sector returns back to the main dataframe
        df = pd.merge(df, sector_returns, on=['timestamp', 'sector'], how='left')
        
        # Calculate relative performance vs sector
        df['return_vs_sector'] = df['daily_return'] - df['sector_return']
        
        # Calculate rolling relative performance
        for period in [30, 90]:
            df[f'rolling_return_vs_sector_{period}d'] = (
                df.groupby('symbol')['return_vs_sector'].rolling(period).mean().reset_index(level=0, drop=True)
            )
        
        return df
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the generated features.
        
        Returns:
            Dictionary with feature metadata
        """
        metadata = {
            'feature_groups': self.feature_groups,
            'lookback_periods': self.lookback_periods,
            'max_lookback': self.max_lookback
        }
        
        return metadata