#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering
------------------
This module implements feature engineering for the entry timing model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import pandas_ta as ta

from utils.logging import setup_logger

# Sentiment analysis functionality
# Note: ml_training_engine_modified module not found in the codebase
SENTIMENT_FEATURES_AVAILABLE = False

# Setup logging
logger = setup_logger('feature_engineering')


class FeatureEngineer:
    """
    Feature engineering for the entry timing model.
    """

    def __init__(self, config: Dict):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        
        # Feature parameters
        self.feature_groups = config.get('feature_groups', {
            'price': True,
            'volume': True,
            'technical': True,
            'orderbook': False,
            'sentiment': SENTIMENT_FEATURES_AVAILABLE  # Enable if available
        })
        
        # Technical parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.bb_period = config.get('bb_period', 20)
        self.bb_stddev = config.get('bb_stddev', 2.0)
        self.stoch_period = config.get('stoch_period', 14)
        self.stoch_slowk = config.get('stoch_slowk', 3)
        self.stoch_slowd = config.get('stoch_slowd', 3)
        self.adx_period = config.get('adx_period', 14)
        self.atr_period = config.get('atr_period', 14)
        
        logger.info("FeatureEngineer initialized")
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if not self.feature_groups.get('technical', True):
            logger.info("Technical indicators disabled, skipping")
            return data
        
        logger.info("Adding technical indicators")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        
        result_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Sort by timestamp to ensure correct calculations
            symbol_df = symbol_df.sort_values('timestamp')
            
            # Extract price and volume series
            closes = symbol_df['close'].values
            highs = symbol_df['high'].values
            lows = symbol_df['low'].values
            volumes = symbol_df['volume'].values
            
            # Calculate RSI
            try:
                symbol_df['rsi_14'] = ta.RSI(closes, timeperiod=self.rsi_period)
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}")
                symbol_df['rsi_14'] = np.nan
            
            # Calculate MACD
            try:
                macd, macd_signal, _ = ta.MACD(
                    closes, 
                    fastperiod=self.macd_fast, 
                    slowperiod=self.macd_slow, 
                    signalperiod=self.macd_signal
                )
                symbol_df['macd'] = macd
                symbol_df['macd_signal'] = macd_signal
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")
                symbol_df['macd'] = np.nan
                symbol_df['macd_signal'] = np.nan
            
            # Calculate Bollinger Bands
            try:
                upper, middle, lower = ta.BBANDS(
                    closes, 
                    timeperiod=self.bb_period, 
                    nbdevup=self.bb_stddev, 
                    nbdevdn=self.bb_stddev
                )
                # Calculate BB position (0 = at bottom, 1 = at top)
                bb_range = upper - lower
                symbol_df['bb_position'] = (closes - lower) / bb_range
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
                symbol_df['bb_position'] = np.nan
            
            # Calculate Stochastic Oscillator
            try:
                slowk, slowd = ta.STOCH(
                    highs, lows, closes, 
                    fastk_period=self.stoch_period,
                    slowk_period=self.stoch_slowk,
                    slowd_period=self.stoch_slowd
                )
                symbol_df['stoch_k'] = slowk
                symbol_df['stoch_d'] = slowd
            except Exception as e:
                logger.warning(f"Error calculating Stochastic Oscillator: {str(e)}")
                symbol_df['stoch_k'] = np.nan
                symbol_df['stoch_d'] = np.nan
            
            # Calculate ADX
            try:
                symbol_df['adx_14'] = ta.ADX(
                    highs, lows, closes, 
                    timeperiod=self.adx_period
                )
            except Exception as e:
                logger.warning(f"Error calculating ADX: {str(e)}")
                symbol_df['adx_14'] = np.nan
            
            # Calculate ATR and ATR ratio
            try:
                atr = ta.ATR(
                    highs, lows, closes, 
                    timeperiod=self.atr_period
                )
                # ATR as a percentage of price
                symbol_df['atr_ratio'] = atr / closes * 100
            except Exception as e:
                logger.warning(f"Error calculating ATR: {str(e)}")
                symbol_df['atr_ratio'] = np.nan
            
            result_dfs.append(symbol_df)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        
        # Fill missing values at the beginning
        result = result.fillna(method='bfill')
        
        num_features = len(result.columns) - len(data.columns)
        logger.info(f"Added {num_features} technical indicators")
        
        return result
    
    def add_orderbook_features(self, data: pd.DataFrame, orderbook_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add order book features to the data.
        
        Args:
            data: DataFrame with OHLCV data
            orderbook_data: DataFrame with order book data
            
        Returns:
            DataFrame with added order book features
        """
        if not self.feature_groups.get('orderbook', False):
            logger.info("Order book features disabled, skipping")
            return data
        
        if orderbook_data is None or len(orderbook_data) == 0:
            logger.warning("No order book data provided, skipping")
            return data
        
        logger.info("Adding order book features")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure required columns in orderbook_data
        required_columns = ['timestamp', 'symbol', 'bid_price', 'bid_size', 'ask_price', 'ask_size']
        if not all(col in orderbook_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in orderbook_data.columns]
            logger.error(f"Missing required columns in orderbook_data: {missing}")
            return df
        
        # Make a copy of orderbook_data
        ob_data = orderbook_data.copy()
        
        # Calculate order book features
        ob_data['bid_ask_spread'] = ob_data['ask_price'] - ob_data['bid_price']
        ob_data['bid_ask_spread_pct'] = ob_data['bid_ask_spread'] / ob_data['bid_price'] * 100
        ob_data['bid_depth'] = ob_data['bid_size'] * ob_data['bid_price']  # Value in base currency
        ob_data['ask_depth'] = ob_data['ask_size'] * ob_data['ask_price']
        ob_data['order_imbalance'] = (ob_data['bid_depth'] - ob_data['ask_depth']) / (ob_data['bid_depth'] + ob_data['ask_depth'])
        
        # Keep only the features we need
        ob_features = ob_data[['timestamp', 'symbol', 'bid_ask_spread', 'bid_ask_spread_pct', 
                              'bid_depth', 'ask_depth', 'order_imbalance']]
        
        # Merge order book features with main data
        # Using nearest timestamp match
        ob_features['timestamp'] = pd.to_datetime(ob_features['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        
        result_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_ob = ob_features[ob_features['symbol'] == symbol].copy()
            
            if len(symbol_ob) == 0:
                logger.warning(f"No order book data for symbol {symbol}, skipping")
                result_dfs.append(symbol_df)
                continue
            
            # Sort by timestamp
            symbol_df = symbol_df.sort_values('timestamp')
            symbol_ob = symbol_ob.sort_values('timestamp')
            
            # Merge using nearest timestamp
            # This assumes order book data might not align perfectly with price data
            merged = pd.merge_asof(
                symbol_df, 
                symbol_ob, 
                on='timestamp', 
                by='symbol',
                direction='nearest',
                tolerance=pd.Timedelta('1min')  # Maximum 1 minute difference
            )
            
            result_dfs.append(merged)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        
        num_features = len(result.columns) - len(df.columns)
        logger.info(f"Added {num_features} order book features")
        
        return result
    
    def add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to the data.
        
        Args:
            data: DataFrame with timestamp column
            
        Returns:
            DataFrame with added temporal features
        """
        logger.info("Adding temporal features")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Normalize hour to account for market patterns
        # Convert hour and minute to time of day (0 to 1)
        df['time_of_day'] = (df['hour'] * 60 + df['minute']) / (24 * 60)
        
        # Convert day of week to cyclic features
        # This creates a continuous representation that wraps around
        df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        
        # Convert time of day to cyclic features
        df['tod_sin'] = np.sin(df['time_of_day'] * 2 * np.pi)
        df['tod_cos'] = np.cos(df['time_of_day'] * 2 * np.pi)
        
        # Drop intermediate columns
        df = df.drop(['hour', 'minute', 'day_of_week', 'time_of_day'], axis=1)
        
        logger.info(f"Added 4 temporal features")
        
        return df
    
    def add_derived_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived price features to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added price features
        """
        if not self.feature_groups.get('price', True):
            logger.info("Price features disabled, skipping")
            return data
        
        logger.info("Adding derived price features")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        
        result_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Sort by timestamp to ensure correct calculations
            symbol_df = symbol_df.sort_values('timestamp')
            
            # Price volatility (high-low range as percent of open)
            symbol_df['price_volatility'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['open'] * 100
            
            # Log returns
            symbol_df['log_return'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            
            # Moving averages (5 and 20 periods)
            symbol_df['ma_5'] = symbol_df['close'].rolling(window=5).mean()
            symbol_df['ma_20'] = symbol_df['close'].rolling(window=20).mean()
            
            # Relative position to moving averages
            symbol_df['ma5_position'] = symbol_df['close'] / symbol_df['ma_5'] - 1
            symbol_df['ma20_position'] = symbol_df['close'] / symbol_df['ma_20'] - 1
            
            # Moving average convergence/divergence
            symbol_df['ma_ratio'] = symbol_df['ma_5'] / symbol_df['ma_20']
            
            result_dfs.append(symbol_df)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        
        # Fill missing values at the beginning
        result = result.fillna(method='bfill')
        
        num_features = len(result.columns) - len(data.columns)
        logger.info(f"Added {num_features} derived price features")
        
        return result
    
    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume features
        """
        if not self.feature_groups.get('volume', True):
            logger.info("Volume features disabled, skipping")
            return data
        
        logger.info("Adding volume features")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        
        result_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Sort by timestamp to ensure correct calculations
            symbol_df = symbol_df.sort_values('timestamp')
            
            # Volume rolling metrics
            symbol_df['volume_ma_5'] = symbol_df['volume'].rolling(window=5).mean()
            symbol_df['volume_ma_20'] = symbol_df['volume'].rolling(window=20).mean()
            
            # Relative volume
            symbol_df['relative_volume_5'] = symbol_df['volume'] / symbol_df['volume_ma_5']
            symbol_df['relative_volume_20'] = symbol_df['volume'] / symbol_df['volume_ma_20']
            
            # On-balance volume (OBV)
            close_diff = np.diff(symbol_df['close'].values, prepend=symbol_df['close'].iloc[0])
            obv = np.zeros_like(symbol_df['close'].values)
            
            for i in range(1, len(obv)):
                if close_diff[i] > 0:
                    obv[i] = obv[i-1] + symbol_df['volume'].iloc[i]
                elif close_diff[i] < 0:
                    obv[i] = obv[i-1] - symbol_df['volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            symbol_df['obv'] = obv
            
            # OBV rate of change
            symbol_df['obv_roc'] = symbol_df['obv'].pct_change(periods=5) * 100
            
            result_dfs.append(symbol_df)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        
        # Fill missing values at the beginning
        result = result.fillna(method='bfill')
        
        num_features = len(result.columns) - len(data.columns)
        logger.info(f"Added {num_features} volume features")
        
        return result
    
    def create_features(self, price_data: pd.DataFrame, orderbook_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create all features for the entry timing model.
        
        Args:
            price_data: DataFrame with OHLCV data
            orderbook_data: Optional DataFrame with order book data
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating features for entry timing model")
        
        # Make a copy to avoid modifying the original
        df = price_data.copy()
        
        # Verify required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            return df
        
        # Add features step by step
        
        # 1. Derived price features
        df = self.add_derived_price_features(df)
        
        # 2. Volume features
        df = self.add_volume_features(df)
        
        # 3. Technical indicators
        df = self.add_technical_indicators(df)
        
        # 4. Order book features (if available)
        if orderbook_data is not None and len(orderbook_data) > 0:
            df = self.add_orderbook_features(df, orderbook_data)
        
        # 5. Temporal features
        df = self.add_temporal_features(df)
        
        # Final cleanup
        # Drop rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN values")
        
        logger.info(f"Created features dataset with {len(df)} rows and {len(df.columns)} columns")
        
        return df
