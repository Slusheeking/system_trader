#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Peak Detection Features
---------------------
This module provides feature generation for the peak detection model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

# Setup logging
logger = logging.getLogger(__name__)

class PeakDetectionFeatures:
    """
    Feature generator for the peak detection model.
    
    This class handles the generation of features for detecting price peaks
    in financial time series data.
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
            'price': True,
            'volume': True,
            'momentum': True,
            'volatility': True,
            'pattern': True
        })
        
        # Technical indicator parameters
        self.rsi_periods = config.get('rsi_periods', [2, 7, 14, 21])
        self.ma_periods = config.get('ma_periods', [5, 10, 20, 50, 200])
        self.bb_periods = config.get('bb_periods', [20])
        self.atr_periods = config.get('atr_periods', [14])
        
        # Peak detection parameters
        self.peak_window = config.get('peak_window', 5)
        self.peak_threshold = config.get('peak_threshold', 0.005)
        self.lookahead_window = config.get('lookahead_window', 10)
        
        # Feature statistics
        self.feature_stats = {}
        self.feature_list = []
        
        logger.info("Peak Detection Features initialized")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for peak detection.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with generated features and target variable
        """
        logger.info("Generating features for peak detection")
        
        # Check required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
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
            
            # Store feature list
            self.feature_list = [col for col in result_df.columns if col not in ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'peak']]
            
            # Calculate feature statistics
            self._calculate_feature_stats(result_df)
            
            logger.info(f"Generated {len(self.feature_list)} features for {len(result_df)} rows")
            
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
        
        # Ensure we have enough data
        min_data_points = max(self.ma_periods + [self.peak_window, self.lookahead_window]) * 2
        if len(df) < min_data_points:
            logger.warning(f"Not enough data points for {df['symbol'].iloc[0]}: {len(df)} < {min_data_points}")
            return pd.DataFrame()
        
        # Generate price features
        if self.feature_groups.get('price', True):
            df = self._generate_price_features(df)
        
        # Generate volume features
        if self.feature_groups.get('volume', True):
            df = self._generate_volume_features(df)
        
        # Generate momentum features
        if self.feature_groups.get('momentum', True):
            df = self._generate_momentum_features(df)
        
        # Generate volatility features
        if self.feature_groups.get('volatility', True):
            df = self._generate_volatility_features(df)
        
        # Generate pattern features
        if self.feature_groups.get('pattern', True):
            df = self._generate_pattern_features(df)
        
        # Generate target variable (peaks)
        df = self._generate_peak_targets(df)
        
        # Drop rows with missing values
        df = df.dropna()
        
        return df
    
    def _generate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with price features
        """
        df = data.copy()
        
        # Price returns
        df['return_1d'] = df['close'].pct_change()
        
        # Log returns
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Price relative to moving average
            df[f'close_to_ma_{period}'] = df['close'] / df[f'ma_{period}']
        
        # Moving average crossovers
        for i, period1 in enumerate(self.ma_periods[:-1]):
            for period2 in self.ma_periods[i+1:]:
                df[f'ma_cross_{period1}_{period2}'] = df[f'ma_{period1}'] / df[f'ma_{period2}']
        
        # Bollinger Bands
        for period in self.bb_periods:
            # Calculate middle band (SMA)
            df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            df[f'bb_std_{period}'] = df['close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + 2 * df[f'bb_std_{period}']
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - 2 * df[f'bb_std_{period}']
            
            # Calculate bandwidth
            df[f'bb_bandwidth_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
            
            # Calculate %B
            df[f'bb_b_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Price channels
        for period in [10, 20, 50]:
            df[f'highest_high_{period}'] = df['high'].rolling(window=period).max()
            df[f'lowest_low_{period}'] = df['low'].rolling(window=period).min()
            
            # Price relative to channels
            df[f'close_to_high_{period}'] = df['close'] / df[f'highest_high_{period}']
            df[f'close_to_low_{period}'] = df['close'] / df[f'lowest_low_{period}']
            
            # Channel width
            df[f'channel_width_{period}'] = (df[f'highest_high_{period}'] - df[f'lowest_low_{period}']) / df['close']
        
        return df
    
    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volume-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with volume features
        """
        df = data.copy()
        
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            
            # Volume relative to moving average
            df[f'volume_to_ma_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # Volume rate of change
        df['volume_roc_1d'] = df['volume'].pct_change()
        
        # Volume standard deviation
        df['volume_std_10d'] = df['volume'].rolling(window=10).std() / df['volume'].rolling(window=10).mean()
        
        # On-balance volume (OBV)
        df['obv'] = 0
        df.loc[df['close'] > df['close'].shift(1), 'obv'] = df['volume']
        df.loc[df['close'] < df['close'].shift(1), 'obv'] = -df['volume']
        df['obv'] = df['obv'].cumsum()
        
        # Price-volume trend
        df['pvt'] = df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['pvt'] = df['pvt'].cumsum()
        
        return df
    
    def _generate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with momentum features
        """
        df = data.copy()
        
        # RSI
        for period in self.rsi_periods:
            # Calculate price changes
            delta = df['close'].diff()
            
            # Calculate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        for period in [14]:
            # Calculate %K
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            
            # Calculate %D (3-day SMA of %K)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
        
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
        
        # Average True Range (ATR)
        for period in self.atr_periods:
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            df['tr'] = pd.DataFrame([tr1, tr2, tr3]).max()
            
            # Calculate ATR
            df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()
            
            # ATR percent
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close'] * 100
        
        # Historical volatility
        for period in [5, 10, 20]:
            # Calculate log returns
            log_returns = np.log(df['close'] / df['close'].shift(1))
            
            # Calculate standard deviation of log returns
            df[f'volatility_{period}'] = log_returns.rolling(window=period).std() * np.sqrt(252)
        
        # Bollinger Band width as volatility measure
        for period in self.bb_periods:
            if f'bb_bandwidth_{period}' not in df.columns:
                # Calculate middle band (SMA)
                middle_band = df['close'].rolling(window=period).mean()
                
                # Calculate standard deviation
                std_dev = df['close'].rolling(window=period).std()
                
                # Calculate upper and lower bands
                upper_band = middle_band + 2 * std_dev
                lower_band = middle_band - 2 * std_dev
                
                # Calculate bandwidth
                df[f'bb_bandwidth_{period}'] = (upper_band - lower_band) / middle_band
        
        return df
    
    def _generate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pattern-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with pattern features
        """
        df = data.copy()
        
        # Candlestick patterns
        
        # Doji
        df['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)
        
        # Hammer
        df['hammer'] = (
            ((df['high'] - df['low']) > 3 * (df['open'] - df['close'])) &
            ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6) &
            ((df['open'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6)
        ).astype(int)
        
        # Shooting Star
        df['shooting_star'] = (
            ((df['high'] - df['low']) > 3 * (df['open'] - df['close'])) &
            ((df['high'] - df['close']) / (0.001 + df['high'] - df['low']) > 0.6) &
            ((df['high'] - df['open']) / (0.001 + df['high'] - df['low']) > 0.6)
        ).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['close'] > df['open'])
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['close'] < df['open'])
        ).astype(int)
        
        # Price patterns
        
        # Higher highs and higher lows (uptrend)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['uptrend'] = ((df['higher_high'].rolling(window=3).sum() >= 2) & 
                         (df['higher_low'].rolling(window=3).sum() >= 2)).astype(int)
        
        # Lower highs and lower lows (downtrend)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['downtrend'] = ((df['lower_high'].rolling(window=3).sum() >= 2) & 
                          (df['lower_low'].rolling(window=3).sum() >= 2)).astype(int)
        
        # Clean up intermediate columns
        df = df.drop(['higher_high', 'higher_low', 'lower_high', 'lower_low'], axis=1)
        
        return df
    
    def _generate_peak_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate peak target variable.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            DataFrame with peak target variable
        """
        df = data.copy()
        
        # Initialize peak column
        df['peak'] = 0
        
        # Get peak window and lookahead window from config
        peak_window = self.peak_window
        lookahead_window = self.lookahead_window
        peak_threshold = self.peak_threshold
        
        # Ensure we have enough data
        if len(df) <= peak_window + lookahead_window:
            logger.warning(f"Not enough data for peak detection: {len(df)} <= {peak_window + lookahead_window}")
            return df
        
        # Iterate through the data
        for i in range(peak_window, len(df) - lookahead_window):
            # Get current price
            current_price = df['close'].iloc[i]
            
            # Check if current price is higher than previous peak_window prices
            is_higher_than_prev = all(current_price > df['close'].iloc[i-peak_window:i])
            
            # Check if current price is higher than next lookahead_window prices
            is_higher_than_next = all(current_price > df['close'].iloc[i+1:i+lookahead_window+1])
            
            # Check if price change exceeds threshold
            price_change = (current_price / df['close'].iloc[i-1] - 1)
            exceeds_threshold = price_change > peak_threshold
            
            # Mark as peak if all conditions are met
            if is_higher_than_prev and is_higher_than_next and exceeds_threshold:
                df['peak'].iloc[i] = 1
        
        return df
    
    def _calculate_feature_stats(self, data: pd.DataFrame) -> None:
        """
        Calculate feature statistics.
        
        Args:
            data: DataFrame with features
            
        Returns:
            None
        """
        # Calculate basic statistics for each feature
        for feature in self.feature_list:
            if feature in data.columns:
                self.feature_stats[feature] = {
                    'mean': data[feature].mean(),
                    'std': data[feature].std(),
                    'min': data[feature].min(),
                    'max': data[feature].max(),
                    'median': data[feature].median()
                }
    
    def get_feature_list(self) -> List[str]:
        """
        Get list of generated features.
        
        Returns:
            List of feature names
        """
        return self.feature_list
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature statistics.
        
        Returns:
            Dictionary with feature statistics
        """
        return self.feature_stats
    
    def identify_key_features(self, n: int = 20) -> List[str]:
        """
        Identify key features for the model.
        
        Args:
            n: Number of key features to return
            
        Returns:
            List of key feature names
        """
        # This is a placeholder implementation
        # In a real implementation, this would use feature importance from the model
        # or correlation with the target variable
        
        # For now, just return the first n features
        return self.feature_list[:min(n, len(self.feature_list))]
