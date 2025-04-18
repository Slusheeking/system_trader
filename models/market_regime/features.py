#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Regime Features Module
----------------------------
This module generates features for the enhanced market regime detection model,
including standard market metrics and options flow data.
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Setup logging
logger = logging.getLogger(__name__)


class MarketRegimeFeatures:
    """
    Feature generator for the Enhanced Market Regime Detection Model.
    
    This class builds features for market regime detection, combining standard
    market metrics with options flow data from Unusual Whales.
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
            'returns': True,
            'volatility': True,
            'trend': True,
            'breadth': True,
            'sentiment': True,
            'options_flow': True  # Options flow data from Unusual Whales
        })
        
        # Lookback windows for features
        self.return_windows = config.get('return_windows', [1, 5, 10, 21, 63])
        self.vol_windows = config.get('vol_windows', [10, 21, 63])
        self.ma_windows = config.get('ma_windows', [10, 21, 50, 200])
        self.options_windows = config.get('options_windows', [5, 10, 21])
        
        # Lookback window for feature calculation
        self.lookback_window = max(
            self.return_windows + self.vol_windows + 
            self.ma_windows + self.options_windows + [200]  # Add 200 for 200-day MA
        )
        
        # Initialize feature metadata
        self.feature_metadata = {
            'feature_groups': [],
            'all_features': [],
            'categorical_features': [],
            'numeric_features': []
        }
        
        # Store feature statistics
        self.feature_stats = {}
        
        # Initialize the scaler
        self.scaler = StandardScaler()
        
        logger.info("Market Regime Features generator initialized")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for market regime detection.
        
        Args:
            data: DataFrame with market data and options flow data
            
        Returns:
            DataFrame with features for regime detection
        """
        logger.info(f"Generating market regime features from {len(data)} rows of data")
        
        if data.empty:
            logger.error("Empty input data")
            return pd.DataFrame()
        
        # Check required columns
        required_columns = ['timestamp', 'close']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Ensure sorted by timestamp
        data = data.sort_values('timestamp')
        
        # Generate features
        try:
            features = self._generate_market_features(data)
            
            # Store feature list
            self._update_feature_metadata(features)
            
            # Calculate feature statistics
            self._calculate_feature_stats(features)
            
            return features
        except Exception as e:
            logger.error(f"Error generating market regime features: {str(e)}")
            return pd.DataFrame()
    
    def _generate_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for market regime detection.
        
        This method processes market data and creates features for regime detection.
        
        Args:
            data: DataFrame with market data and options flow data
            
        Returns:
            DataFrame with features for regime detection
        """
        df = data.copy()
        
        # Initialize feature groups processed
        feature_groups_processed = []
        
        # Returns features
        if self.feature_groups.get('returns', True):
            df = self._add_returns_features(df)
            feature_groups_processed.append('returns')
        
        # Volatility features
        if self.feature_groups.get('volatility', True):
            df = self._add_volatility_features(df)
            feature_groups_processed.append('volatility')
        
        # Trend features
        if self.feature_groups.get('trend', True):
            df = self._add_trend_features(df)
            feature_groups_processed.append('trend')
        
        # Market breadth features
        if self.feature_groups.get('breadth', True):
            df = self._add_breadth_features(df)
            feature_groups_processed.append('breadth')
        
        # Sentiment features
        if self.feature_groups.get('sentiment', True):
            df = self._add_sentiment_features(df)
            feature_groups_processed.append('sentiment')
        
        # Options flow features
        if self.feature_groups.get('options_flow', True):
            df = self._add_options_flow_features(df)
            feature_groups_processed.append('options_flow')
        
        # Store feature groups processed
        self.feature_metadata['feature_groups'] = feature_groups_processed
        
        # Drop rows with NaN values due to lookback windows
        df = df.dropna()
        
        return df
    
    def _add_returns_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add return-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with return features added
        """
        df = data.copy()
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Return over different windows
        for window in self.return_windows:
            df[f'return_{window}d'] = df['close'].pct_change(window)
        
        # Return acceleration (change in return rate)
        for window in self.return_windows[:-1]:
            df[f'return_accel_{window}d'] = df[f'return_{window}d'].pct_change(5)
        
        # Cumulative returns
        for window in self.return_windows:
            df[f'cum_return_{window}d'] = (1 + df['returns']).rolling(window).apply(np.prod, raw=True) - 1
        
        # Return dispersion (standard deviation of returns)
        for window in self.return_windows:
            df[f'return_dispersion_{window}d'] = df['returns'].rolling(window).std()
        
        # Return skewness (asymmetry of returns)
        for window in self.return_windows:
            df[f'return_skew_{window}d'] = df['returns'].rolling(window).skew()
        
        # Return kurtosis (tail heaviness of returns)
        for window in self.return_windows:
            df[f'return_kurt_{window}d'] = df['returns'].rolling(window).kurt()
        
        return df
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with volatility features added
        """
        df = data.copy()
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Realized volatility over different windows
        for window in self.vol_windows:
            df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Volatility of volatility
        for window in self.vol_windows:
            df[f'vol_of_vol_{window}d'] = df[f'volatility_{window}d'].rolling(window=window).std()
        
        # Volatility ratios
        short_window = self.vol_windows[0]  # Shortest window
        for window in self.vol_windows[1:]:  # Longer windows
            df[f'vol_ratio_{short_window}_{window}d'] = df[f'volatility_{short_window}d'] / df[f'volatility_{window}d']
        
        # Volatility trend
        for window in self.vol_windows:
            df[f'vol_trend_{window}d'] = df[f'volatility_{window}d'].pct_change(5)
        
        # Add VIX features if available
        if 'vix' in df.columns:
            # VIX to realized volatility ratio
            for window in self.vol_windows:
                df[f'vix_to_realized_{window}d'] = df['vix'] / df[f'volatility_{window}d']
            
            # VIX ratios
            for window in self.vol_windows:
                vix_ma = df['vix'].rolling(window=window).mean()
                df[f'vix_ratio_{window}d'] = df['vix'] / vix_ma
        
        # Garman-Klass volatility if OHLC data available
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            for window in self.vol_windows:
                df[f'gk_vol_{window}d'] = self._calculate_gk_volatility(df, window)
        
        # GARCH volatility estimation
        # Note: For production code, you might want to use a proper GARCH implementation
        # Here we just use a simple proxy based on EWMA
        for window in self.vol_windows:
            half_life = window / 2
            span = 2 * half_life
            df[f'garch_vol_{window}d'] = df['returns'].ewm(span=span).std() * np.sqrt(252)
        
        return df
    
    def _calculate_gk_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator.
        
        Args:
            data: DataFrame with OHLC data
            window: Window for volatility calculation
            
        Returns:
            Series with GK volatility
        """
        log_hl = (np.log(data['high'] / data['low'])) ** 2
        log_co = (np.log(data['close'] / data['open'])) ** 2
        
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        gk_vol = np.sqrt(gk.rolling(window).mean() * 252)
        
        return gk_vol
    
    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with trend features added
        """
        df = data.copy()
        
        # Calculate moving averages
        for window in self.ma_windows:
            df[f'ma_{window}d'] = df['close'].rolling(window=window).mean()
        
        # Price relative to moving averages
        for window in self.ma_windows:
            df[f'price_to_ma_{window}d'] = df['close'] / df[f'ma_{window}d']
        
        # Moving average crossovers
        short_windows = self.ma_windows[:-1]
        for i, short_window in enumerate(short_windows):
            for long_window in self.ma_windows[i+1:]:
                df[f'ma_cross_{short_window}_{long_window}d'] = df[f'ma_{short_window}d'] / df[f'ma_{long_window}d']
        
        # Golden Cross / Death Cross indicators (50-day vs 200-day MA)
        if 50 in self.ma_windows and 200 in self.ma_windows:
            df['golden_cross'] = (df['ma_50d'] > df['ma_200d']).astype(int)
            df['death_cross'] = (df['ma_50d'] < df['ma_200d']).astype(int)
            
            # Distance from cross
            df['distance_from_cross'] = (df['ma_50d'] / df['ma_200d'] - 1) * 100
        
        # Calculate RSI
        if 'close' in df.columns:
            df['rsi_14d'] = ta.rsi(df['close'], length=14)
        
        # Calculate MACD
        if 'close' in df.columns:
            macd_result = ta.macd(
                df['close'], 
                fast=12, 
                slow=26, 
                signal=9
            )
            
            df['macd'] = macd_result['MACD_12_26_9']
            df['macd_signal'] = macd_result['MACDs_12_26_9']
            df['macd_hist'] = macd_result['MACDh_12_26_9']
        
        # Calculate ADX (trend strength)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx_result['ADX_14']
        
        # Trend duration (days since trend change)
        if 50 in self.ma_windows and 200 in self.ma_windows:
            # Up trend duration
            df['uptrend_change'] = ((df['ma_50d'] > df['ma_200d']) & 
                                  (df['ma_50d'].shift(1) <= df['ma_200d'].shift(1))).astype(int)
            df['uptrend_duration'] = self._calculate_streak_length(df, 'uptrend_change')
            
            # Down trend duration
            df['downtrend_change'] = ((df['ma_50d'] < df['ma_200d']) & 
                                    (df['ma_50d'].shift(1) >= df['ma_200d'].shift(1))).astype(int)
            df['downtrend_duration'] = self._calculate_streak_length(df, 'downtrend_change')
        
        return df
    
    def _calculate_streak_length(self, data: pd.DataFrame, change_col: str) -> pd.Series:
        """
        Calculate the length of a streak (days since last change).
        
        Args:
            data: DataFrame with change indicators
            change_col: Column name with change indicators (1 for change, 0 for no change)
            
        Returns:
            Series with streak lengths
        """
        # Find indices where changes occur
        change_indices = data.index[data[change_col] == 1].tolist()
        
        # If no changes, return all zeros
        if not change_indices:
            return pd.Series(0, index=data.index)
        
        # Initialize result series
        streak_length = pd.Series(0, index=data.index)
        
        # For each day, calculate days since last change
        for i, idx in enumerate(data.index):
            # Find the most recent change before this day
            last_change = None
            for change_idx in change_indices:
                if change_idx <= idx:
                    last_change = change_idx
                else:
                    break
            
            if last_change is not None:
                # Calculate days since last change
                days_since = data.index.get_loc(idx) - data.index.get_loc(last_change)
                streak_length.loc[idx] = days_since
        
        return streak_length
    
    def _add_breadth_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market breadth features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with market breadth features added
        """
        df = data.copy()
        
        # Add market breadth features if available
        breadth_cols = [
            'advance_decline', 'new_highs', 'new_lows', 
            'percent_above_200d_ma', 'percent_above_50d_ma'
        ]
        
        for col in breadth_cols:
            if col in df.columns:
                # Raw values
                # Moving averages
                for window in [5, 10, 21]:
                    df[f'{col}_ma_{window}d'] = df[col].rolling(window=window).mean()
                
                # Rate of change
                for window in [1, 5, 10]:
                    df[f'{col}_roc_{window}d'] = df[col].pct_change(window)
        
        # Add advance-decline line if advance_decline is available
        if 'advance_decline' in df.columns:
            df['ad_line'] = df['advance_decline'].cumsum()
            
            # AD line slope
            for window in [5, 10, 21]:
                df[f'ad_line_slope_{window}d'] = df['ad_line'].diff(window) / window
        
        # Add McClellan Oscillator if advance_decline is available
        if 'advance_decline' in df.columns:
            # McClellan Oscillator = 19-day EMA of AD - 39-day EMA of AD
            df['advance_decline_ema_19d'] = df['advance_decline'].ewm(span=19, adjust=False).mean()
            df['advance_decline_ema_39d'] = df['advance_decline'].ewm(span=39, adjust=False).mean()
            df['mcclellan_oscillator'] = df['advance_decline_ema_19d'] - df['advance_decline_ema_39d']
            
            # McClellan Summation Index
            df['mcclellan_summation'] = df['mcclellan_oscillator'].cumsum()
        
        # High-Low Index
        if all(col in df.columns for col in ['new_highs', 'new_lows']):
            df['high_low_index'] = df['new_highs'] / (df['new_highs'] + df['new_lows'])
            df['high_low_index'].fillna(0.5, inplace=True)  # Fill with neutral value when both are 0
            
            # High-Low Index moving averages
            for window in [5, 10, 21]:
                df[f'high_low_index_ma_{window}d'] = df['high_low_index'].rolling(window=window).mean()
        
        return df
    
    def _add_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment-based features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with sentiment features added
        """
        df = data.copy()
        
        # VIX features
        if 'vix' in df.columns:
            # VIX moving averages
            for window in [5, 10, 21, 63]:
                df[f'vix_ma_{window}d'] = df['vix'].rolling(window=window).mean()
            
            # VIX ratios
            for window in [5, 10, 21, 63]:
                df[f'vix_ratio_{window}d'] = df['vix'] / df[f'vix_ma_{window}d']
            
            # VIX rate of change
            for window in [1, 5, 10]:
                df[f'vix_roc_{window}d'] = df['vix'].pct_change(window)
            
            # VIX percentile rank (over past year)
            df['vix_rank_252d'] = df['vix'].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        # Put-Call Ratio features
        if 'put_call_ratio' in df.columns:
            # Put-Call Ratio moving averages
            for window in [5, 10, 21]:
                df[f'put_call_ratio_ma_{window}d'] = df['put_call_ratio'].rolling(window=window).mean()
            
            # Put-Call Ratio extremes
            df['pcr_extreme'] = ((df['put_call_ratio'] > 1.2) | (df['put_call_ratio'] < 0.7)).astype(int)
            
            # Put-Call Ratio percentile rank (over past quarter)
            df['pcr_rank_63d'] = df['put_call_ratio'].rolling(window=63).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        # Fear & Greed features (if available)
        if 'fear_greed_index' in df.columns:
            # Fear & Greed moving averages
            for window in [5, 10, 21]:
                df[f'fear_greed_ma_{window}d'] = df['fear_greed_index'].rolling(window=window).mean()
            
            # Fear & Greed extremes
            df['fear_extreme'] = (df['fear_greed_index'] < 25).astype(int)
            df['greed_extreme'] = (df['fear_greed_index'] > 75).astype(int)
        
        return df
    
    def _add_options_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add options flow-based features from Unusual Whales data.
        
        Args:
            data: DataFrame with market data and options flow data
            
        Returns:
            DataFrame with options flow features added
        """
        df = data.copy()
        
        # Options flow features
        options_cols = [
            'put_call_ratio', 'call_premium_volume', 'put_premium_volume',
            'smart_money_direction', 'unusual_activity_score', 'implied_volatility',
            'iv_skew', 'gamma_exposure'
        ]
        
        # Process each options flow feature if available
        for col in options_cols:
            if col in df.columns:
                # Moving averages
                for window in self.options_windows:
                    df[f'{col}_ma_{window}d'] = df[col].rolling(window=window).mean()
                
                # Rate of change
                for window in [1, 5]:
                    df[f'{col}_roc_{window}d'] = df[col].pct_change(window)
                
                # Z-score (how many standard deviations from the mean)
                for window in [21, 63]:
                    mean = df[col].rolling(window=window).mean()
                    std = df[col].rolling(window=window).std()
                    df[f'{col}_zscore_{window}d'] = (df[col] - mean) / std
                
                # Percentile rank
                for window in [21, 63]:
                    df[f'{col}_rank_{window}d'] = df[col].rolling(window=window).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    )
        
        # Premium imbalance
        if all(col in df.columns for col in ['call_premium_volume', 'put_premium_volume']):
            df['premium_imbalance'] = (df['call_premium_volume'] - df['put_premium_volume']) / (df['call_premium_volume'] + df['put_premium_volume'])
            
            # Premium imbalance moving averages
            for window in self.options_windows:
                df[f'premium_imbalance_ma_{window}d'] = df['premium_imbalance'].rolling(window=window).mean()
        
        # Smart money vs put_call_ratio divergence
        if all(col in df.columns for col in ['smart_money_direction', 'put_call_ratio']):
            # Normalize both to 0-1 range for comparison
            smart_money_norm = df['smart_money_direction']  # Assuming already 0-1
            pcr_norm = 1 - (df['put_call_ratio'] / (df['put_call_ratio'] + 1))  # Transform to 0-1
            
            df['smart_money_pcr_divergence'] = smart_money_norm - pcr_norm
            
            # Divergence moving average
            for window in self.options_windows:
                df[f'smart_money_pcr_divergence_ma_{window}d'] = df['smart_money_pcr_divergence'].rolling(window=window).mean()
        
        # IV to historical volatility ratio
        if all(col in df.columns for col in ['implied_volatility', 'volatility_21d']):
            df['iv_hv_ratio'] = df['implied_volatility'] / df['volatility_21d']
            
            # IV/HV ratio moving averages
            for window in self.options_windows:
                df[f'iv_hv_ratio_ma_{window}d'] = df['iv_hv_ratio'].rolling(window=window).mean()
        
        # Gamma exposure to price ratio
        if 'gamma_exposure' in df.columns:
            df['gamma_exposure_to_price'] = df['gamma_exposure'] / df['close']
            
            # Gamma exposure to price moving averages
            for window in self.options_windows:
                df[f'gamma_exposure_to_price_ma_{window}d'] = df['gamma_exposure_to_price'].rolling(window=window).mean()
        
        # Combined options sentiment indicator
        options_sentiment_cols = [col for col in options_cols if col in df.columns]
        if len(options_sentiment_cols) >= 3:
            # Create a simple sentiment score (0-1) based on available indicators
            sentiment_scores = []
            
            if 'put_call_ratio' in df.columns:
                # Lower put_call_ratio is bullish
                pcr_score = 1 - (df['put_call_ratio'] / (df['put_call_ratio'] + 1))
                sentiment_scores.append(pcr_score)
            
            if 'smart_money_direction' in df.columns:
                # Higher smart_money_direction is bullish
                sentiment_scores.append(df['smart_money_direction'])
            
            if 'unusual_activity_score' in df.columns:
                # Normalize unusual_activity_score to 0-1
                unusual_score = df['unusual_activity_score'] / df['unusual_activity_score'].max()
                sentiment_scores.append(unusual_score)
            
            if 'gamma_exposure' in df.columns:
                # Positive gamma exposure is bullish
                gamma_score = (df['gamma_exposure'] > 0).astype(float)
                sentiment_scores.append(gamma_score)
            
            # Calculate average sentiment score
            if sentiment_scores:
                df['options_sentiment'] = pd.concat(sentiment_scores, axis=1).mean(axis=1)
                
                # Options sentiment moving averages
                for window in self.options_windows:
                    df[f'options_sentiment_ma_{window}d'] = df['options_sentiment'].rolling(window=window).mean()
                
                # Options sentiment trend
                df['options_sentiment_trend'] = df['options_sentiment'].diff(5)
        
        return df
    
    def _update_feature_metadata(self, features: pd.DataFrame) -> None:
        """
        Update feature metadata after feature generation.
        
        Args:
            features: DataFrame with generated features
            
        Returns:
            None
        """
        # Store feature list
        self.feature_metadata['all_features'] = [col for col in features.columns 
                                             if col not in ['timestamp']]
        
        # Identify numeric features
        numeric_dtypes = ['int64', 'float64', 'int32', 'float32']
        self.feature_metadata['numeric_features'] = [col for col in self.feature_metadata['all_features'] 
                                                 if features[col].dtype.name in numeric_dtypes]
        
        # Identify categorical features
        self.feature_metadata['categorical_features'] = [col for col in self.feature_metadata['all_features'] 
                                                     if col not in self.feature_metadata['numeric_features']]
        
        # Log feature counts
        logger.info(f"Generated {len(self.feature_metadata['all_features'])} features")
        logger.info(f"  Numeric features: {len(self.feature_metadata['numeric_features'])}")
        logger.info(f"  Categorical features: {len(self.feature_metadata['categorical_features'])}")
    
    def _calculate_feature_stats(self, features: pd.DataFrame) -> None:
        """
        Calculate feature statistics for standardization and analysis.
        
        Args:
            features: DataFrame with generated features
            
        Returns:
            None
        """
        numeric_features = self.feature_metadata['numeric_features']
        
        # Calculate statistics for numeric features
        stats = {}
        for col in numeric_features:
            stats[col] = {
                'mean': features[col].mean(),
                'std': features[col].std(),
                'min': features[col].min(),
                'max': features[col].max(),
                'median': features[col].median(),
                'skew': features[col].skew(),
                'kurtosis': features[col].kurtosis()
            }
        
        self.feature_stats = stats
    
    def get_feature_metadata(self) -> Dict:
        """
        Get feature metadata.
        
        Returns:
            Dict with feature metadata
        """
        return self.feature_metadata
    
    def get_feature_stats(self) -> Dict:
        """
        Get feature statistics.
        
        Returns:
            Dict with feature statistics
        """
        return self.feature_stats
    
    def standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize numeric features.
        
        Args:
            features: DataFrame with features to standardize
            
        Returns:
            DataFrame with standardized features
        """
        df = features.copy()
        
        # Get numeric features
        numeric_features = [col for col in self.feature_metadata['numeric_features'] 
                          if col in df.columns]
        
        # Standardize numeric features
        if numeric_features:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        
        return df
