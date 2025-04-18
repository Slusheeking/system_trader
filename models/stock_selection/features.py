#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stock Selection Features Module
-------------------------------
This module handles feature engineering for the stock selection model.
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple, Union

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('stock_selection_features')

# Try to import sentiment analysis functionality
try:
    # First try to import from ml_training_engine
    try:
        from ml_training_engine import generate_sentiment_features
        SENTIMENT_FEATURES_AVAILABLE = True
    except (ImportError, AttributeError):
        # If that fails, use our own implementation
        SENTIMENT_FEATURES_AVAILABLE = False
        
        # Define a simple sentiment feature generator
        def generate_sentiment_features(texts):
            """
            Generate sentiment features from text.
            
            Args:
                texts: List of text strings
                
            Returns:
                List of dictionaries with sentiment features
            """
            logger.info(f"Using fallback sentiment feature generator for {len(texts)} texts")
            
            # Try to use our sentiment analyzer
            try:
                from nlp.sentiment_analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer(use_rule_based=True)
                
                results = []
                for text in texts:
                    sentiment = analyzer.analyze(text)
                    results.append({
                        'sentiment_score': sentiment.get('score', 0),
                        'sentiment_magnitude': sentiment.get('magnitude', 0),
                        'sentiment_label': sentiment.get('classification', 'neutral')
                    })
                return results
            except Exception as e:
                logger.warning(f"Error using SentimentAnalyzer: {str(e)}")
                
                # Return dummy results
                return [{'sentiment_score': 0, 'sentiment_magnitude': 0, 'sentiment_label': 'neutral'} 
                        for _ in texts]
except Exception as e:
    logger.warning(f"Error setting up sentiment features: {str(e)}")
    SENTIMENT_FEATURES_AVAILABLE = False

class StockSelectionFeatures:
    """
    Feature generator for stock selection model.
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
            'price_action': True,
            'volume': True,
            'volatility': True,
            'technical': True,
            'market_context': True,
            'sentiment': SENTIMENT_FEATURES_AVAILABLE  # Enable if available
        })
        
        # Timeframes for features (in minutes for intraday features)
        self.timeframes = config.get('timeframes', [1, 5, 15, 30, 60])
        
        # Lookback periods for various indicators
        self.lookback_periods = config.get('lookback_periods', {
            'rsi': [7, 14, 21],
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': [20],
            'atr': [14],
            'adx': [14],
            'sentiment': [1, 3, 7, 14]  # Days for sentiment lookback
        })
        
        # Sentiment analysis parameters
        self.sentiment_config = config.get('sentiment', {
            'enabled': SENTIMENT_FEATURES_AVAILABLE,
            'weight': 0.5,  # Weight for sentiment features
            'sources': ['news', 'social', 'options']  # Data sources for sentiment
        })
        
        # Maximum feature lookback (for determining data requirements)
        self.max_lookback = config.get('max_lookback', 200)
        
        logger.info("Stock Selection Features initialized")
    
    def generate_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for stock selection from market data.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with generated features
        """
        # Copy input data to avoid modifying the original
        data = market_data.copy()
        
        # Check required columns
        required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns for feature generation: {missing}")
            return pd.DataFrame()
        
        # Apply feature generation functions for each enabled group
        if self.feature_groups.get('price_action', True):
            data = self._generate_price_action_features(data)
        
        if self.feature_groups.get('volume', True):
            data = self._generate_volume_features(data)
        
        if self.feature_groups.get('volatility', True):
            data = self._generate_volatility_features(data)
        
        if self.feature_groups.get('technical', True):
            data = self._generate_technical_features(data)
        
        if self.feature_groups.get('market_context', True) and 'market_return' in data.columns:
            data = self._generate_market_context_features(data)
        
        if self.feature_groups.get('sentiment', True) and 'news_sentiment' in data.columns:
            data = self._generate_sentiment_features(data)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Get list of generated features
        feature_columns = [col for col in data.columns if col not in required_columns]
        logger.info(f"Generated {len(feature_columns)} features")
        
        return data
    
    def _generate_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price action related features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price action features added
        """
        logger.info("Generating price action features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Price momentum at different timeframes
            for window in [5, 10, 20, 60]:
                symbol_data[f'return_{window}m'] = symbol_data['close'].pct_change(window)
            
            # Moving averages and their ratios
            for window in [5, 10, 20, 50, 200]:
                symbol_data[f'ma_{window}'] = symbol_data['close'].rolling(window=window).mean()
                symbol_data[f'close_to_ma_{window}'] = symbol_data['close'] / symbol_data[f'ma_{window}']
            
            # Exponential moving averages
            for window in [5, 10, 20, 50]:
                symbol_data[f'ema_{window}'] = symbol_data['close'].ewm(span=window, adjust=False).mean()
                symbol_data[f'close_to_ema_{window}'] = symbol_data['close'] / symbol_data[f'ema_{window}']
            
            # Moving average convergence/divergence relationships
            symbol_data['ma_5_10_ratio'] = symbol_data['ma_5'] / symbol_data['ma_10']
            symbol_data['ma_10_20_ratio'] = symbol_data['ma_10'] / symbol_data['ma_20']
            symbol_data['ma_20_50_ratio'] = symbol_data['ma_20'] / symbol_data['ma_50']
            
            # Price levels
            symbol_data['dist_to_high_10d'] = symbol_data['close'] / symbol_data['high'].rolling(10).max() - 1
            symbol_data['dist_to_low_10d'] = symbol_data['close'] / symbol_data['low'].rolling(10).min() - 1
            
            # Price gaps
            symbol_data['overnight_gap'] = symbol_data['open'] / symbol_data['close'].shift(1) - 1
            
            # Candle patterns
            try:
                # Implement candle pattern detection directly since pandas_ta might not have these functions
                
                # Hammer pattern (bullish)
                body_size = abs(symbol_data['close'] - symbol_data['open'])
                lower_shadow = symbol_data[['open', 'close']].min(axis=1) - symbol_data['low']
                upper_shadow = symbol_data['high'] - symbol_data[['open', 'close']].max(axis=1)
                
                # Hammer criteria: long lower shadow, small upper shadow, small body
                symbol_data['pattern_hammer'] = ((lower_shadow > 2 * body_size) & 
                                           (upper_shadow < 0.1 * body_size) & 
                                           (body_size > 0)).astype(int) * 100
                
                # Engulfing pattern
                prev_body_size = body_size.shift(1)
                prev_close = symbol_data['close'].shift(1)
                prev_open = symbol_data['open'].shift(1)
                
                # Bullish engulfing
                bullish_engulfing = ((symbol_data['close'] > symbol_data['open']) &  # Current candle is bullish
                                    (prev_close < prev_open) &  # Previous candle is bearish
                                    (symbol_data['open'] <= prev_close) &  # Current open is lower than or equal to previous close
                                    (symbol_data['close'] >= prev_open))   # Current close is higher than or equal to previous open
                
                # Bearish engulfing
                bearish_engulfing = ((symbol_data['close'] < symbol_data['open']) &  # Current candle is bearish
                                    (prev_close > prev_open) &  # Previous candle is bullish
                                    (symbol_data['open'] >= prev_close) &  # Current open is higher than or equal to previous close
                                    (symbol_data['close'] <= prev_open))   # Current close is lower than or equal to previous open
                
                symbol_data['pattern_engulfing'] = (bullish_engulfing.astype(int) * 100) - (bearish_engulfing.astype(int) * 100)
                
                # Simplified morning star (bullish reversal)
                symbol_data['pattern_morning_star'] = ((symbol_data['close'].shift(2) < symbol_data['open'].shift(2)) &  # First candle is bearish
                                                 (abs(symbol_data['close'].shift(1) - symbol_data['open'].shift(1)) < body_size.shift(2) * 0.5) &  # Second candle is small
                                                 (symbol_data['close'] > symbol_data['open']) &  # Third candle is bullish
                                                 (symbol_data['close'] > (symbol_data['open'].shift(2) + symbol_data['close'].shift(2)) / 2)  # Third candle closes above midpoint of first candle
                                                ).astype(int) * 100
                
                # Simplified shooting star (bearish)
                symbol_data['pattern_shooting_star'] = ((upper_shadow > 2 * body_size) & 
                                                  (lower_shadow < 0.1 * body_size) & 
                                                  (body_size > 0) &
                                                  (symbol_data['close'].shift(1) > symbol_data['open'].shift(1))  # Previous candle was bullish
                                                 ).astype(int) * 100
                
                # Simplified evening star (bearish reversal)
                symbol_data['pattern_evening_star'] = ((symbol_data['close'].shift(2) > symbol_data['open'].shift(2)) &  # First candle is bullish
                                                 (abs(symbol_data['close'].shift(1) - symbol_data['open'].shift(1)) < body_size.shift(2) * 0.5) &  # Second candle is small
                                                 (symbol_data['close'] < symbol_data['open']) &  # Third candle is bearish
                                                 (symbol_data['close'] < (symbol_data['open'].shift(2) + symbol_data['close'].shift(2)) / 2)  # Third candle closes below midpoint of first candle
                                                ).astype(int) * 100
                
                # Doji (indecision)
                symbol_data['pattern_doji'] = (body_size < (symbol_data['high'] - symbol_data['low']) * 0.1).astype(int) * 100
            except Exception as e:
                logger.warning(f"Error generating candle patterns: {str(e)}")
            
            # Range features
            symbol_data['daily_range'] = (symbol_data['high'] - symbol_data['low']) / symbol_data['close']
            symbol_data['daily_range_ma5'] = symbol_data['daily_range'].rolling(5).mean()
            
            # Price velocity and acceleration
            symbol_data['price_velocity'] = symbol_data['close'].diff(1)
            symbol_data['price_acceleration'] = symbol_data['price_velocity'].diff(1)
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volume related features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        logger.info("Generating volume features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Volume momentum
            for window in [5, 10, 20]:
                symbol_data[f'volume_ratio_{window}'] = symbol_data['volume'] / symbol_data['volume'].rolling(window=window).mean()
            
            # Relative volume (compared to average volume at this time)
            symbol_data['relative_volume'] = symbol_data['volume'] / symbol_data['volume'].rolling(20).mean()
            
            # Volume trend
            symbol_data['volume_trend_3d'] = (symbol_data['volume'].rolling(3).mean() / 
                                         symbol_data['volume'].rolling(10).mean())
            
            # On-balance volume (OBV)
            symbol_data['obv'] = (symbol_data['volume'] * 
                             ((symbol_data['close'] > symbol_data['close'].shift(1)).astype(int) - 
                              (symbol_data['close'] < symbol_data['close'].shift(1)).astype(int))).cumsum()
            symbol_data['obv_ratio_10d'] = symbol_data['obv'] / symbol_data['obv'].shift(10)
            
            # Volume-weighted average price (VWAP)
            try:
                # Assuming intraday data with a time component
                symbol_data['vwap'] = (symbol_data['volume'] * (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3).cumsum() / symbol_data['volume'].cumsum()
                symbol_data['close_to_vwap'] = symbol_data['close'] / symbol_data['vwap']
            except Exception as e:
                logger.warning(f"Error calculating VWAP: {str(e)}")
            
            # Volume spikes
            symbol_data['volume_spike'] = symbol_data['volume'] > (symbol_data['volume'].rolling(10).mean() * 2)
            
            # Price-volume relationship
            symbol_data['price_volume_correlation'] = symbol_data['close'].rolling(20).corr(symbol_data['volume'])
            
            # Money flow index
            try:
                typical_price = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
                money_flow = typical_price * symbol_data['volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
                
                money_ratio = positive_flow / negative_flow
                symbol_data['mfi_14'] = 100 - (100 / (1 + money_ratio))
            except Exception as e:
                logger.warning(f"Error calculating MFI: {str(e)}")
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility related features.
        
        Args:
            data: DataFrame with OHLCV data
            
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
            
            # Calculate returns if not present
            if 'daily_return' not in symbol_data.columns:
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
            
            # Average True Range (ATR)
            try:
                symbol_data['tr'] = np.maximum(
                    symbol_data['high'] - symbol_data['low'],
                    np.maximum(
                        abs(symbol_data['high'] - symbol_data['close'].shift(1)),
                        abs(symbol_data['low'] - symbol_data['close'].shift(1))
                    )
                )
                
                for period in self.lookback_periods.get('atr', [14]):
                    symbol_data[f'atr_{period}'] = symbol_data['tr'].rolling(period).mean()
                    symbol_data[f'atr_ratio_{period}'] = symbol_data[f'atr_{period}'] / symbol_data['close']
            except Exception as e:
                logger.warning(f"Error calculating ATR: {str(e)}")
            
            # Historical volatility
            for window in [10, 20, 60]:
                symbol_data[f'volatility_{window}'] = symbol_data['daily_return'].rolling(window).std() * np.sqrt(252)
            
            # Volatility trend
            symbol_data['volatility_trend'] = symbol_data['volatility_10'] / symbol_data['volatility_60']
            
            # Bollinger Bands
            for period in self.lookback_periods.get('bollinger', [20]):
                symbol_data[f'bb_middle_{period}'] = symbol_data['close'].rolling(period).mean()
                symbol_data[f'bb_std_{period}'] = symbol_data['close'].rolling(period).std()
                symbol_data[f'bb_upper_{period}'] = symbol_data[f'bb_middle_{period}'] + 2 * symbol_data[f'bb_std_{period}']
                symbol_data[f'bb_lower_{period}'] = symbol_data[f'bb_middle_{period}'] - 2 * symbol_data[f'bb_std_{period}']
                symbol_data[f'bb_width_{period}'] = (symbol_data[f'bb_upper_{period}'] - symbol_data[f'bb_lower_{period}']) / symbol_data[f'bb_middle_{period}']
                symbol_data[f'bb_position_{period}'] = (symbol_data['close'] - symbol_data[f'bb_lower_{period}']) / (symbol_data[f'bb_upper_{period}'] - symbol_data[f'bb_lower_{period}'])
            
            # Keltner Channels
            try:
                for period in self.lookback_periods.get('atr', [14]):
                    symbol_data[f'kc_middle_{period}'] = symbol_data['close'].rolling(period).mean()
                    symbol_data[f'kc_upper_{period}'] = symbol_data[f'kc_middle_{period}'] + 2 * symbol_data[f'atr_{period}']
                    symbol_data[f'kc_lower_{period}'] = symbol_data[f'kc_middle_{period}'] - 2 * symbol_data[f'atr_{period}']
                    symbol_data[f'kc_width_{period}'] = (symbol_data[f'kc_upper_{period}'] - symbol_data[f'kc_lower_{period}']) / symbol_data[f'kc_middle_{period}']
                    
                    # Squeeze Momentum - ensure Bollinger Bands columns exist before comparing
                    bb_lower_col = f'bb_lower_{period}'
                    bb_upper_col = f'bb_upper_{period}'
                    
                    if bb_lower_col in symbol_data.columns and bb_upper_col in symbol_data.columns:
                        symbol_data[f'squeeze_{period}'] = (symbol_data[bb_lower_col] > symbol_data[f'kc_lower_{period}']) & (symbol_data[bb_upper_col] < symbol_data[f'kc_upper_{period}'])
                    else:
                        # If Bollinger Bands columns don't exist, create a default squeeze indicator
                        symbol_data[f'squeeze_{period}'] = False
            except Exception as e:
                logger.warning(f"Error calculating Keltner Channels: {str(e)}")
            
            # Volatility-adjusted returns
            symbol_data['return_volatility_ratio'] = symbol_data['daily_return'] / symbol_data['volatility_20']
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicator features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicator features added
        """
        logger.info("Generating technical indicator features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Relative Strength Index (RSI)
            try:
                for period in self.lookback_periods.get('rsi', [14]):
                    delta = symbol_data['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(period).mean()
                    loss = -delta.where(delta < 0, 0).rolling(period).mean()
                    rs = gain / loss
                    symbol_data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                    
                    # RSI momentum
                    symbol_data[f'rsi_{period}_change'] = symbol_data[f'rsi_{period}'].diff(3)
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}")
            
            # MACD
            try:
                macd_config = self.lookback_periods.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
                
                symbol_data['ema_fast'] = symbol_data['close'].ewm(span=macd_config['fast'], adjust=False).mean()
                symbol_data['ema_slow'] = symbol_data['close'].ewm(span=macd_config['slow'], adjust=False).mean()
                symbol_data['macd'] = symbol_data['ema_fast'] - symbol_data['ema_slow']
                symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=macd_config['signal'], adjust=False).mean()
                symbol_data['macd_hist'] = symbol_data['macd'] - symbol_data['macd_signal']
                
                # MACD momentum
                symbol_data['macd_hist_change'] = symbol_data['macd_hist'].diff(1)
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")
            
            # Stochastic oscillator
            try:
                for period in [14]:
                    symbol_data[f'stoch_k_{period}'] = 100 * ((symbol_data['close'] - symbol_data['low'].rolling(period).min()) / 
                                                        (symbol_data['high'].rolling(period).max() - symbol_data['low'].rolling(period).min()))
                    symbol_data[f'stoch_d_{period}'] = symbol_data[f'stoch_k_{period}'].rolling(3).mean()
                    
                    # Stochastic momentum
                    symbol_data[f'stoch_k_{period}_change'] = symbol_data[f'stoch_k_{period}'].diff(3)
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {str(e)}")
            
            # Average Directional Index (ADX)
            try:
                for period in self.lookback_periods.get('adx', [14]):
                    # Calculate True Range
                    symbol_data['tr'] = np.maximum(
                        symbol_data['high'] - symbol_data['low'],
                        np.maximum(
                            abs(symbol_data['high'] - symbol_data['close'].shift(1)),
                            abs(symbol_data['low'] - symbol_data['close'].shift(1))
                        )
                    )
                    
                    # Calculate Directional Movement
                    symbol_data['plus_dm'] = np.where(
                        (symbol_data['high'] - symbol_data['high'].shift(1)) > (symbol_data['low'].shift(1) - symbol_data['low']),
                        np.maximum(symbol_data['high'] - symbol_data['high'].shift(1), 0),
                        0
                    )
                    
                    symbol_data['minus_dm'] = np.where(
                        (symbol_data['low'].shift(1) - symbol_data['low']) > (symbol_data['high'] - symbol_data['high'].shift(1)),
                        np.maximum(symbol_data['low'].shift(1) - symbol_data['low'], 0),
                        0
                    )
                    
                    # Calculate smoothed averages
                    symbol_data[f'atr_{period}'] = symbol_data['tr'].rolling(period).mean()
                    symbol_data[f'plus_di_{period}'] = 100 * (symbol_data['plus_dm'].rolling(period).mean() / symbol_data[f'atr_{period}'])
                    symbol_data[f'minus_di_{period}'] = 100 * (symbol_data['minus_dm'].rolling(period).mean() / symbol_data[f'atr_{period}'])
                    
                    # Calculate ADX
                    symbol_data[f'dx_{period}'] = 100 * abs(symbol_data[f'plus_di_{period}'] - symbol_data[f'minus_di_{period}']) / (symbol_data[f'plus_di_{period}'] + symbol_data[f'minus_di_{period}'])
                    symbol_data[f'adx_{period}'] = symbol_data[f'dx_{period}'].rolling(period).mean()
            except Exception as e:
                logger.warning(f"Error calculating ADX: {str(e)}")
            
            # Commodity Channel Index (CCI)
            try:
                for period in [20]:
                    # Calculate typical price
                    symbol_data['tp'] = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
                    
                    # Calculate CCI
                    symbol_data[f'cci_{period}'] = (symbol_data['tp'] - symbol_data['tp'].rolling(period).mean()) / (0.015 * symbol_data['tp'].rolling(period).std())
            except Exception as e:
                logger.warning(f"Error calculating CCI: {str(e)}")
            
            # Ichimoku Cloud
            try:
                # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
                symbol_data['tenkan_sen'] = (symbol_data['high'].rolling(9).max() + symbol_data['low'].rolling(9).min()) / 2
                
                # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
                symbol_data['kijun_sen'] = (symbol_data['high'].rolling(26).max() + symbol_data['low'].rolling(26).min()) / 2
                
                # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2
                symbol_data['senkou_span_a'] = ((symbol_data['tenkan_sen'] + symbol_data['kijun_sen']) / 2).shift(26)
                
                # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
                symbol_data['senkou_span_b'] = ((symbol_data['high'].rolling(52).max() + symbol_data['low'].rolling(52).min()) / 2).shift(26)
                
                # Chikou Span (Lagging Span): Close price shifted back 26 periods
                symbol_data['chikou_span'] = symbol_data['close'].shift(-26)
                
                # Ichimoku signals
                symbol_data['ichimoku_bullish'] = ((symbol_data['close'] > symbol_data['senkou_span_a']) & 
                                              (symbol_data['close'] > symbol_data['senkou_span_b']) & 
                                              (symbol_data['tenkan_sen'] > symbol_data['kijun_sen']))
                
                symbol_data['ichimoku_bearish'] = ((symbol_data['close'] < symbol_data['senkou_span_a']) & 
                                              (symbol_data['close'] < symbol_data['senkou_span_b']) & 
                                              (symbol_data['tenkan_sen'] < symbol_data['kijun_sen']))
            except Exception as e:
                logger.warning(f"Error calculating Ichimoku: {str(e)}")
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_market_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market context features.
        
        Args:
            data: DataFrame with OHLCV data and market data
            
        Returns:
            DataFrame with market context features added
        """
        logger.info("Generating market context features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Calculate daily returns if not present
            if 'daily_return' not in symbol_data.columns:
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
            
            # Relative strength vs market
            for window in [5, 10, 20, 60]:
                symbol_data[f'rs_vs_market_{window}'] = (
                    (1 + symbol_data['daily_return']).rolling(window).prod() / 
                    (1 + symbol_data['market_return']).rolling(window).prod() - 1
                )
            
            # Beta calculation
            try:
                # Calculate rolling covariance and market variance
                symbol_data['rolling_cov'] = symbol_data['daily_return'].rolling(60).cov(symbol_data['market_return'])
                symbol_data['rolling_market_var'] = symbol_data['market_return'].rolling(60).var()
                
                # Calculate beta
                symbol_data['beta_60d'] = symbol_data['rolling_cov'] / symbol_data['rolling_market_var']
                
                # Beta trend
                symbol_data['beta_trend'] = symbol_data['beta_60d'] / symbol_data['beta_60d'].rolling(10).mean()
                
                # Clean up intermediate columns
                symbol_data = symbol_data.drop(['rolling_cov', 'rolling_market_var'], axis=1)
            except Exception as e:
                logger.warning(f"Error calculating beta: {str(e)}")
            
            # Sector correlations (if available)
            if 'sector_return' in symbol_data.columns:
                # Correlation with sector
                symbol_data['sector_correlation'] = symbol_data['daily_return'].rolling(20).corr(symbol_data['sector_return'])
                
                # Relative strength vs sector
                for window in [5, 10, 20]:
                    symbol_data[f'rs_vs_sector_{window}'] = (
                        (1 + symbol_data['daily_return']).rolling(window).prod() / 
                        (1 + symbol_data['sector_return']).rolling(window).prod() - 1
                    )
            
            # Market regime features (if available)
            if 'market_regime' in symbol_data.columns:
                # Performance in different regimes
                symbol_data['return_in_current_regime'] = symbol_data.groupby('market_regime')['daily_return'].transform(
                    lambda x: x.rolling(10).mean()
                )
                
                # Regime persistence
                symbol_data['regime_persistence'] = symbol_data.groupby('market_regime').cumcount() + 1
            
            # Market breadth features (if available)
            for breadth_indicator in ['advance_decline', 'new_highs_lows', 'vix']:
                if breadth_indicator in symbol_data.columns:
                    # Add raw indicator
                    symbol_data[f'{breadth_indicator}_ma10'] = symbol_data[breadth_indicator].rolling(10).mean()
                    symbol_data[f'{breadth_indicator}_ratio'] = symbol_data[breadth_indicator] / symbol_data[f'{breadth_indicator}_ma10']
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _generate_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate sentiment related features.
        
        Args:
            data: DataFrame with OHLCV data and sentiment data
            
        Returns:
            DataFrame with sentiment features added
        """
        logger.info("Generating sentiment features")
        df = data.copy()
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Ensure data is sorted by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # News sentiment features
            if 'news_sentiment' in symbol_data.columns:
                # Smoothed sentiment
                symbol_data['sentiment_ma_3d'] = symbol_data['news_sentiment'].rolling(3).mean()
                symbol_data['sentiment_ma_7d'] = symbol_data['news_sentiment'].rolling(7).mean()
                
                # Sentiment change
                symbol_data['sentiment_change'] = symbol_data['sentiment_ma_3d'] - symbol_data['sentiment_ma_3d'].shift(3)
                
                # Sentiment volatility
                symbol_data['sentiment_volatility'] = symbol_data['news_sentiment'].rolling(7).std()
                
                # Sentiment trend
                symbol_data['sentiment_trend'] = symbol_data['sentiment_ma_3d'] / symbol_data['sentiment_ma_7d']
            
            # Social media sentiment (if available)
            if 'social_sentiment' in symbol_data.columns:
                # Smoothed social sentiment
                symbol_data['social_sentiment_ma_3d'] = symbol_data['social_sentiment'].rolling(3).mean()
                
                # Combined sentiment score
                if 'news_sentiment' in symbol_data.columns:
                    # Weight news more than social by default
                    news_weight = self.sentiment_config.get('news_weight', 0.7)
                    social_weight = self.sentiment_config.get('social_weight', 0.3)
                    
                    symbol_data['combined_sentiment'] = (
                        news_weight * symbol_data['sentiment_ma_3d'] +
                        social_weight * symbol_data['social_sentiment_ma_3d']
                    )
            
            # Options flow sentiment (if available)
            if 'options_sentiment' in symbol_data.columns:
                # Smoothed options sentiment
                symbol_data['options_sentiment_ma_3d'] = symbol_data['options_sentiment'].rolling(3).mean()
                
                # Options sentiment change
                symbol_data['options_sentiment_change'] = symbol_data['options_sentiment'] - symbol_data['options_sentiment'].shift(3)
                
                # Add to combined sentiment if available
                if 'combined_sentiment' in symbol_data.columns:
                    # Add options sentiment with lower weight
                    options_weight = self.sentiment_config.get('options_weight', 0.2)
                    
                    # Normalize weights
                    total_weight = news_weight + social_weight + options_weight
                    news_weight = news_weight / total_weight
                    social_weight = social_weight / total_weight
                    options_weight = options_weight / total_weight
                    
                    symbol_data['combined_sentiment'] = (
                        news_weight * symbol_data['sentiment_ma_3d'] +
                        social_weight * symbol_data['social_sentiment_ma_3d'] +
                        options_weight * symbol_data['options_sentiment_ma_3d']
                    )
            
            # Try to use ML training engine's sentiment features if available
            if SENTIMENT_FEATURES_AVAILABLE:
                try:
                    # Get news text if available
                    if 'news_text' in symbol_data.columns:
                        # Get unique news texts for this symbol
                        news_texts = symbol_data['news_text'].dropna().unique().tolist()
                        
                        if news_texts:
                            # Generate sentiment features using ML training engine
                            sentiment_features = generate_sentiment_features(news_texts)
                            
                            # Map sentiment features to dataframe
                            # This is a simplified approach - in a real system, you'd need to match texts to timestamps
                            if sentiment_features and len(sentiment_features) > 0:
                                # Use the most recent sentiment features
                                latest_features = sentiment_features[0]
                                
                                for key, value in latest_features.items():
                                    symbol_data[f'ml_{key}'] = value
                                
                                logger.info(f"Added ML-based sentiment features for {symbol}")
                except Exception as e:
                    logger.warning(f"Error generating ML-based sentiment features: {str(e)}")
            
            result_dfs.append(symbol_data)
        
        return pd.concat(result_dfs, ignore_index=True)

# Function to get feature names for a specific group
def get_feature_names(feature_generator: StockSelectionFeatures, group: str = None) -> List[str]:
    """
    Get a list of feature names for a specific group or all features.
    
    Args:
        feature_generator: StockSelectionFeatures instance
        group: Optional feature group name
        
    Returns:
        List of feature names
    """
    # Get all possible features (placeholder implementation)
    all_features = {
        'price_action': [
            'return_5m', 'return_10m', 'return_20m', 'return_60m',
            'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_200',
            'close_to_ma_5', 'close_to_ma_10', 'close_to_ma_20', 'close_to_ma_50', 'close_to_ma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'close_to_ema_5', 'close_to_ema_10', 'close_to_ema_20', 'close_to_ema_50',
            'ma_5_10_ratio', 'ma_10_20_ratio', 'ma_20_50_ratio',
            'dist_to_high_10d', 'dist_to_low_10d',
            'overnight_gap',
            'pattern_hammer', 'pattern_engulfing', 'pattern_morning_star', 
            'pattern_shooting_star', 'pattern_evening_star', 'pattern_doji',
            'daily_range', 'daily_range_ma5',
            'price_velocity', 'price_acceleration'
        ],
        'volume': [
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
            'relative_volume', 'volume_trend_3d',
            'obv', 'obv_ratio_10d', 
            'vwap', 'close_to_vwap',
            'volume_spike',
            'price_volume_correlation',
            'mfi_14'
        ],
        'volatility': [
            'tr', 'atr_14', 'atr_ratio_14',
            'volatility_10', 'volatility_20', 'volatility_60',
            'volatility_trend',
            'bb_middle_20', 'bb_std_20', 'bb_upper_20', 'bb_lower_20', 'bb_width_20', 'bb_position_20',
            'kc_middle_14', 'kc_upper_14', 'kc_lower_14', 'kc_width_14',
            'squeeze_14',
            'return_volatility_ratio'
        ],
        'technical': [
            'rsi_7', 'rsi_14', 'rsi_21',
            'rsi_7_change', 'rsi_14_change', 'rsi_21_change',
            'macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
            'stoch_k_14', 'stoch_d_14', 'stoch_k_14_change',
            'plus_di_14', 'minus_di_14', 'adx_14',
            'cci_20',
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
            'ichimoku_bullish', 'ichimoku_bearish'
        ],
        'market_context': [
            'rs_vs_market_5', 'rs_vs_market_10', 'rs_vs_market_20', 'rs_vs_market_60',
            'beta_60d', 'beta_trend',
            'sector_correlation',
            'rs_vs_sector_5', 'rs_vs_sector_10', 'rs_vs_sector_20',
            'return_in_current_regime', 'regime_persistence',
            'advance_decline_ma10', 'advance_decline_ratio',
            'new_highs_lows_ma10', 'new_highs_lows_ratio',
            'vix_ma10', 'vix_ratio'
        ],
        'sentiment': [
            'sentiment_ma_3d', 'sentiment_ma_7d',
            'sentiment_change', 'sentiment_volatility', 'sentiment_trend',
            'social_sentiment_ma_3d',
            'combined_sentiment',
            'options_sentiment_ma_3d', 'options_sentiment_change',
            'ml_sentiment_score', 'ml_sentiment_magnitude', 'ml_sentiment_label'
        ]
    }
    
    # Return features for a specific group or all features
    if group is not None and group in all_features:
        return all_features[group]
    elif group is None:
        # Combine all features
        combined_features = []
        for feature_list in all_features.values():
            combined_features.extend(feature_list)
        return combined_features
    else:
        logger.warning(f"Unknown feature group: {group}")
        return []