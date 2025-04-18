#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineer Module
----------------------
This module generates features for machine learning models from various data sources.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.logging import setup_logger
from feature_store.feature_store import FeatureStore

# Try to import sentiment analyzers
try:
    from nlp.sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYZER_AVAILABLE = False

try:
    from nlp.finbert_integration import analyzer as rule_based_analyzer
    RULE_BASED_ANALYZER_AVAILABLE = True
except ImportError:
    RULE_BASED_ANALYZER_AVAILABLE = False

# Setup logging
logger = setup_logger('feature_engineer', category='data')


class FeatureEngineer:
    """
    Generates features for machine learning models from various data sources.
    
    This class is responsible for:
    1. Generating technical indicators from price data
    2. Extracting sentiment features from news and social data
    3. Creating options flow features
    4. Normalizing and scaling features
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logger
        self.config = config or {}

        # Initialize Feature Store
        self.feature_store = FeatureStore(self.config.get('feature_store_config'))

        # Technical indicator parameters
        self.short_window = self.config.get('short_window', 5)
        self.medium_window = self.config.get('medium_window', 20)
        self.long_window = self.config.get('long_window', 50)
        self.very_long_window = self.config.get('very_long_window', 200)
        
        # Volatility parameters
        self.atr_period = self.config.get('atr_period', 14)
        self.bollinger_period = self.config.get('bollinger_period', 20)
        self.bollinger_std = self.config.get('bollinger_std', 2.0)
        
        # Volume parameters
        self.volume_ma_period = self.config.get('volume_ma_period', 20)
        self.obv_ma_period = self.config.get('obv_ma_period', 10)
        
        # Sentiment parameters
        self.sentiment_decay = self.config.get('sentiment_decay', 0.9)
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.05)
        
        # Initialize sentiment analyzers if available
        self.sentiment_analyzer = None
        self.rule_based_analyzer = None
        
        if SENTIMENT_ANALYZER_AVAILABLE:
            try:
                # Check if rule-based implementation is preferred
                use_rule_based = self.config.get('use_rule_based', True)
                
                # Initialize the appropriate analyzer
                self.sentiment_analyzer = SentimentAnalyzer(
                    use_finbert=self.config.get('use_finbert', False),
                    use_rule_based=use_rule_based
                )
                self.logger.info("Initialized SentimentAnalyzer")
            except Exception as e:
                self.logger.error(f"Failed to initialize SentimentAnalyzer: {str(e)}")
        
        if RULE_BASED_ANALYZER_AVAILABLE:
            self.rule_based_analyzer = rule_based_analyzer
            self.logger.info("Initialized rule-based sentiment analyzer")
        
        # Options parameters
        self.options_volume_threshold = self.config.get('options_volume_threshold', 100)
        self.unusual_score_threshold = self.config.get('unusual_score_threshold', 80)
        
        self.logger.info("FeatureEngineer initialized")
    
    def generate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate features from price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary of features
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for price feature generation")
            return {}
        
        # Ensure DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            self.logger.warning(f"DataFrame missing required columns: {required_columns}")
            return {}
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure numeric types
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop NaN values
        df = df.dropna(subset=['close'])
        
        if len(df) < self.medium_window:
            self.logger.warning(f"Not enough data points for feature generation: {len(df)} < {self.medium_window}")
            return {}
        
        features = {}
        
        # Price momentum features
        features.update(self._generate_momentum_features(df))
        
        # Volatility features
        features.update(self._generate_volatility_features(df))
        
        # Volume features
        features.update(self._generate_volume_features(df))
        
        # Pattern recognition features
        features.update(self._generate_pattern_features(df))
        
        # Register generated features in the feature store
        for feature_name, feature_value in features.items():
            # Create basic metadata for now, enhance later with more details
            metadata = {
                'source_data': 'price_data',
                'generation_method': 'technical_indicator',
                'value': feature_value # Store the latest value as part of metadata for quick access
            }
            self.feature_store.register_feature(
                feature_name=feature_name,
                feature_metadata=metadata,
                version="latest", # Or generate a version based on data timestamp/hash
                description=f"Generated from price data: {feature_name}"
            )
            # Log lineage if applicable (e.g., SMA is derived from 'close')
            if 'sma' in feature_name or 'ema' in feature_name or 'macd' in feature_name or 'rsi' in feature_name or 'stoch' in feature_name or 'return' in feature_name or 'roc' in feature_name:
                 self.feature_store.log_feature_lineage(
                     feature_name=feature_name,
                     version="latest",
                     source_features=[{'name': 'close', 'version': 'raw_price'}], # Assuming 'close' is a registered raw feature
                     transformation=f"Calculated {feature_name} from close price"
                 )

        return features

    def _generate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate momentum-based features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary of momentum features
        """
        features = {}
        
        # Simple moving averages
        try:
            df['sma_short'] = df.ta.sma(length=self.short_window, close='close')
            df['sma_medium'] = df.ta.sma(length=self.medium_window, close='close')
            df['sma_long'] = df.ta.sma(length=self.long_window, close='close')
            
            # SMA-based features
            last_close = df['close'].iloc[-1]
            features['mom_sma_short_dist'] = (last_close / df['sma_short'].iloc[-1]) - 1
            features['mom_sma_medium_dist'] = (last_close / df['sma_medium'].iloc[-1]) - 1
            features['mom_sma_long_dist'] = (last_close / df['sma_long'].iloc[-1]) - 1
            
            # SMA crossovers
            features['mom_sma_short_above_medium'] = 1 if df['sma_short'].iloc[-1] > df['sma_medium'].iloc[-1] else -1
            features['mom_sma_medium_above_long'] = 1 if df['sma_medium'].iloc[-1] > df['sma_long'].iloc[-1] else -1
            
            # SMA slopes
            features['mom_sma_short_slope'] = (df['sma_short'].iloc[-1] / df['sma_short'].iloc[-5]) - 1
            features['mom_sma_medium_slope'] = (df['sma_medium'].iloc[-1] / df['sma_medium'].iloc[-5]) - 1
        except Exception as e:
            self.logger.warning(f"Error calculating SMA features: {str(e)}")
        
        # Exponential moving averages
        try:
            df['ema_short'] = df.ta.ema(length=self.short_window, close='close')
            df['ema_medium'] = df.ta.ema(length=self.medium_window, close='close')
            df['ema_long'] = df.ta.ema(length=self.long_window, close='close')
            
            # EMA-based features
            features['mom_ema_short_dist'] = (last_close / df['ema_short'].iloc[-1]) - 1
            features['mom_ema_medium_dist'] = (last_close / df['ema_medium'].iloc[-1]) - 1
            features['mom_ema_long_dist'] = (last_close / df['ema_long'].iloc[-1]) - 1
            
            # EMA crossovers
            features['mom_ema_short_above_medium'] = 1 if df['ema_short'].iloc[-1] > df['ema_medium'].iloc[-1] else -1
            features['mom_ema_medium_above_long'] = 1 if df['ema_medium'].iloc[-1] > df['ema_long'].iloc[-1] else -1
        except Exception as e:
            self.logger.warning(f"Error calculating EMA features: {str(e)}")
        
        # MACD
        try:
            macd_result = df.ta.macd(close='close', fast=12, slow=26, signal=9)
            macd = macd_result['MACD_12_26_9']
            macd_signal = macd_result['MACDs_12_26_9']
            macd_hist = macd_result['MACDh_12_26_9']
            
            features['mom_macd'] = macd.iloc[-1]
            features['mom_macd_signal'] = macd_signal.iloc[-1]
            features['mom_macd_hist'] = macd_hist.iloc[-1]
            features['mom_macd_above_signal'] = 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1
            features['mom_macd_hist_change'] = macd_hist.iloc[-1] - macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        except Exception as e:
            self.logger.warning(f"Error calculating MACD features: {str(e)}")
        
        # RSI
        try:
            df['rsi'] = df.ta.rsi(length=14, close='close')
            features['mom_rsi'] = df['rsi'].iloc[-1]
            features['mom_rsi_slope'] = df['rsi'].iloc[-1] - df['rsi'].iloc[-5] if len(df['rsi']) > 5 else 0
            features['mom_rsi_overbought'] = 1 if df['rsi'].iloc[-1] > 70 else 0
            features['mom_rsi_oversold'] = 1 if df['rsi'].iloc[-1] < 30 else 0
        except Exception as e:
            self.logger.warning(f"Error calculating RSI features: {str(e)}")
        
        # Stochastic Oscillator
        try:
            stoch_result = df.ta.stoch(high='high', low='low', close='close', k=5, d=3, smooth_k=3)
            slowk = stoch_result['STOCHk_5_3_3']
            slowd = stoch_result['STOCHd_5_3_3']
            
            features['mom_stoch_k'] = slowk.iloc[-1]
            features['mom_stoch_d'] = slowd.iloc[-1]
            features['mom_stoch_above_80'] = 1 if slowk.iloc[-1] > 80 else 0
            features['mom_stoch_below_20'] = 1 if slowk.iloc[-1] < 20 else 0
            features['mom_stoch_k_above_d'] = 1 if slowk.iloc[-1] > slowd.iloc[-1] else -1
        except Exception as e:
            self.logger.warning(f"Error calculating Stochastic features: {str(e)}")
        
        # Price momentum
        try:
            # Returns over different periods
            features['mom_return_1d'] = (df['close'].iloc[-1] / df['close'].iloc[-2]) - 1 if len(df) > 1 else 0
            features['mom_return_5d'] = (df['close'].iloc[-1] / df['close'].iloc[-6]) - 1 if len(df) > 5 else 0
            features['mom_return_10d'] = (df['close'].iloc[-1] / df['close'].iloc[-11]) - 1 if len(df) > 10 else 0
            features['mom_return_20d'] = (df['close'].iloc[-1] / df['close'].iloc[-21]) - 1 if len(df) > 20 else 0
            
            # Rate of change
            df['roc'] = df.ta.roc(length=10, close='close')
            features['mom_roc'] = df['roc'].iloc[-1]
        except Exception as e:
            self.logger.warning(f"Error calculating momentum features: {str(e)}")
        
        return features
        # No need to register/log lineage here, done in the calling method
        return features

    def _generate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate volatility-based features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary of volatility features
        """
        features = {}
        
        # ATR (Average True Range)
        try:
            df['atr'] = df.ta.atr(length=self.atr_period, high='high', low='low', close='close')
            last_close = df['close'].iloc[-1]
            features['vol_atr'] = df['atr'].iloc[-1]
            features['vol_atr_pct'] = df['atr'].iloc[-1] / last_close
        except Exception as e:
            self.logger.warning(f"Error calculating ATR features: {str(e)}")
        
        # Bollinger Bands
        try:
            bbands = df.ta.bbands(length=self.bollinger_period, std=self.bollinger_std, close='close')
            upper = bbands[f'BBU_{self.bollinger_period}_{self.bollinger_std}']
            middle = bbands[f'BBM_{self.bollinger_period}_{self.bollinger_std}']
            lower = bbands[f'BBL_{self.bollinger_period}_{self.bollinger_std}']
            
            features['vol_bb_width'] = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            features['vol_bb_position'] = (last_close - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            features['vol_bb_upper_dist'] = (upper.iloc[-1] / last_close) - 1
            features['vol_bb_lower_dist'] = (last_close / lower.iloc[-1]) - 1
        except Exception as e:
            self.logger.warning(f"Error calculating Bollinger Band features: {str(e)}")
        
        # Historical volatility
        try:
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate rolling standard deviation of returns
            df['vol_5d'] = df['returns'].rolling(window=5).std()
            df['vol_20d'] = df['returns'].rolling(window=20).std()
            
            features['vol_5d'] = df['vol_5d'].iloc[-1]
            features['vol_20d'] = df['vol_20d'].iloc[-1]
            features['vol_ratio'] = df['vol_5d'].iloc[-1] / df['vol_20d'].iloc[-1] if df['vol_20d'].iloc[-1] > 0 else 1
        except Exception as e:
            self.logger.warning(f"Error calculating volatility features: {str(e)}")
        
        # Keltner Channels
        try:
            keltner = df.ta.kc(high='high', low='low', close='close', length=20, scalar=2)
            df['keltner_middle'] = keltner[f'KC_20_2_EMA']
            df['keltner_upper'] = keltner[f'KCU_20_2']
            df['keltner_lower'] = keltner[f'KCL_20_2']
            
            features['vol_keltner_width'] = (df['keltner_upper'].iloc[-1] - df['keltner_lower'].iloc[-1]) / df['keltner_middle'].iloc[-1]
            features['vol_keltner_position'] = (last_close - df['keltner_lower'].iloc[-1]) / (df['keltner_upper'].iloc[-1] - df['keltner_lower'].iloc[-1])
        except Exception as e:
            self.logger.warning(f"Error calculating Keltner Channel features: {str(e)}")
        
        # High-Low Range
        try:
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            features['vol_daily_range'] = df['daily_range'].iloc[-1]
            features['vol_daily_range_ma5'] = df['daily_range'].rolling(window=5).mean().iloc[-1]
        except Exception as e:
            self.logger.warning(f"Error calculating range features: {str(e)}")
        
        return features
        # No need to register/log lineage here, done in the calling method
        return features

    def _generate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate volume-based features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary of volume features
        """
        features = {}
        
        # Volume moving averages
        try:
            df['volume_ma'] = df.ta.sma(length=self.volume_ma_period, close='volume')
            features['volume_ratio'] = df['volume'].iloc[-1] / df['volume_ma'].iloc[-1]
            features['volume_trend'] = (df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()) - 1
        except Exception as e:
            self.logger.warning(f"Error calculating volume MA features: {str(e)}")
        
        # On-Balance Volume (OBV)
        try:
            df['obv'] = df.ta.obv(close='close', volume='volume')
            df['obv_ma'] = df.ta.sma(length=self.obv_ma_period, close='obv')
            
            features['volume_obv_slope'] = (df['obv'].iloc[-1] - df['obv'].iloc[-5]) / df['obv'].iloc[-5] if df['obv'].iloc[-5] != 0 else 0
            features['volume_obv_above_ma'] = 1 if df['obv'].iloc[-1] > df['obv_ma'].iloc[-1] else -1
        except Exception as e:
            self.logger.warning(f"Error calculating OBV features: {str(e)}")
        
        # Chaikin Money Flow
        try:
            df['cmf'] = df.ta.adosc(high='high', low='low', close='close', volume='volume', fast=3, slow=10)
            features['volume_cmf'] = df['cmf'].iloc[-1]
            features['volume_cmf_slope'] = df['cmf'].iloc[-1] - df['cmf'].iloc[-5] if len(df['cmf']) > 5 else 0
        except Exception as e:
            self.logger.warning(f"Error calculating CMF features: {str(e)}")
        
        # Volume-price relationship
        try:
            # Up/down volume
            df['up_day'] = (df['close'] > df['close'].shift(1)).astype(int)
            df['down_day'] = (df['close'] < df['close'].shift(1)).astype(int)
            
            df['up_volume'] = df['volume'] * df['up_day']
            df['down_volume'] = df['volume'] * df['down_day']
            
            up_vol_5d = df['up_volume'].rolling(window=5).sum().iloc[-1]
            down_vol_5d = df['down_volume'].rolling(window=5).sum().iloc[-1]
            
            features['volume_up_down_ratio'] = up_vol_5d / down_vol_5d if down_vol_5d > 0 else 1
        except Exception as e:
            self.logger.warning(f"Error calculating volume-price features: {str(e)}")
        
        # VWAP
        try:
            # Calculate VWAP if not already in the DataFrame
            if 'vwap' not in df.columns:
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
                df['vwap'] = df['vwap'].cumsum() / df['volume'].cumsum()
            
            features['vwap_distance'] = (df['close'].iloc[-1] / df['vwap'].iloc[-1]) - 1
        except Exception as e:
            self.logger.warning(f"Error calculating VWAP features: {str(e)}")
        
        return features
        # No need to register/log lineage here, done in the calling method
        return features

    def _generate_pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate pattern recognition features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary of pattern features
        """
        features = {}
        
        # Candlestick patterns
        try:
            # Bullish patterns
            hammer = df.ta.cdl_pattern(name="hammer", open_='open', high='high', low='low', close='close')
            features['pattern_hammer'] = hammer.iloc[-1] / 100 if not hammer.empty else 0
            
            morning_star = df.ta.cdl_pattern(name="morningstar", open_='open', high='high', low='low', close='close')
            features['pattern_morning_star'] = morning_star.iloc[-1] / 100 if not morning_star.empty else 0
            
            engulfing = df.ta.cdl_pattern(name="engulfing", open_='open', high='high', low='low', close='close')
            features['pattern_engulfing_bullish'] = engulfing.iloc[-1] / 100 if not engulfing.empty else 0
            
            # Bearish patterns
            hanging_man = df.ta.cdl_pattern(name="hangingman", open_='open', high='high', low='low', close='close')
            features['pattern_hanging_man'] = hanging_man.iloc[-1] / 100 if not hanging_man.empty else 0
            
            evening_star = df.ta.cdl_pattern(name="eveningstar", open_='open', high='high', low='low', close='close')
            features['pattern_evening_star'] = evening_star.iloc[-1] / 100 if not evening_star.empty else 0
            
            # For bearish engulfing, we use the same pattern but negate it when bearish
            features['pattern_engulfing_bearish'] = -engulfing.iloc[-1] / 100 if not engulfing.empty else 0
            
            # Doji patterns
            doji = df.ta.cdl_pattern(name="doji", open_='open', high='high', low='low', close='close')
            features['pattern_doji'] = doji.iloc[-1] / 100 if not doji.empty else 0
        except Exception as e:
            self.logger.warning(f"Error calculating candlestick pattern features: {str(e)}")
        
        # Support/Resistance
        try:
            # Find recent highs and lows
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            
            last_close = df['close'].iloc[-1]
            
            # Distance to support/resistance
            features['pattern_resistance_dist'] = (recent_high / last_close) - 1
            features['pattern_support_dist'] = (last_close / recent_low) - 1
            
            # Breakout detection
            prev_high = df['high'].iloc[-40:-20].max()
            prev_low = df['low'].iloc[-40:-20].min()
            
            features['pattern_breakout_up'] = 1 if last_close > prev_high else 0
            features['pattern_breakout_down'] = 1 if last_close < prev_low else 0
        except Exception as e:
            self.logger.warning(f"Error calculating support/resistance features: {str(e)}")
        
        # Trend strength
        try:
            # ADX (Average Directional Index)
            adx_result = df.ta.adx(high='high', low='low', close='close', length=14)
            df['adx'] = adx_result[f'ADX_14']
            features['pattern_adx'] = df['adx'].iloc[-1]
            features['pattern_strong_trend'] = 1 if df['adx'].iloc[-1] > 25 else 0
        except Exception as e:
            self.logger.warning(f"Error calculating trend strength features: {str(e)}")
        
        return features
    
        # No need to register/log lineage here, done in the calling method
        return features

    def generate_news_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate features from news data.
        
        Args:
            df: DataFrame with news data
            
        Returns:
            Dictionary of features
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for news feature generation")
            return {}
        
        # Check if we need to analyze the sentiment first
        if 'text' in df.columns and 'sentiment_score' not in df.columns:
            self.logger.info("Analyzing sentiment for news data")
            
            # Use available sentiment analyzer
            if self.sentiment_analyzer:
                df['sentiment_analysis'] = df['text'].apply(lambda x: self.sentiment_analyzer.analyze(x))
                df['sentiment_score'] = df['sentiment_analysis'].apply(lambda x: x['score'])
                df['sentiment_magnitude'] = df['sentiment_analysis'].apply(lambda x: x['magnitude'])
                df['sentiment_classification'] = df['sentiment_analysis'].apply(lambda x: x['classification'])
            elif self.rule_based_analyzer:
                df['sentiment_analysis'] = df['text'].apply(lambda x: self.rule_based_analyzer.analyze_sentiment(x))
                df['sentiment_score'] = df['sentiment_analysis'].apply(lambda x: x['normalized_score'])
                df['sentiment_magnitude'] = df['sentiment_analysis'].apply(lambda x: abs(x['normalized_score']))
                df['sentiment_classification'] = df['sentiment_analysis'].apply(lambda x: 'bullish' if x['sentiment'] == 'positive'
                                                                              else ('bearish' if x['sentiment'] == 'negative' else 'neutral'))
        
        # Ensure DataFrame has required columns
        required_columns = ['sentiment_score', 'sentiment_magnitude']
        if not all(col in df.columns for col in required_columns):
            self.logger.warning(f"DataFrame missing required columns: {required_columns}")
            return {}
        
        features = {}
        
        try:
            # Sort by time
            df = df.sort_index()
            
            # Calculate time-weighted sentiment
            now = df.index.max()
            df['days_ago'] = (now - df.index).total_seconds() / (24 * 3600)
            df['weight'] = np.exp(-df['days_ago'] * (1 - self.sentiment_decay))
            
            # Normalize weights
            total_weight = df['weight'].sum()
            if total_weight > 0:
                df['norm_weight'] = df['weight'] / total_weight
            else:
                df['norm_weight'] = 1.0 / len(df)
            
            # Calculate weighted sentiment
            weighted_sentiment = (df['sentiment_score'] * df['norm_weight']).sum()
            weighted_magnitude = (df['sentiment_magnitude'] * df['norm_weight']).sum()
            
            features['sentiment_score'] = weighted_sentiment
            features['sentiment_magnitude'] = weighted_magnitude
            features['sentiment_signal'] = weighted_sentiment * weighted_magnitude
            
            # Count articles by sentiment
            bullish_count = len(df[df['sentiment_score'] > self.sentiment_threshold])
            bearish_count = len(df[df['sentiment_score'] < -self.sentiment_threshold])
            neutral_count = len(df) - bullish_count - bearish_count
            
            features['bullish_count'] = bullish_count
            features['bearish_count'] = bearish_count
            features['neutral_count'] = neutral_count
            
            # Sentiment ratio
            if bearish_count > 0:
                features['bull_bear_ratio'] = bullish_count / bearish_count
            else:
                features['bull_bear_ratio'] = bullish_count if bullish_count > 0 else 1
            
            # Recent sentiment trend
            if len(df) >= 10:
                recent_df = df.iloc[-10:]
                older_df = df.iloc[-20:-10] if len(df) >= 20 else df.iloc[:-10]
                
                if not recent_df.empty and not older_df.empty:
                    recent_sentiment = recent_df['sentiment_score'].mean()
                    older_sentiment = older_df['sentiment_score'].mean()
                    features['sentiment_trend'] = recent_sentiment - older_sentiment
            
            # Volume of news
            features['news_volume'] = len(df)
            
            # Recent news volume
            one_day_ago = now - timedelta(days=1)
            features['recent_news_volume'] = len(df[df.index >= one_day_ago])
            
            # News volume trend
            if len(df) > 0:
                three_days_ago = now - timedelta(days=3)
                seven_days_ago = now - timedelta(days=7)
                
                recent_volume = len(df[df.index >= three_days_ago])
                older_volume = len(df[(df.index >= seven_days_ago) & (df.index < three_days_ago)])
                
                if older_volume > 0:
                    features['news_volume_trend'] = (recent_volume / older_volume) - 1
                else:
                    features['news_volume_trend'] = recent_volume if recent_volume > 0 else 0
        
        except Exception as e:
            self.logger.error(f"Error generating news features: {str(e)}")
        
        # Register generated features in the feature store
        for feature_name, feature_value in features.items():
            metadata = {
                'source_data': 'news_data',
                'generation_method': 'sentiment_analysis',
                'value': feature_value
            }
            self.feature_store.register_feature(
                feature_name=f"news_{feature_name}", # Prefix to avoid naming conflicts
                feature_metadata=metadata,
                version="latest",
                description=f"Generated from news data: {feature_name}"
            )
            # Log lineage (assuming raw news data is a registered feature)
            self.feature_store.log_feature_lineage(
                feature_name=f"news_{feature_name}",
                version="latest",
                source_features=[{'name': 'raw_news', 'version': 'latest'}],
                transformation=f"Calculated {feature_name} from raw news data"
            )

        return features

    def generate_social_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate features from social media data.
        
        Args:
            df: DataFrame with social data
            
        Returns:
            Dictionary of features
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for social feature generation")
            return {}
        
        # Check if we need to analyze the sentiment first
        if ('text' in df.columns or 'content' in df.columns) and 'sentiment_score' not in df.columns:
            self.logger.info("Analyzing sentiment for social media data")
            
            # Determine which column contains the text
            text_column = 'text' if 'text' in df.columns else 'content'
            
            # Use available sentiment analyzer
            if self.sentiment_analyzer:
                df['sentiment_analysis'] = df[text_column].apply(lambda x: self.sentiment_analyzer.analyze(x))
                df['sentiment_score'] = df['sentiment_analysis'].apply(lambda x: x['score'])
                df['sentiment_magnitude'] = df['sentiment_analysis'].apply(lambda x: x['magnitude'])
                df['sentiment_classification'] = df['sentiment_analysis'].apply(lambda x: x['classification'])
            elif self.rule_based_analyzer:
                df['sentiment_analysis'] = df[text_column].apply(lambda x: self.rule_based_analyzer.analyze_sentiment(x))
                df['sentiment_score'] = df['sentiment_analysis'].apply(lambda x: x['normalized_score'])
                df['sentiment_magnitude'] = df['sentiment_analysis'].apply(lambda x: abs(x['normalized_score']))
                df['sentiment_classification'] = df['sentiment_analysis'].apply(lambda x: 'bullish' if x['sentiment'] == 'positive'
                                                                              else ('bearish' if x['sentiment'] == 'negative' else 'neutral'))
        
        # Ensure DataFrame has required columns
        required_columns = ['sentiment_score', 'sentiment_magnitude']
        if not all(col in df.columns for col in required_columns):
            self.logger.warning(f"DataFrame missing required columns: {required_columns}")
            return {}
        
        features = {}
        
        try:
            # Sort by time
            df = df.sort_index()
            
            # Calculate time-weighted sentiment
            now = df.index.max()
            df['hours_ago'] = (now - df.index).total_seconds() / 3600
            df['weight'] = np.exp(-df['hours_ago'] * (1 - self.sentiment_decay))
            
            # Add score-based weight if available
            if 'score' in df.columns:
                # Normalize scores
                max_score = df['score'].max()
                if max_score > 0:
                    df['score_weight'] = df['score'] / max_score
                else:
                    df['score_weight'] = 1.0
                
                # Combine time and score weights
                df['weight'] = df['weight'] * (1 + df['score_weight'])
            
            # Normalize weights
            total_weight = df['weight'].sum()
            if total_weight > 0:
                df['norm_weight'] = df['weight'] / total_weight
            else:
                df['norm_weight'] = 1.0 / len(df)
            
            # Calculate weighted sentiment
            weighted_sentiment = (df['sentiment_score'] * df['norm_weight']).sum()
            weighted_magnitude = (df['sentiment_magnitude'] * df['norm_weight']).sum()
            
            features['sentiment_score'] = weighted_sentiment
            features['sentiment_magnitude'] = weighted_magnitude
            features['sentiment_signal'] = weighted_sentiment * weighted_magnitude
            
            # Count posts by sentiment
            bullish_count = len(df[df['sentiment_score'] > self.sentiment_threshold])
            bearish_count = len(df[df['sentiment_score'] < -self.sentiment_threshold])
            neutral_count = len(df) - bullish_count - bearish_count
            
            features['bullish_count'] = bullish_count
            features['bearish_count'] = bearish_count
            features['neutral_count'] = neutral_count
            
            # Sentiment ratio
            if bearish_count > 0:
                features['bull_bear_ratio'] = bullish_count / bearish_count
            else:
                features['bull_bear_ratio'] = bullish_count if bullish_count > 0 else 1
            
            # Recent sentiment trend
            if len(df) >= 10:
                recent_df = df.iloc[-10:]
                older_df = df.iloc[-20:-10] if len(df) >= 20 else df.iloc[:-10]
                
                if not recent_df.empty and not older_df.empty:
                    recent_sentiment = recent_df['sentiment_score'].mean()
                    older_sentiment = older_df['sentiment_score'].mean()
                    features['sentiment_trend'] = recent_sentiment - older_sentiment
            
            # Volume of posts
            features['post_volume'] = len(df)
            
            # Recent post volume
            six_hours_ago = now - timedelta(hours=6)
            features['recent_post_volume'] = len(df[df.index >= six_hours_ago])
            
            # Post volume trend
            if len(df) > 0:
                twelve_hours_ago = now - timedelta(hours=12)
                day_ago = now - timedelta(days=1)
                
                recent_volume = len(df[df.index >= twelve_hours_ago])
                older_volume = len(df[(df.index >= day_ago) & (df.index < twelve_hours_ago)])
                
                if older_volume > 0:
                    features['post_volume_trend'] = (recent_volume / older_volume) - 1
                else:
                    features['post_volume_trend'] = recent_volume if recent_volume > 0 else 0
            
            # Platform-specific features
            if 'platform' in df.columns:
                # Reddit-specific features
                reddit_df = df[df['platform'] == 'reddit']
                if not reddit_df.empty:
                    features['reddit_post_count'] = len(reddit_df)
                    features['reddit_sentiment'] = reddit_df['sentiment_score'].mean()
                    
                    # Subreddit diversity
                    if 'subreddit' in df.columns:
                        features['subreddit_count'] = reddit_df['subreddit'].nunique()                        
                        # Most active subreddit
                        subreddit_counts = reddit_df['subreddit'].value_counts()
                        if not subreddit_counts.empty:
                            features['top_subreddit_ratio'] = subreddit_counts.iloc[0] / len(reddit_df)
        
        except Exception as e:
            self.logger.error(f"Error generating social features: {str(e)}")
        
        # Register generated features in the feature store
        for feature_name, feature_value in features.items():
            metadata = {
                'source_data': 'social_data',
                'generation_method': 'sentiment_analysis',
                'value': feature_value
            }
            self.feature_store.register_feature(
                feature_name=f"social_{feature_name}", # Prefix to avoid naming conflicts
                feature_metadata=metadata,
                version="latest",
                description=f"Generated from social data: {feature_name}"
            )
            # Log lineage (assuming raw social data is a registered feature)
            self.feature_store.log_feature_lineage(
                feature_name=f"social_{feature_name}",
                version="latest",
                source_features=[{'name': 'raw_social', 'version': 'latest'}],
                transformation=f"Calculated {feature_name} from raw social data"
            )

        return features

    def generate_options_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate features from options flow data.
        
        Args:
            df: DataFrame with options data
            
        Returns:
            Dictionary of features
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for options feature generation")
            return {}
        
        # Ensure DataFrame has required columns
        required_columns = ['option_type', 'volume', 'open_interest']
        if not all(col in df.columns for col in required_columns):
            self.logger.warning(f"DataFrame missing required columns: {required_columns}")
            return {}
        
        features = {}
        
        try:
            # Sort by time
            df = df.sort_index()
            
            # Filter for significant volume
            significant_df = df[df['volume'] >= self.options_volume_threshold]
            
            if significant_df.empty:
                self.logger.warning("No significant options volume found")
                return {}
            
            # Split by call/put
            calls_df = significant_df[significant_df['option_type'] == 'call']
            puts_df = significant_df[significant_df['option_type'] == 'put']
            
            # Volume metrics
            total_volume = significant_df['volume'].sum()
            call_volume = calls_df['volume'].sum() if not calls_df.empty else 0
            put_volume = puts_df['volume'].sum() if not puts_df.empty else 0
            
            features['total_options_volume'] = total_volume
            
            # Put/Call ratio
            if call_volume > 0:
                features['put_call_ratio'] = put_volume / call_volume
            else:
                features['put_call_ratio'] = 1.0 if put_volume > 0 else 0.5
            
            # Open interest metrics
            total_oi = significant_df['open_interest'].sum()
            call_oi = calls_df['open_interest'].sum() if not calls_df.empty else 0
            put_oi = puts_df['open_interest'].sum() if not puts_df.empty else 0
            
            features['total_open_interest'] = total_oi
            
            # Volume to open interest ratio
            if total_oi > 0:
                features['volume_to_oi_ratio'] = total_volume / total_oi
            else:
                features['volume_to_oi_ratio'] = 1.0
            
            # Unusual activity
            if 'unusual_score' in df.columns:
                unusual_df = significant_df[significant_df['unusual_score'] >= self.unusual_score_threshold]
                features['unusual_count'] = len(unusual_df)
                
                if not unusual_df.empty:
                    unusual_calls = unusual_df[unusual_df['option_type'] == 'call']
                    unusual_puts = unusual_df[unusual_df['option_type'] == 'put']
                    
                    features['unusual_call_count'] = len(unusual_calls)
                    features['unusual_put_count'] = len(unusual_puts)
                    
                    # Unusual sentiment
                    if 'sentiment_score' in unusual_df.columns:
                        features['unusual_sentiment'] = unusual_df['sentiment_score'].mean()
            
            # Implied volatility
            if 'implied_volatility' in df.columns:
                features['avg_implied_volatility'] = df['implied_volatility'].mean()
                
                # IV skew (puts vs calls)
                if not calls_df.empty and not puts_df.empty:
                    call_iv = calls_df['implied_volatility'].mean()
                    put_iv = puts_df['implied_volatility'].mean()
                    features['iv_skew'] = put_iv - call_iv
            
            # Premium flow
            if 'premium' in df.columns:
                total_premium = df['premium'].sum()
                call_premium = calls_df['premium'].sum() if not calls_df.empty else 0
                put_premium = puts_df['premium'].sum() if not puts_df.empty else 0
                
                features['total_premium'] = total_premium
                features['call_premium_ratio'] = call_premium / total_premium if total_premium > 0 else 0.5
                features['put_premium_ratio'] = put_premium / total_premium if total_premium > 0 else 0.5
            
            # Time-based metrics
            now = df.index.max()
            recent_df = df[df.index >= (now - timedelta(hours=4))]
            
            if not recent_df.empty:
                recent_volume = recent_df['volume'].sum()
                features['recent_options_volume'] = recent_volume
                features['recent_volume_ratio'] = recent_volume / total_volume if total_volume > 0 else 0
        
        except Exception as e:
            self.logger.error(f"Error generating options features: {str(e)}")
        
        # Register generated features in the feature store
        for feature_name, feature_value in features.items():
            metadata = {
                'source_data': 'options_flow_data',
                'generation_method': 'options_metrics',
                'value': feature_value
            }
            self.feature_store.register_feature(
                feature_name=f"options_{feature_name}", # Prefix to avoid naming conflicts
                feature_metadata=metadata,
                version="latest",
                description=f"Generated from options flow data: {feature_name}"
            )
            # Log lineage (assuming raw options data is a registered feature)
            self.feature_store.log_feature_lineage(
                feature_name=f"options_{feature_name}",
                version="latest",
                source_features=[{'name': 'raw_options', 'version': 'latest'}],
                transformation=f"Calculated {feature_name} from raw options data"
            )

        return features
