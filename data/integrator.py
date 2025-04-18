#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Integrator Module
---------------------
This module integrates data from multiple sources (price, news, social, options)
and generates features for machine learning models.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from utils.logging import setup_logger
from data.collectors.factory import get_collector
from data.database.timeseries_db import get_timescale_client
from data.database.redis_client import get_redis_client
from data.processors.feature_engineer import FeatureEngineer

# Try to import sentiment analysis functionality
try:
    from ml_training_engine_modified import generate_sentiment_features
    SENTIMENT_FEATURES_AVAILABLE = True
except ImportError:
    SENTIMENT_FEATURES_AVAILABLE = False

# Setup logging
logger = setup_logger('data_integrator', category='data')

# Initialize database clients
db_client = get_timescale_client()
redis_client = get_redis_client()


class DataIntegrator:
    """
    Integrates data from multiple sources and generates features for ML models.
    
    This class is responsible for:
    1. Collecting data from various sources (price, news, social, options)
    2. Preprocessing and cleaning the data
    3. Generating features for ML models
    4. Calculating composite scores for trading signals
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data integrator.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logger
        self.config = config or {}
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(config)
        
        # Feature group weights for composite scoring
        self.feature_weights = self.config.get('feature_weights', {
            'price_momentum': 0.25,
            'price_volatility': 0.15,
            'volume_profile': 0.10,
            'news_sentiment': 0.20,
            'social_sentiment': 0.15,
            'options_flow': 0.15
        })
        
        # Cache TTL (in seconds)
        self.cache_ttl = self.config.get('cache_ttl_seconds', 3600)  # 1 hour
        
        # Lookback periods for different data types
        self.price_lookback_days = self.config.get('price_lookback_days', 30)
        self.news_lookback_days = self.config.get('news_lookback_days', 7)
        self.social_lookback_days = self.config.get('social_lookback_days', 3)
        self.options_lookback_days = self.config.get('options_lookback_days', 5)
        
        self.logger.info("DataIntegrator initialized")
    
    async def collect_all_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all sources for a symbol.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            
        Returns:
            Dictionary of DataFrames with data from different sources
        """
        self.logger.info(f"Collecting all data for {symbol} from {start} to {end}")
        
        # Check cache first
        cache_key = f"all_data:{symbol}:{start.isoformat()}:{end.isoformat()}"
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            self.logger.info(f"Using cached data for {symbol}")
            return self._deserialize_dataframes(cached_data)
        
        # Collect data from different sources in parallel
        tasks = [
            self._collect_price_data(symbol, start, end),
            self._collect_news_data(symbol, start, end),
            self._collect_social_data(symbol, start, end),
            self._collect_options_data(symbol, start, end)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        data = {
            'price': results[0],
            'news': results[1],
            'social': results[2],
            'options': results[3]
        }
        
        # Cache results
        redis_client.set(cache_key, self._serialize_dataframes(data), self.cache_ttl)
        
        return data
    
    async def _collect_price_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Collect price data for a symbol.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with price data
        """
        try:
            # Get price data from database
            df = db_client.query_market_data(symbol, start, end)
            
            if df.empty:
                self.logger.warning(f"No price data found for {symbol}, trying collector")
                
                # Try to collect from API
                collector = get_collector('polygon')
                collector.set_symbol(symbol)
                collector.collection_mode = 'market_data'
                records = await collector.collect(start, end)
                
                if records:
                    # Convert to DataFrame
                    data = []
                    for record in records:
                        record_dict = record.model_dump()
                        data.append({
                            'time': record_dict.get('timestamp'),
                            'symbol': record_dict.get('symbol'),
                            'open': record_dict.get('open'),
                            'high': record_dict.get('high'),
                            'low': record_dict.get('low'),
                            'close': record_dict.get('close'),
                            'volume': record_dict.get('volume'),
                            'vwap': record_dict.get('vwap')
                        })
                    
                    df = pd.DataFrame(data)
            
            # Ensure DataFrame has the right columns
            if not df.empty:
                # Convert time to datetime if it's not already
                if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'])
                
                # Set time as index
                if 'time' in df.columns:
                    df = df.set_index('time')
                
                # Sort by time
                df = df.sort_index()
                
                self.logger.info(f"Collected {len(df)} price records for {symbol}")
            else:
                self.logger.warning(f"No price data available for {symbol}")
                df = pd.DataFrame()
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error collecting price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_news_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Collect news data for a symbol.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with news data
        """
        try:
            # Get news data from database
            df = db_client.query_news_data(symbol, start, end)
            
            if df.empty:
                self.logger.warning(f"No news data found for {symbol}, trying collector")
                
                # Try to collect from API
                collector = get_collector('polygon')
                collector.set_symbol(symbol)
                collector.collection_mode = 'news'
                records = await collector.collect(start, end)
                
                if records:
                    # Convert to DataFrame
                    data = []
                    for record in records:
                        record_dict = record.model_dump()
                        extended_data = record_dict.get('extended_data', {})
                        
                        data.append({
                            'time': record_dict.get('timestamp'),
                            'symbol': record_dict.get('symbol'),
                            'news_id': extended_data.get('news_id'),
                            'title': extended_data.get('title'),
                            'author': extended_data.get('author'),
                            'source': extended_data.get('source'),
                            'url': extended_data.get('url'),
                            'sentiment_score': record_dict.get('sentiment_score'),
                            'sentiment_magnitude': record_dict.get('sentiment_magnitude')
                        })
                    
                    df = pd.DataFrame(data)
            
            # Ensure DataFrame has the right columns
            if not df.empty:
                # Convert time to datetime if it's not already
                if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'])
                
                # Set time as index
                if 'time' in df.columns:
                    df = df.set_index('time')
                
                # Sort by time
                df = df.sort_index()
                
                self.logger.info(f"Collected {len(df)} news records for {symbol}")
            else:
                self.logger.warning(f"No news data available for {symbol}")
                df = pd.DataFrame()
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error collecting news data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_social_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Collect social media data for a symbol.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with social data
        """
        try:
            # Get social data from database
            df = db_client.query_social_data(symbol, start, end)
            
            if df.empty:
                self.logger.warning(f"No social data found for {symbol}, trying collector")
                
                # Try to collect from API
                collector = get_collector('reddit')
                collector.set_symbol(symbol)
                records = await collector.collect(start, end)
                
                if records:
                    # Convert to DataFrame
                    data = []
                    for record in records:
                        record_dict = record.model_dump()
                        extended_data = record_dict.get('extended_data', {})
                        
                        data.append({
                            'time': record_dict.get('timestamp'),
                            'symbol': record_dict.get('symbol'),
                            'source': record_dict.get('source'),
                            'platform': 'reddit',
                            'subreddit': extended_data.get('subreddit'),
                            'post_id': extended_data.get('id'),
                            'parent_id': extended_data.get('post_id'),
                            'author': extended_data.get('author'),
                            'content_type': extended_data.get('post_type'),
                            'sentiment_score': record_dict.get('sentiment_score'),
                            'sentiment_magnitude': record_dict.get('sentiment_magnitude'),
                            'score': extended_data.get('score', 0)
                        })
                    
                    df = pd.DataFrame(data)
            
            # Ensure DataFrame has the right columns
            if not df.empty:
                # Convert time to datetime if it's not already
                if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'])
                
                # Set time as index
                if 'time' in df.columns:
                    df = df.set_index('time')
                
                # Sort by time
                df = df.sort_index()
                
                self.logger.info(f"Collected {len(df)} social records for {symbol}")
            else:
                self.logger.warning(f"No social data available for {symbol}")
                df = pd.DataFrame()
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error collecting social data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_options_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Collect options flow data for a symbol.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with options data
        """
        try:
            # Try to get options data from Unusual Whales collector
            collector = get_collector('unusual_whales')
            if collector:
                collector.set_symbol(symbol)
                records = await collector.collect(start, end)
                
                if records:
                    # Convert to DataFrame
                    data = []
                    for record in records:
                        record_dict = record.model_dump()
                        extended_data = record_dict.get('extended_data', {})
                        
                        data.append({
                            'time': record_dict.get('timestamp'),
                            'symbol': record_dict.get('symbol'),
                            'contract': extended_data.get('contract'),
                            'strike': extended_data.get('strike'),
                            'expiration': extended_data.get('expiration'),
                            'option_type': extended_data.get('option_type'),
                            'premium': extended_data.get('premium'),
                            'volume': extended_data.get('volume'),
                            'open_interest': extended_data.get('open_interest'),
                            'implied_volatility': extended_data.get('implied_volatility'),
                            'sentiment_score': record_dict.get('sentiment_score', 0),
                            'unusual_score': extended_data.get('unusual_score', 0)
                        })
                    
                    df = pd.DataFrame(data)
                    
                    # Ensure DataFrame has the right columns
                    if not df.empty:
                        # Convert time to datetime if it's not already
                        if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                            df['time'] = pd.to_datetime(df['time'])
                        
                        # Set time as index
                        if 'time' in df.columns:
                            df = df.set_index('time')
                        
                        # Sort by time
                        df = df.sort_index()
                        
                        self.logger.info(f"Collected {len(df)} options records for {symbol}")
                        return df
            
            self.logger.warning(f"No options data available for {symbol}")
            return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error collecting options data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _serialize_dataframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Serialize DataFrames to JSON for caching.
        
        Args:
            data_dict: Dictionary of DataFrames
            
        Returns:
            Dictionary with serialized DataFrames
        """
        serialized = {}
        
        for key, df in data_dict.items():
            if not df.empty:
                # Reset index to include time as a column
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                
                # Convert to JSON
                serialized[key] = df.to_json(orient='records', date_format='iso')
            else:
                serialized[key] = None
        
        return serialized
    
    def _deserialize_dataframes(self, serialized: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Deserialize DataFrames from JSON.
        
        Args:
            serialized: Dictionary with serialized DataFrames
            
        Returns:
            Dictionary of DataFrames
        """
        data_dict = {}
        
        for key, json_str in serialized.items():
            if json_str:
                # Convert from JSON
                df = pd.read_json(json_str, orient='records')
                
                # Convert time to datetime if it exists
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                
                data_dict[key] = df
            else:
                data_dict[key] = pd.DataFrame()
        
        return data_dict
    
    def generate_enhanced_features(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Generate enhanced features from multiple data sources.
        
        Args:
            symbol: Stock symbol
            data: Dictionary of DataFrames with data from different sources
            
        Returns:
            Dictionary of features
        """
        self.logger.info(f"Generating enhanced features for {symbol}")
        
        # Check cache first
        cache_key = f"features:{symbol}"
        cached_features = redis_client.get(cache_key)
        
        if cached_features:
            self.logger.info(f"Using cached features for {symbol}")
            return cached_features
        
        # Extract DataFrames
        price_df = data.get('price', pd.DataFrame())
        news_df = data.get('news', pd.DataFrame())
        social_df = data.get('social', pd.DataFrame())
        options_df = data.get('options', pd.DataFrame())
        
        # Initialize features dictionary
        features = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate price features
        if not price_df.empty:
            price_features = self.feature_engineer.generate_price_features(price_df)
            features.update({f"price_{k}": v for k, v in price_features.items()})
        
        # Generate news features
        if not news_df.empty:
            # Generate standard news features
            news_features = self.feature_engineer.generate_news_features(news_df)
            features.update({f"news_{k}": v for k, v in news_features.items()})
            
            # Add ML-based sentiment features if available
            if SENTIMENT_FEATURES_AVAILABLE and 'content' in news_df.columns:
                try:
                    # Get unique news content
                    news_texts = news_df['content'].dropna().unique().tolist()
                    
                    if news_texts:
                        # Generate sentiment features using ML
                        ml_sentiment_features = generate_sentiment_features(news_texts)
                        
                        # Aggregate sentiment features
                        if ml_sentiment_features:
                            avg_score = sum(f.get('sentiment_score', 0) for f in ml_sentiment_features) / len(ml_sentiment_features)
                            avg_magnitude = sum(f.get('sentiment_magnitude', 0) for f in ml_sentiment_features) / len(ml_sentiment_features)
                            bullish_count = sum(1 for f in ml_sentiment_features if f.get('sentiment_is_bullish', 0) > 0)
                            bearish_count = sum(1 for f in ml_sentiment_features if f.get('sentiment_is_bearish', 0) > 0)
                            
                            # Add to features
                            features.update({
                                'news_ml_sentiment_score': avg_score,
                                'news_ml_sentiment_magnitude': avg_magnitude,
                                'news_ml_bullish_ratio': bullish_count / len(ml_sentiment_features) if ml_sentiment_features else 0,
                                'news_ml_bearish_ratio': bearish_count / len(ml_sentiment_features) if ml_sentiment_features else 0
                            })
                            
                            self.logger.info(f"Added ML-based sentiment features for {symbol} news")
                except Exception as e:
                    self.logger.warning(f"Error generating ML-based sentiment features for news: {str(e)}")
        
        # Generate social features
        if not social_df.empty:
            # Generate standard social features
            social_features = self.feature_engineer.generate_social_features(social_df)
            features.update({f"social_{k}": v for k, v in social_features.items()})
            
            # Add ML-based sentiment features if available
            if SENTIMENT_FEATURES_AVAILABLE and 'content' in social_df.columns:
                try:
                    # Get unique social media content
                    social_texts = social_df['content'].dropna().unique().tolist()
                    
                    if social_texts:
                        # Generate sentiment features using ML
                        ml_sentiment_features = generate_sentiment_features(social_texts)
                        
                        # Aggregate sentiment features
                        if ml_sentiment_features:
                            avg_score = sum(f.get('sentiment_score', 0) for f in ml_sentiment_features) / len(ml_sentiment_features)
                            avg_magnitude = sum(f.get('sentiment_magnitude', 0) for f in ml_sentiment_features) / len(ml_sentiment_features)
                            bullish_count = sum(1 for f in ml_sentiment_features if f.get('sentiment_is_bullish', 0) > 0)
                            bearish_count = sum(1 for f in ml_sentiment_features if f.get('sentiment_is_bearish', 0) > 0)
                            
                            # Add to features
                            features.update({
                                'social_ml_sentiment_score': avg_score,
                                'social_ml_sentiment_magnitude': avg_magnitude,
                                'social_ml_bullish_ratio': bullish_count / len(ml_sentiment_features) if ml_sentiment_features else 0,
                                'social_ml_bearish_ratio': bearish_count / len(ml_sentiment_features) if ml_sentiment_features else 0
                            })
                            
                            self.logger.info(f"Added ML-based sentiment features for {symbol} social media")
                except Exception as e:
                    self.logger.warning(f"Error generating ML-based sentiment features for social media: {str(e)}")
        
        # Generate options features
        if not options_df.empty:
            options_features = self.feature_engineer.generate_options_features(options_df)
            features.update({f"options_{k}": v for k, v in options_features.items()})
        
        # Cache features
        redis_client.set(cache_key, features, self.cache_ttl)
        
        return features
    
    def calculate_composite_score(self, symbol: str, features: Dict[str, float]) -> float:
        """
        Calculate a composite score from multiple feature groups.
        
        Args:
            symbol: Stock symbol
            features: Dictionary of features
            
        Returns:
            Composite score (-1 to 1 range)
        """
        self.logger.info(f"Calculating composite score for {symbol}")
        
        # Group features by type
        feature_groups = {
            'price_momentum': [],
            'price_volatility': [],
            'volume_profile': [],
            'news_sentiment': [],
            'social_sentiment': [],
            'options_flow': []
        }
        
        # Assign features to groups
        for key, value in features.items():
            if not isinstance(value, (int, float)):
                continue
            
            if key.startswith('price_mom_') or key.startswith('price_trend_'):
                feature_groups['price_momentum'].append(value)
            elif key.startswith('price_vol_') or key.startswith('price_atr_'):
                feature_groups['price_volatility'].append(value)
            elif key.startswith('price_volume_') or key.startswith('price_vwap_'):
                feature_groups['volume_profile'].append(value)
            elif key.startswith('news_'):
                feature_groups['news_sentiment'].append(value)
            elif key.startswith('social_'):
                feature_groups['social_sentiment'].append(value)
            elif key.startswith('options_'):
                feature_groups['options_flow'].append(value)
        
        # Calculate group scores
        group_scores = {}
        for group, values in feature_groups.items():
            if values:
                # Normalize values to -1 to 1 range
                normalized = [max(-1, min(1, v)) for v in values]
                # Calculate average
                group_scores[group] = sum(normalized) / len(normalized)
            else:
                group_scores[group] = 0
        
        # Calculate weighted composite score
        composite_score = 0
        total_weight = 0
        
        for group, score in group_scores.items():
            weight = self.feature_weights.get(group, 0)
            composite_score += score * weight
            total_weight += weight
        
        # Normalize to -1 to 1 range
        if total_weight > 0:
            composite_score /= total_weight
        
        # Ensure score is in -1 to 1 range
        composite_score = max(-1, min(1, composite_score))
        
        # Cache score
        cache_key = f"score:{symbol}"
        redis_client.set(cache_key, {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'composite_score': composite_score,
            'group_scores': group_scores
        }, self.cache_ttl)
        
        return composite_score
    
    async def prepare_training_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Prepare training data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with features and target variables
        """
        self.logger.info(f"Preparing training data for {symbol} with {days} days of history")
        
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Collect data
        data = await self.collect_all_data(symbol, start, end)
        
        # Extract price DataFrame for target calculation
        price_df = data.get('price', pd.DataFrame())
        
        if price_df.empty:
            self.logger.warning(f"No price data available for {symbol}, cannot prepare training data")
            return pd.DataFrame()
        
        # Create daily feature DataFrame
        daily_features = []
        
        # Process each day
        current = start
        while current <= end:
            next_day = current + timedelta(days=1)
            
            # Get data up to current day
            current_data = {
                'price': price_df[price_df.index <= current],
                'news': data.get('news', pd.DataFrame())[data.get('news', pd.DataFrame()).index <= current],
                'social': data.get('social', pd.DataFrame())[data.get('social', pd.DataFrame()).index <= current],
                'options': data.get('options', pd.DataFrame())[data.get('options', pd.DataFrame()).index <= current]
            }
            
            # Generate features
            if not current_data['price'].empty:
                features = self.generate_enhanced_features(symbol, current_data)
                
                # Add date
                features['date'] = current.date().isoformat()
                
                # Calculate target variables (next day return)
                next_day_data = price_df[(price_df.index > current) & (price_df.index <= next_day)]
                
                if not next_day_data.empty:
                    current_close = current_data['price']['close'].iloc[-1]
                    next_close = next_day_data['close'].iloc[-1] if not next_day_data.empty else None
                    
                    if next_close is not None:
                        features['target_next_day_return'] = (next_close / current_close) - 1
                    
                    # Add high and low returns
                    if 'high' in next_day_data.columns and 'low' in next_day_data.columns:
                        next_high = next_day_data['high'].max()
                        next_low = next_day_data['low'].min()
                        
                        features['target_next_day_high_return'] = (next_high / current_close) - 1
                        features['target_next_day_low_return'] = (next_low / current_close) - 1
                
                daily_features.append(features)
            
            # Move to next day
            current = next_day
        
        # Convert to DataFrame
        if daily_features:
            df = pd.DataFrame(daily_features)
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            self.logger.info(f"Prepared {len(df)} training samples for {symbol}")
            return df
        else:
            self.logger.warning(f"No training data prepared for {symbol}")
            return pd.DataFrame()
