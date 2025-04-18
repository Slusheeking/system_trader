#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Cache Module
---------------
Provides caching functionality for market data and features.
Uses Redis for high-performance in-memory caching with configurable TTL.
Specialized for websocket real-time data caching and REST API data buffering.
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import threading

from utils.logging import setup_logger
from data.database.redis_client import RedisClient, get_redis_client

# Setup logging
logger = setup_logger('data_cache', category='data')

# Module-level Redis client
redis_client = get_redis_client()


class DataCache:
    """
    Cache for market data and features using Redis.
    Provides methods for storing and retrieving data with automatic serialization.
    Specialized methods for handling real-time WebSocket data vs REST API data.
    """
    
    def __init__(self, namespace: str = 'cache', ttl: int = 86400):
        """
        Initialize the data cache.
        
        Args:
            namespace: Redis namespace prefix
            ttl: Default time-to-live in seconds (default: 24 hours)
        """
        self.namespace = namespace
        self.default_ttl = ttl
        self.redis = redis_client
        self._lock = threading.RLock()  # Re-entrant lock for thread safety
        
        # WebSocket specific settings
        self.websocket_ttl = 300  # 5 minutes for websocket data by default
        self.max_stream_length = 10000  # Maximum length of stream entries before trimming
        
        if not self.redis:
            logger.warning("Redis client not available, caching will be disabled")
        else:
            logger.info(f"DataCache initialized with namespace '{namespace}' and TTL {ttl}s")
    
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis:
            return None
        
        try:
            # Prefix key with namespace
            namespaced_key = f"{self.namespace}:{key}"
            
            # Get from Redis using cache_get
            return self.redis.cache_get(namespaced_key)
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            Boolean indicating success
        """
        if not self.redis:
            return False
        
        try:
            # Prefix key with namespace
            namespaced_key = f"{self.namespace}:{key}"
            
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Set in Redis using cache_set
            return self.redis.cache_set(namespaced_key, value, ttl)
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Boolean indicating success
        """
        if not self.redis:
            return False
        
        try:
            # Prefix key with namespace
            namespaced_key = f"{self.namespace}:{key}"
            
            # Delete from Redis using cache_invalidate
            return bool(self.redis.cache_invalidate(namespaced_key))
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Boolean indicating existence
        """
        if not self.redis:
            return False
        
        try:
            # Prefix key with namespace
            namespaced_key = f"{self.namespace}:{key}"
            
            # Check existence
            return self.redis.exists(namespaced_key)
        except Exception as e:
            logger.error(f"Error checking cache existence: {str(e)}")
            return False
    
    def clear(self, pattern: str = "*") -> int:
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0
        
        try:
            # Prefix pattern with namespace
            namespaced_pattern = f"{self.namespace}:{pattern}"
            
            # Clear cache using cache_invalidate
            return self.redis.cache_invalidate(namespaced_pattern)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    def get_market_data(self, symbol: str, start_date: str, end_date: str, 
                       interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get cached market data for a symbol and date range.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with market data or None if not cached
        """
        # Generate cache key
        key = self._make_market_data_key(symbol, start_date, end_date, interval)
        
        # Get from cache
        return self.get(key)
    
    def set_market_data(self, symbol: str, data: pd.DataFrame, start_date: str, 
                       end_date: str, interval: str = '1d', ttl: Optional[int] = None) -> bool:
        """
        Cache market data for a symbol and date range.
        
        Args:
            symbol: Ticker symbol
            data: DataFrame with market data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            Boolean indicating success
        """
        # Generate cache key
        key = self._make_market_data_key(symbol, start_date, end_date, interval)
        
        # Set in cache
        return self.set(key, data, ttl)
    
    def get_features(self, symbols: List[str], start_date: str, end_date: str, 
                    feature_sets: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Get cached features for symbols and date range.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            feature_sets: List of feature set names
            
        Returns:
            DataFrame with features or None if not cached
        """
        # Generate cache key
        key = self._make_features_key(symbols, start_date, end_date, feature_sets)
        
        # Get from cache
        return self.get(key)
    
    def set_features(self, symbols: List[str], data: pd.DataFrame, start_date: str, 
                    end_date: str, feature_sets: Optional[List[str]] = None, 
                    ttl: Optional[int] = None) -> bool:
        """
        Cache features for symbols and date range.
        
        Args:
            symbols: List of ticker symbols
            data: DataFrame with features
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            feature_sets: List of feature set names
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            Boolean indicating success
        """
        # Generate cache key
        key = self._make_features_key(symbols, start_date, end_date, feature_sets)
        
        # Set in cache
        return self.set(key, data, ttl)
    
    def get_model_prediction(self, model_name: str, input_hash: str) -> Any:
        """
        Get cached model prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of input data
            
        Returns:
            Cached prediction or None if not found
        """
        # Generate cache key
        key = f"model:{model_name}:{input_hash}"
        
        # Get from cache
        return self.get(key)
    
    def set_model_prediction(self, model_name: str, input_hash: str, 
                            prediction: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache model prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of input data
            prediction: Prediction to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            Boolean indicating success
        """
        # Generate cache key
        key = f"model:{model_name}:{input_hash}"
        
        # Set in cache
        return self.set(key, prediction, ttl)
    
    def hash_input(self, input_data: Any) -> str:
        """
        Create a hash of input data for caching.
        
        Args:
            input_data: Input data to hash
            
        Returns:
            Hash string
        """
        try:
            # Serialize input data
            if isinstance(input_data, (dict, list, tuple, set)):
                serialized = json.dumps(input_data, sort_keys=True)
            elif isinstance(input_data, (np.ndarray, pd.DataFrame, pd.Series)):
                serialized = str(input_data.shape)
            else:
                serialized = str(input_data)
            
            # Create hash
            if isinstance(serialized, str):
                serialized = serialized.encode('utf-8')
            
            return hashlib.md5(serialized).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing input data: {str(e)}")
            # Fallback to timestamp + random number
            return f"{int(time.time())}-{np.random.randint(1000000)}"
    
    def _make_market_data_key(self, symbol: str, start_date: str, end_date: str, 
                             interval: str) -> str:
        """
        Generate a cache key for market data.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Cache key
        """
        return f"market:{symbol}:{start_date}:{end_date}:{interval}"
    
    def _make_features_key(self, symbols: List[str], start_date: str, end_date: str, 
                          feature_sets: Optional[List[str]]) -> str:
        """
        Generate a cache key for features.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            feature_sets: List of feature set names
            
        Returns:
            Cache key
        """
        # Sort symbols for consistent key generation
        sorted_symbols = sorted(symbols)
        
        # Create feature sets string
        if feature_sets:
            feature_str = "_".join(sorted(feature_sets))
        else:
            feature_str = "all"
        
        # Create a hash of the symbols list
        symbols_str = ",".join(sorted_symbols)
        symbols_hash = hashlib.md5(symbols_str.encode()).hexdigest()[:8]
        
        return f"features:{symbols_hash}:{start_date}:{end_date}:{feature_str}"
    
    # ---- WebSocket-specific methods for real-time data ----
    
    def add_websocket_data(self, symbol: str, data_type: str, data: Dict[str, Any]) -> bool:
        """
        Add real-time WebSocket data to Redis.
        
        Args:
            symbol: Ticker symbol
            data_type: Type of data (trades, quotes, aggregate)
            data: Data dictionary from WebSocket
            
        Returns:
            Boolean indicating success
        """
        if not self.redis:
            return False
        
        try:
            with self._lock:
                # For WebSocket data, we'll use Redis Sorted Sets with a score as the timestamp
                # This allows us to easily retrieve data by time range and trim old data
                
                # Get timestamp from data or use current time
                if 'timestamp' in data:
                    timestamp = self._ensure_timestamp(data['timestamp'])
                else:
                    timestamp = time.time()
                
                # Generate key for the sorted set
                key = f"ws:{data_type}:{symbol}"
                namespaced_key = f"{self.namespace}:{key}"
                
                # Add to sorted set
                self.redis.zadd(namespaced_key, {json.dumps(data): timestamp})
                
                # Set expiry if not set
                if not self.redis.ttl(namespaced_key) > 0:
                    self.redis.expire(namespaced_key, self.websocket_ttl)
                
                # Trim to max length to prevent unbounded growth
                set_size = self.redis.zcard(namespaced_key)
                if set_size > self.max_stream_length:
                    # Keep only the newest max_stream_length items
                    excess = set_size - self.max_stream_length
                    self.redis.zremrangebyrank(namespaced_key, 0, excess - 1)
                
                return True
        except Exception as e:
            logger.error(f"Error adding WebSocket data: {str(e)}")
            return False
    
    def get_latest_websocket_data(self, symbol: str, data_type: str, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the latest WebSocket data for a symbol.
        
        Args:
            symbol: Ticker symbol
            data_type: Type of data (trades, quotes, aggregate)
            count: Number of latest items to retrieve
            
        Returns:
            List of data dictionaries
        """
        if not self.redis:
            return []
        
        try:
            # Generate key
            key = f"ws:{data_type}:{symbol}"
            namespaced_key = f"{self.namespace}:{key}"
            
            # Get latest items from sorted set
            items = self.redis.zrange(namespaced_key, -count, -1, desc=True, withscores=True)
            
            # Convert to list of dictionaries
            result = []
            for item, score in items:
                try:
                    data = json.loads(item)
                    # Add timestamp if not in the data
                    if 'timestamp' not in data:
                        data['timestamp'] = score
                    result.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse WebSocket data: {item}")
            
            return result
        except Exception as e:
            logger.error(f"Error getting latest WebSocket data: {str(e)}")
            return []
    
    def get_websocket_data_range(self, symbol: str, data_type: str, 
                               start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Get WebSocket data for a time range.
        
        Args:
            symbol: Ticker symbol
            data_type: Type of data (trades, quotes, aggregate)
            start_time: Start time as Unix timestamp
            end_time: End time as Unix timestamp
            
        Returns:
            List of data dictionaries
        """
        if not self.redis:
            return []
        
        try:
            # Generate key
            key = f"ws:{data_type}:{symbol}"
            namespaced_key = f"{self.namespace}:{key}"
            
            # Get items in time range from sorted set
            items = self.redis.zrangebyscore(namespaced_key, start_time, end_time, withscores=True)
            
            # Convert to list of dictionaries
            result = []
            for item, score in items:
                try:
                    data = json.loads(item)
                    # Add timestamp if not in the data
                    if 'timestamp' not in data:
                        data['timestamp'] = score
                    result.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse WebSocket data: {item}")
            
            return result
        except Exception as e:
            logger.error(f"Error getting WebSocket data range: {str(e)}")
            return []
    
    def get_all_active_symbols(self, data_type: str = None) -> Set[str]:
        """
        Get all symbols that have active WebSocket data.
        
        Args:
            data_type: Optional type of data to filter by
            
        Returns:
            Set of active symbols
        """
        if not self.redis:
            return set()
        
        try:
            pattern = f"{self.namespace}:ws:*"
            if data_type:
                pattern = f"{self.namespace}:ws:{data_type}:*"
            
            # Get all keys matching the pattern
            keys = self.redis.keys(pattern)
            
            # Extract symbols from keys
            symbols = set()
            for key in keys:
                # Convert from bytes to string if needed
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                # Extract symbol from key pattern "namespace:ws:data_type:symbol"
                parts = key.split(':')
                if len(parts) >= 4:
                    symbols.add(parts[3])
            
            return symbols
        except Exception as e:
            logger.error(f"Error getting active symbols: {str(e)}")
            return set()
    
    def clear_websocket_data(self, symbol: str = None, data_type: str = None) -> int:
        """
        Clear WebSocket data from Redis.
        
        Args:
            symbol: Optional symbol to clear (None for all symbols)
            data_type: Optional data type to clear (None for all types)
            
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0
        
        try:
            # Build pattern based on parameters
            if symbol and data_type:
                pattern = f"ws:{data_type}:{symbol}"
            elif data_type:
                pattern = f"ws:{data_type}:*"
            elif symbol:
                pattern = f"ws:*:{symbol}"
            else:
                pattern = "ws:*"
            
            # Clear matching keys
            return self.clear(pattern)
        except Exception as e:
            logger.error(f"Error clearing WebSocket data: {str(e)}")
            return 0
    
    def _ensure_timestamp(self, timestamp: Any) -> float:
        """
        Ensure timestamp is in Unix timestamp format (seconds since epoch).
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            Unix timestamp as float
        """
        # If already a float or int, check if it's milliseconds or seconds
        if isinstance(timestamp, (int, float)):
            # If timestamp is in milliseconds (13 digits), convert to seconds
            if timestamp > 1e12:
                return timestamp / 1000.0
            return float(timestamp)
        
        # If string, try to parse as ISO format
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.timestamp()
            except ValueError:
                pass
        
        # If datetime object
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        
        # Default to current time
        return time.time()


# Module-level instance for convenience
_default_cache = None

def get_data_cache(namespace: str = 'cache', ttl: int = 86400) -> DataCache:
    """
    Get or create the default data cache.
    
    Args:
        namespace: Redis namespace prefix
        ttl: Default time-to-live in seconds
        
    Returns:
        DataCache instance
    """
    global _default_cache
    
    if _default_cache is None:
        _default_cache = DataCache(namespace, ttl)
    
    return _default_cache


if __name__ == "__main__":
    # Example usage
    cache = DataCache()
    
    # Cache some data
    data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'timestamp': '2023-04-17T09:30:00Z'
    }
    
    cache.set('test_key', data)
    
    # Retrieve from cache
    cached_data = cache.get('test_key')
    
    if cached_data:
        print("Retrieved from cache:", cached_data)
    else:
        print("Data not found in cache")
    
    # Test WebSocket caching
    ws_data = {
        'symbol': 'AAPL',
        'price': 150.75,
        'size': 100,
        'timestamp': time.time()
    }
    
    cache.add_websocket_data('AAPL', 'trades', ws_data)
    
    # Get latest WebSocket data
    latest = cache.get_latest_websocket_data('AAPL', 'trades')
    print("Latest WebSocket data:", latest)
    
    # Clear cache
    cache.clear()
