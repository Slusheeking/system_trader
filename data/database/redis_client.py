#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redis Client
-----------
Provides high-performance caching and data storage using Redis.
Handles market data caching, feature storage, and model prediction caching.
"""

import logging
import json
import pickle
import hashlib
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
import redis
import pandas as pd
import numpy as np
from functools import wraps

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('redis_client', category='data')


def cache_decorator(expires: int = 300):
    """
    Decorator for caching function results in Redis.
    
    Args:
        expires: Cache expiration time in seconds (default: 5 minutes)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate a cache key
            key_parts = [func.__name__]
            # Add args to key
            for arg in args:
                key_parts.append(str(arg))
            # Add kwargs to key (sorted to ensure consistency)
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            cache_key = f"cache:{':'.join(key_parts)}"
            
            # Check if result is in cache
            cached_result = self.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # If not in cache, execute function
            result = func(self, *args, **kwargs)
            
            # Store in cache
            if result is not None:
                self.set(cache_key, result, expires)
                logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator


class RedisClient:
    """
    Client for interacting with Redis for caching and data storage.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Redis client.
        
        Args:
            config: Redis configuration settings
        """
        self.config = config or {}
        
        # Redis connection settings
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 6379)
        self.db = self.config.get('db', 0)
        self.password = self.config.get('password')
        self.socket_timeout = self.config.get('socket_timeout', 5)
        self.socket_connect_timeout = self.config.get('socket_connect_timeout', 5)
        self.retry_on_timeout = self.config.get('retry_on_timeout', True)
        self.max_connections = self.config.get('max_connections', 10)
        
        # Redis client options
        self.use_connection_pool = self.config.get('use_connection_pool', True)
        self.decode_responses = self.config.get('decode_responses', False)
        
        # Namespace for keys
        self.namespace = self.config.get('namespace', 'trading')
        
        # Default expiration time (24 hours)
        self.default_expiry = self.config.get('default_expiry', 86400)
        
        # Redis client
        self.redis = self._initialize_redis()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Connect and verify connection
        self._test_connection()
        
        logger.info(f"RedisClient initialized for {self.host}:{self.port}/db{self.db}")
    
    def _initialize_redis(self) -> redis.Redis:
        """
        Initialize Redis client.
        
        Returns:
            Redis client instance
        """
        try:
            if self.use_connection_pool:
                # Create connection pool
                pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    max_connections=self.max_connections,
                    decode_responses=self.decode_responses
                )
                # Create Redis client with connection pool
                return redis.Redis(connection_pool=pool)
            else:
                # Create direct Redis client
                return redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    decode_responses=self.decode_responses
                )
        except Exception as e:
            logger.error(f"Error initializing Redis client: {str(e)}")
            raise
    
    def _test_connection(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Ping Redis server
            self.redis.ping()
            logger.info("Successfully connected to Redis server")
            return True
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error testing Redis connection: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health and connectivity.
        
        Returns:
            Dictionary with health status information
        """
        try:
            start_time = time.time()
            is_connected = self._test_connection()
            response_time = time.time() - start_time
            
            health_info = {
                "status": "healthy" if is_connected else "unhealthy",
                "connected": is_connected,
                "response_time_ms": round(response_time * 1000, 2),
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "timestamp": datetime.now().isoformat()
            }
            
            if is_connected:
                # Add basic Redis info if connected
                info = self.redis.info(section="memory")
                health_info["memory_used"] = info.get("used_memory_human", "unknown")
                health_info["max_memory"] = info.get("maxmemory_human", "unlimited")
                health_info["memory_usage_pct"] = info.get("used_memory_peak_perc", "unknown")
            
            return health_info
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _make_key(self, key: str) -> str:
        """
        Create a namespaced key.
        
        Args:
            key: Original key
            
        Returns:
            Namespaced key
        """
        return f"{self.namespace}:{key}"
    
    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """
        Set a value in Redis with optional expiration.
        
        Args:
            key: Key name
            value: Value to store
            expiry: Expiration time in seconds (None for default)
            
        Returns:
            Boolean indicating success
        """
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Create namespaced key
                namespaced_key = self._make_key(key)
                
                # Use default expiry if not specified
                if expiry is None:
                    expiry = self.default_expiry
                
                # Serialize value if needed
                serialized_value = self._serialize_value(value)
                
                # Set in Redis with expiry
                self.redis.set(namespaced_key, serialized_value, ex=expiry)
                
                return True
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection error on attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Redis connection error after {max_retries} attempts: {str(e)}")
                    return False
            except Exception as e:
                logger.error(f"Error setting Redis key {key}: {str(e)}")
                return False
        
        return False
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache with optional TTL.
        Alias for set() to match the required interface.
        
        Args:
            key: Key name
            value: Value to store
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            Boolean indicating success
        """
        return self.set(key, value, ttl)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis.
        
        Args:
            key: Key name
            default: Default value if key doesn't exist
            
        Returns:
            Value or default
        """
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Create namespaced key
                namespaced_key = self._make_key(key)
                
                # Get from Redis
                value = self.redis.get(namespaced_key)
                
                # Return default if not found
                if value is None:
                    return default
                
                # Deserialize value
                return self._deserialize_value(value)
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection error on attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Redis connection error after {max_retries} attempts: {str(e)}")
                    return default
            except Exception as e:
                logger.error(f"Error getting Redis key {key}: {str(e)}")
                return default
        
        return default
    
    def cache_get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        Alias for get() to match the required interface.
        
        Args:
            key: Key name
            default: Default value if key doesn't exist
            
        Returns:
            Value or default
        """
        return self.get(key, default)
    
    def _serialize_value(self, value: Any) -> bytes:
        """
        Serialize a value for Redis storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value
        """
        # Simple types that Redis handles natively
        if isinstance(value, (str, int, float, bytes, bytearray)):
            if isinstance(value, str):
                return value.encode('utf-8')
            elif isinstance(value, (int, float)):
                return str(value).encode('utf-8')
            elif isinstance(value, (bytes, bytearray)):
                return value
        
        # Handle common types
        if isinstance(value, (list, dict, tuple, set)) and all(isinstance(x, (str, int, float)) for x in value):
            # Simple collection of simple types - use JSON
            return json.dumps(value).encode('utf-8')
        
        # For pandas DataFrame
        if isinstance(value, pd.DataFrame):
            # Convert to JSON with special type handling
            return value.to_json().encode('utf-8')
        
        # For numpy array
        if isinstance(value, np.ndarray):
            # Handle numpy array depending on size
            if value.size < 1000:  # Small array - use JSON
                return json.dumps(value.tolist()).encode('utf-8')
            else:  # Large array - pickle
                return pickle.dumps(value, protocol=4)
        
        # For dates and times
        if isinstance(value, (datetime, timedelta)):
            return str(value).encode('utf-8')
        
        # For everything else, use pickle
        return pickle.dumps(value, protocol=4)
    
    def _deserialize_value(self, value: bytes) -> Any:
        """
        Deserialize a value from Redis.
        
        Args:
            value: Value to deserialize
            
        Returns:
            Deserialized value
        """
        if value is None:
            return None
        
        # First try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Next try pickle
        try:
            return pickle.loads(value)
        except pickle.UnpicklingError:
            pass
        
        # If all else fails, return as string
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Key name
            
        Returns:
            Boolean indicating success
        """
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Create namespaced key
                namespaced_key = self._make_key(key)
                
                # Delete from Redis
                self.redis.delete(namespaced_key)
                
                return True
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection error on attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Redis connection error after {max_retries} attempts: {str(e)}")
                    return False
            except Exception as e:
                logger.error(f"Error deleting Redis key {key}: {str(e)}")
                return False
        
        return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Key name
            
        Returns:
            Boolean indicating existence
        """
        try:
            # Create namespaced key
            namespaced_key = self._make_key(key)
            
            # Check existence
            return bool(self.redis.exists(namespaced_key))
        except Exception as e:
            logger.error(f"Error checking Redis key {key}: {str(e)}")
            return False
    
    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Key name
            seconds: Expiration time in seconds
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create namespaced key
            namespaced_key = self._make_key(key)
            
            # Set expiration
            return bool(self.redis.expire(namespaced_key, seconds))
        except Exception as e:
            logger.error(f"Error setting expiry for Redis key {key}: {str(e)}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        Get time to live for a key.
        
        Args:
            key: Key name
            
        Returns:
            Time to live in seconds or -1 if no expiry, -2 if key doesn't exist
        """
        try:
            # Create namespaced key
            namespaced_key = self._make_key(key)
            
            # Get TTL
            return self.redis.ttl(namespaced_key)
        except Exception as e:
            logger.error(f"Error getting TTL for Redis key {key}: {str(e)}")
            return -2
    
    def hset(self, name: str, key: str, value: Any) -> bool:
        """
        Set a hash field.
        
        Args:
            name: Hash name
            key: Field name
            value: Field value
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Set hash field
            self.redis.hset(namespaced_name, key, serialized_value)
            
            return True
        except Exception as e:
            logger.error(f"Error setting Redis hash field {name}.{key}: {str(e)}")
            return False
    
    def hget(self, name: str, key: str, default: Any = None) -> Any:
        """
        Get a hash field.
        
        Args:
            name: Hash name
            key: Field name
            default: Default value if field doesn't exist
            
        Returns:
            Field value or default
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get hash field
            value = self.redis.hget(namespaced_name, key)
            
            # Return default if not found
            if value is None:
                return default
            
            # Deserialize value
            return self._deserialize_value(value)
        except Exception as e:
            logger.error(f"Error getting Redis hash field {name}.{key}: {str(e)}")
            return default
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """
        Get all fields and values in a hash.
        
        Args:
            name: Hash name
            
        Returns:
            Dictionary of field names and values
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get all hash fields
            hash_dict = self.redis.hgetall(namespaced_name)
            
            # Deserialize values
            result = {}
            for k, v in hash_dict.items():
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                result[key] = self._deserialize_value(v)
            
            return result
        except Exception as e:
            logger.error(f"Error getting all Redis hash fields for {name}: {str(e)}")
            return {}
    
    def hdel(self, name: str, *keys) -> int:
        """
        Delete hash fields.
        
        Args:
            name: Hash name
            *keys: Field names
            
        Returns:
            Number of fields deleted
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Delete hash fields
            return self.redis.hdel(namespaced_name, *keys)
        except Exception as e:
            logger.error(f"Error deleting Redis hash fields for {name}: {str(e)}")
            return 0
    
    def hexists(self, name: str, key: str) -> bool:
        """
        Check if hash field exists.
        
        Args:
            name: Hash name
            key: Field name
            
        Returns:
            Boolean indicating existence
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Check field existence
            return bool(self.redis.hexists(namespaced_name, key))
        except Exception as e:
            logger.error(f"Error checking Redis hash field {name}.{key}: {str(e)}")
            return False
    
    def lpush(self, name: str, *values) -> int:
        """
        Push values onto the head of a list.
        
        Args:
            name: List name
            *values: Values to push
            
        Returns:
            Length of the list after push
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize values
            serialized_values = [self._serialize_value(v) for v in values]
            
            # Push to list
            return self.redis.lpush(namespaced_name, *serialized_values)
        except Exception as e:
            logger.error(f"Error pushing to Redis list {name}: {str(e)}")
            return 0
    
    def rpush(self, name: str, *values) -> int:
        """
        Push values onto the tail of a list.
        
        Args:
            name: List name
            *values: Values to push
            
        Returns:
            Length of the list after push
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize values
            serialized_values = [self._serialize_value(v) for v in values]
            
            # Push to list
            return self.redis.rpush(namespaced_name, *serialized_values)
        except Exception as e:
            logger.error(f"Error pushing to Redis list {name}: {str(e)}")
            return 0
    
    def lpop(self, name: str) -> Any:
        """
        Pop a value from the head of a list.
        
        Args:
            name: List name
            
        Returns:
            Popped value or None
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Pop from list
            value = self.redis.lpop(namespaced_name)
            
            # Return None if not found
            if value is None:
                return None
            
            # Deserialize value
            return self._deserialize_value(value)
        except Exception as e:
            logger.error(f"Error popping from Redis list {name}: {str(e)}")
            return None
    
    def rpop(self, name: str) -> Any:
        """
        Pop a value from the tail of a list.
        
        Args:
            name: List name
            
        Returns:
            Popped value or None
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Pop from list
            value = self.redis.rpop(namespaced_name)
            
            # Return None if not found
            if value is None:
                return None
            
            # Deserialize value
            return self._deserialize_value(value)
        except Exception as e:
            logger.error(f"Error popping from Redis list {name}: {str(e)}")
            return None
    
    def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """
        Get a range of values from a list.
        
        Args:
            name: List name
            start: Start index (0-based)
            end: End index (inclusive)
            
        Returns:
            List of values
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get range from list
            values = self.redis.lrange(namespaced_name, start, end)
            
            # Deserialize values
            return [self._deserialize_value(v) for v in values]
        except Exception as e:
            logger.error(f"Error getting range from Redis list {name}: {str(e)}")
            return []
    
    def llen(self, name: str) -> int:
        """
        Get the length of a list.
        
        Args:
            name: List name
            
        Returns:
            List length
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get list length
            return self.redis.llen(namespaced_name)
        except Exception as e:
            logger.error(f"Error getting length of Redis list {name}: {str(e)}")
            return 0
    
    def sadd(self, name: str, *values) -> int:
        """
        Add values to a set.
        
        Args:
            name: Set name
            *values: Values to add
            
        Returns:
            Number of values added
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize values
            serialized_values = [self._serialize_value(v) for v in values]
            
            # Add to set
            return self.redis.sadd(namespaced_name, *serialized_values)
        except Exception as e:
            logger.error(f"Error adding to Redis set {name}: {str(e)}")
            return 0
    
    def smembers(self, name: str) -> Set[Any]:
        """
        Get all members of a set.
        
        Args:
            name: Set name
            
        Returns:
            Set of values
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get set members
            values = self.redis.smembers(namespaced_name)
            
            # Deserialize values
            return {self._deserialize_value(v) for v in values}
        except Exception as e:
            logger.error(f"Error getting members of Redis set {name}: {str(e)}")
            return set()
    
    def srem(self, name: str, *values) -> int:
        """
        Remove values from a set.
        
        Args:
            name: Set name
            *values: Values to remove
            
        Returns:
            Number of values removed
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize values
            serialized_values = [self._serialize_value(v) for v in values]
            
            # Remove from set
            return self.redis.srem(namespaced_name, *serialized_values)
        except Exception as e:
            logger.error(f"Error removing from Redis set {name}: {str(e)}")
            return 0
    
    def sismember(self, name: str, value: Any) -> bool:
        """
        Check if value is a member of a set.
        
        Args:
            name: Set name
            value: Value to check
            
        Returns:
            Boolean indicating membership
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Check membership
            return bool(self.redis.sismember(namespaced_name, serialized_value))
        except Exception as e:
            logger.error(f"Error checking membership in Redis set {name}: {str(e)}")
            return False
    
    def zadd(self, name: str, mapping: Dict[Any, float]) -> int:
        """
        Add members to a sorted set.
        
        Args:
            name: Sorted set name
            mapping: Mapping of members to scores
            
        Returns:
            Number of members added
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize mapping
            serialized_mapping = {self._serialize_value(k): v for k, v in mapping.items()}
            
            # Add to sorted set
            return self.redis.zadd(namespaced_name, serialized_mapping)
        except Exception as e:
            logger.error(f"Error adding to Redis sorted set {name}: {str(e)}")
            return 0
    
    def zrange(self, name: str, start: int, end: int, desc: bool = False, withscores: bool = False) -> List[Any]:
        """
        Get a range of members from a sorted set.
        
        Args:
            name: Sorted set name
            start: Start index (0-based)
            end: End index (inclusive)
            desc: Whether to get in descending order
            withscores: Whether to include scores
            
        Returns:
            List of members (or tuples of member and score if withscores is True)
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get range from sorted set
            if desc:
                values = self.redis.zrevrange(namespaced_name, start, end, withscores=withscores)
            else:
                values = self.redis.zrange(namespaced_name, start, end, withscores=withscores)
            
            # Deserialize values
            if withscores:
                return [(self._deserialize_value(v[0]), v[1]) for v in values]
            else:
                return [self._deserialize_value(v) for v in values]
        except Exception as e:
            logger.error(f"Error getting range from Redis sorted set {name}: {str(e)}")
            return []
    
    def zcard(self, name: str) -> int:
        """
        Get the number of members in a sorted set.
        
        Args:
            name: Sorted set name
            
        Returns:
            Number of members
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Get sorted set cardinality
            return self.redis.zcard(namespaced_name)
        except Exception as e:
            logger.error(f"Error getting cardinality of Redis sorted set {name}: {str(e)}")
            return 0
    
    def zrem(self, name: str, *values) -> int:
        """
        Remove members from a sorted set.
        
        Args:
            name: Sorted set name
            *values: Members to remove
            
        Returns:
            Number of members removed
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize values
            serialized_values = [self._serialize_value(v) for v in values]
            
            # Remove from sorted set
            return self.redis.zrem(namespaced_name, *serialized_values)
        except Exception as e:
            logger.error(f"Error removing from Redis sorted set {name}: {str(e)}")
            return 0
    
    def zincrby(self, name: str, amount: float, value: Any) -> float:
        """
        Increment the score of a member in a sorted set.
        
        Args:
            name: Sorted set name
            amount: Amount to increment
            value: Member to increment
            
        Returns:
            New score
        """
        try:
            # Create namespaced key
            namespaced_name = self._make_key(name)
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Increment score
            return self.redis.zincrby(namespaced_name, amount, serialized_value)
        except Exception as e:
            logger.error(f"Error incrementing score in Redis sorted set {name}: {str(e)}")
            return 0.0
    
    def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a key's value.
        
        Args:
            key: Key name
            amount: Amount to increment
            
        Returns:
            New value
        """
        try:
            # Create namespaced key
            namespaced_key = self._make_key(key)
            
            # Increment value
            return self.redis.incrby(namespaced_key, amount)
        except Exception as e:
            logger.error(f"Error incrementing Redis key {key}: {str(e)}")
            return 0
    
    def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement a key's value.
        
        Args:
            key: Key name
            amount: Amount to decrement
            
        Returns:
            New value
        """
        try:
            # Create namespaced key
            namespaced_key = self._make_key(key)
            
            # Decrement value
            return self.redis.decrby(namespaced_key, amount)
        except Exception as e:
            logger.error(f"Error decrementing Redis key {key}: {str(e)}")
            return 0
    
    def keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            List of matching keys
        """
        try:
            # Create namespaced pattern
            namespaced_pattern = self._make_key(pattern)
            
            # Get matching keys
            keys = self.redis.keys(namespaced_pattern)
            
            # Remove namespace from keys
            prefix = f"{self.namespace}:"
            return [k.decode('utf-8').replace(prefix, '') if isinstance(k, bytes) else k.replace(prefix, '') for k in keys]
        except Exception as e:
            logger.error(f"Error getting Redis keys matching {pattern}: {str(e)}")
            return []
    
    def flush(self) -> bool:
        """
        Delete all keys in the current namespace.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Get all keys in namespace
            namespaced_pattern = f"{self.namespace}:*"
            keys = self.redis.keys(namespaced_pattern)
            
            # Delete keys if any
            if keys:
                self.redis.delete(*keys)
            
            return True
        except Exception as e:
            logger.error(f"Error flushing Redis namespace {self.namespace}: {str(e)}")
            return False
    
    def clear_cache(self, pattern: str = "cache:*") -> int:
        """
        Clear cached results matching a pattern.
        
        Args:
            pattern: Pattern to match (default: all cache keys)
            
        Returns:
            Number of keys deleted
        """
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Create namespaced pattern
                namespaced_pattern = self._make_key(pattern)
                
                # Get matching keys
                keys = self.redis.keys(namespaced_pattern)
                
                # Delete keys if any
                if keys:
                    return self.redis.delete(*keys)
                
                return 0
            except redis.ConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection error on attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Redis connection error after {max_retries} attempts: {str(e)}")
                    return 0
            except Exception as e:
                logger.error(f"Error clearing Redis cache {pattern}: {str(e)}")
                return 0
        
        return 0
    
    def cache_invalidate(self, key_or_pattern: str) -> int:
        """
        Invalidate cache entries matching a key or pattern.
        
        Args:
            key_or_pattern: Key or pattern to match
            
        Returns:
            Number of keys deleted
        """
        # If the key_or_pattern contains wildcard characters, treat as pattern
        if '*' in key_or_pattern or '?' in key_or_pattern:
            return self.clear_cache(key_or_pattern)
        else:
            # Otherwise treat as a single key
            return 1 if self.delete(key_or_pattern) else 0
    
    def pipeline(self) -> redis.client.Pipeline:
        """
        Get a Redis pipeline for batch operations.
        
        Returns:
            Redis pipeline
        """
        return self.redis.pipeline()
    
    def pubsub(self) -> redis.client.PubSub:
        """
        Get a Redis pubsub object for pub/sub operations.
        
        Returns:
            Redis pubsub object
        """
        return self.redis.pubsub()
    
    def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            Number of clients that received the message
        """
        try:
            # Create namespaced channel
            namespaced_channel = self._make_key(channel)
            
            # Serialize message
            serialized_message = self._serialize_value(message)
            
            # Publish message
            return self.redis.publish(namespaced_channel, serialized_message)
        except Exception as e:
            logger.error(f"Error publishing to Redis channel {channel}: {str(e)}")
            return 0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Dictionary with server info
        """
        try:
            # Get server info
            info = self.redis.info()
            
            # Add namespace info
            namespace_keys = len(self.redis.keys(f"{self.namespace}:*"))
            
            return {
                "server_info": info,
                "namespace": self.namespace,
                "namespace_keys": namespace_keys,
                "connection": {
                    "host": self.host,
                    "port": self.port,
                    "db": self.db
                }
            }
        except Exception as e:
            logger.error(f"Error getting Redis info: {str(e)}")
            return {"error": str(e)}
    
    def cache_market_data(self, symbol: str, data: Dict[str, Any], expiry: int = 3600) -> bool:
        """
        Cache market data for a symbol.
        
        Args:
            symbol: Ticker symbol
            data: Market data
            expiry: Expiration time in seconds (default: 1 hour)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create key
            key = f"market:{symbol}"
            
            # Set data with expiry
            return self.set(key, data, expiry)
        except Exception as e:
            logger.error(f"Error caching market data for {symbol}: {str(e)}")
            return False
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market data for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Market data or None if not cached
        """
        try:
            # Create key
            key = f"market:{symbol}"
            
            # Get data
            return self.get(key)
        except Exception as e:
            logger.error(f"Error getting cached market data for {symbol}: {str(e)}")
            return None
    
    def cache_dataframe(self, key: str, df: pd.DataFrame, expiry: int = 3600) -> bool:
        """
        Cache a pandas DataFrame.
        
        Args:
            key: Cache key
            df: DataFrame to cache
            expiry: Expiration time in seconds (default: 1 hour)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Set DataFrame with expiry
            return self.set(key, df, expiry)
        except Exception as e:
            logger.error(f"Error caching DataFrame for {key}: {str(e)}")
            return False
    
    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get a cached pandas DataFrame.
        
        Args:
            key: Cache key
            
        Returns:
            DataFrame or None if not cached
        """
        try:
            # Get DataFrame
            df = self.get(key)
            
            # Ensure it's a DataFrame
            if df is not None and not isinstance(df, pd.DataFrame):
                try:
                    # Try to convert to DataFrame
                    if isinstance(df, (dict, list)):
                        df = pd.DataFrame(df)
                    elif isinstance(df, str):
                        df = pd.read_json(df)
                    else:
                        logger.warning(f"Cached value for {key} is not a DataFrame and cannot be converted")
                        return None
                except Exception as e:
                    logger.error(f"Error converting cached value to DataFrame: {str(e)}")
                    return None
            
            return df
        except Exception as e:
            logger.error(f"Error getting cached DataFrame for {key}: {str(e)}")
            return None
    
    def cache_model_prediction(self, model_name: str, input_hash: str, prediction: Any, expiry: int = 86400) -> bool:
        """
        Cache a model prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of the input data
            prediction: Prediction to cache
            expiry: Expiration time in seconds (default: 24 hours)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create key
            key = f"model:{model_name}:{input_hash}"
            
            # Set prediction with expiry
            return self.set(key, prediction, expiry)
        except Exception as e:
            logger.error(f"Error caching prediction for {model_name}: {str(e)}")
            return False
    
    def get_model_prediction(self, model_name: str, input_hash: str) -> Optional[Any]:
        """
        Get a cached model prediction.
        
        Args:
            model_name: Name of the model
            input_hash: Hash of the input data
            
        Returns:
            Prediction or None if not cached
        """
        try:
            # Create key
            key = f"model:{model_name}:{input_hash}"
            
            # Get prediction
            return self.get(key)
        except Exception as e:
            logger.error(f"Error getting cached prediction for {model_name}: {str(e)}")
            return None
    
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
                serialized = pickle.dumps(input_data)
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
    
    @cache_decorator(expires=300)
    def cached_function_example(self, arg1: Any, arg2: Any) -> Any:
        """
        Example of a cached function using the decorator.
        
        Args:
            arg1: First argument
            arg2: Second argument
            
        Returns:
            Result
        """
        # This function's results will be cached for 5 minutes
        result = arg1 + arg2
        return result
    
    def close(self):
        """Close the Redis client."""
        try:
            # Close connection pool
            if hasattr(self.redis, 'connection_pool'):
                self.redis.connection_pool.disconnect()
                logger.info("Closed Redis connection pool")
        except Exception as e:
            logger.error(f"Error closing Redis client: {str(e)}")


# Create a singleton instance
_redis_client = None


def get_redis_client(config: Dict[str, Any] = None) -> RedisClient:
    """
    Get or create the Redis client instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        RedisClient instance
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            _redis_client = RedisClient(config)
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {str(e)}")
            # Return a disabled client that will gracefully handle failures
            class DisabledRedisClient:
                def __getattr__(self, name):
                    def noop(*args, **kwargs):
                        logger.debug(f"Redis disabled, ignoring call to {name}")
                        return None
                    return noop
                
                def health_check(self):
                    return {
                        "status": "disabled",
                        "connected": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return DisabledRedisClient()
    
    return _redis_client


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Redis Client')
    parser.add_argument('--host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--db', type=int, default=0, help='Redis database')
    parser.add_argument('--password', type=str, default=None, help='Redis password')
    parser.add_argument('--namespace', type=str, default='trading', help='Redis namespace')
    parser.add_argument('--info', action='store_true', help='Show Redis info')
    parser.add_argument('--flush', action='store_true', help='Flush namespace')
    
    args = parser.parse_args()
    
    # Create config from args
    config = {
        'host': args.host,
        'port': args.port,
        'db': args.db,
        'password': args.password,
        'namespace': args.namespace
    }
    
    # Create client
    redis_client = RedisClient(config)
    
    if args.info:
        # Show Redis info
        info = redis_client.get_info()
        
        print(f"Redis Server: {info['connection']['host']}:{info['connection']['port']}")
        print(f"Database: {info['connection']['db']}")
        print(f"Namespace: {info['namespace']}")
        print(f"Keys in namespace: {info['namespace_keys']}")
        print(f"Memory used: {info['server_info']['used_memory_human']}")
        print(f"Clients connected: {info['server_info']['connected_clients']}")
        print(f"Uptime: {info['server_info']['uptime_in_days']} days")
    
    if args.flush:
        # Flush namespace
        success = redis_client.flush()
        print(f"Flushed namespace {args.namespace}: {success}")
    
    # Close client
    redis_client.close()
