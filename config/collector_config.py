"""
Data Collector Configuration Module

This module contains the CollectorConfig class and helper functions
for loading and accessing configuration settings for all data collectors
in the system.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Union, Tuple
from utils.logging import setup_logger

logger = setup_logger(__name__, category='data')

# Default configurations for all collectors
# This replaces the content previously in collector_config.yaml

# Base configuration for all collectors
BASE_COLLECTOR_CONFIG = {
    "enabled": True,
    "timeout_seconds": 30,
    "retry_attempts": 3,
    "retry_delay_seconds": 2,
    "exponential_backoff": True,
    "cache_data": True,
    "cache_ttl_seconds": 60,
    "validate_data": True,
    "log_response_time": True,
    "concurrent_requests": 5,
    "concurrent_symbols": 20,
    "throttle_requests": True,
}

# Collector-specific default configurations
DEFAULT_CONFIGS = {
    "polygon": {
        **BASE_COLLECTOR_CONFIG,
        "api_key": "${POLYGON_API_KEY}",
        "base_url": "https://api.polygon.io/v2",
        "websocket_url": "wss://socket.polygon.io/stocks",
        "rate_limit_rpm": 200,
        "market_hours_only": True,
        "use_adjusted_values": True,
        "collection_mode": "both",  # market_data, news, or both
        "news_limit": 1000,
        "include_content": True,
        "news_sort": "published_utc",
        "news_order": "desc",
        "data_types": {
            "trades": {
                "enabled": True,
                "collection_interval_seconds": 1,
                "batch_size": 50,
                "fields": ["price", "size", "exchange", "conditions", "timestamp"],
            },
            "quotes": {
                "enabled": True,
                "collection_interval_seconds": 1,
                "batch_size": 50,
                "fields": ["bid_price", "bid_size", "ask_price", "ask_size", "timestamp"],
            },
            "bars": {
                "enabled": True,
                "collection_interval_seconds": 60,
                "batch_size": 100,
                "fields": ["open", "high", "low", "close", "volume", "vwap", "timestamp"],
                "timespan": "minute",
            },
            "snapshot": {
                "enabled": True,
                "collection_interval_seconds": 15,
                "batch_size": 200,
                "fields": ["price", "volume", "prev_close", "day_open", "day_high", "day_low"],
            },
        },
        "websocket": {
            "enabled": True,
            "channels": ["T", "Q", "AM"],
            "reconnect_interval_seconds": 30,
            "heartbeat_interval_seconds": 30,
        },
    },
    
    "yahoo": {
        **BASE_COLLECTOR_CONFIG,
        "api_key": "${YAHOO_API_KEY}",
        "use_rapid_api": True,
        "rapid_api_key": "${RAPID_API_KEY}",
        "rate_limit_rpm": 100,
        "data_types": {
            "quotes": {
                "enabled": True,
                "collection_interval_seconds": 15,
                "batch_size": 50,
                "fields": [
                    "regularMarketPrice", "regularMarketVolume", "regularMarketOpen",
                    "regularMarketDayHigh", "regularMarketDayLow", "regularMarketChange",
                    "regularMarketChangePercent", "bid", "ask", "bidSize", "askSize",
                    "marketCap", "averageDailyVolume3Month"
                ],
            },
            "historical": {
                "enabled": True,
                "collection_interval_seconds": 3600,
                "batch_size": 100,
                "fields": ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
                "period": "1d",
                "interval": "1m",
            },
            "options": {
                "enabled": True,
                "collection_interval_seconds": 300,
                "batch_size": 20,
                "fields": [
                    "contractSymbol", "strike", "expiration", "lastPrice", "bid", "ask",
                    "change", "volume", "openInterest", "impliedVolatility"
                ],
            },
        },
    },
    
    "alpaca": {
        **BASE_COLLECTOR_CONFIG,
        "api_key": "${ALPACA_API_KEY}",
        "api_secret": "${ALPACA_API_SECRET}",
        "base_url": "https://api.alpaca.markets",
        "data_url": "https://data.alpaca.markets",
        "paper_trading": False,
        "rate_limit_rpm": 200,
        "data_types": {
            "trades": {
                "enabled": True,
                "collection_interval_seconds": 1,
                "batch_size": 50,
                "fields": ["price", "size", "exchange", "timestamp"],
            },
            "quotes": {
                "enabled": True,
                "collection_interval_seconds": 1,
                "batch_size": 50,
                "fields": ["bid_price", "bid_size", "ask_price", "ask_size", "timestamp"],
            },
            "bars": {
                "enabled": True,
                "collection_interval_seconds": 60,
                "batch_size": 100,
                "fields": ["open", "high", "low", "close", "volume", "timestamp"],
                "timeframe": "1Min",
            },
            "account": {
                "enabled": True,
                "collection_interval_seconds": 60,
                "fields": ["equity", "buying_power", "cash", "portfolio_value"],
            },
            "positions": {
                "enabled": True,
                "collection_interval_seconds": 30,
                "fields": ["symbol", "qty", "avg_entry_price", "market_value", "unrealized_pl"],
            },
            "orders": {
                "enabled": True,
                "collection_interval_seconds": 10,
                "fields": ["id", "symbol", "side", "type", "qty", "status", "filled_qty", "filled_avg_price"],
            },
        },
        "websocket": {
            "enabled": True,
            "channels": ["trades", "quotes", "bars"],
            "reconnect_interval_seconds": 30,
            "heartbeat_interval_seconds": 30,
        },
    },
    
    "unusual_whales": {
        **BASE_COLLECTOR_CONFIG,
        "api_key": "${UNUSUAL_WHALES_API_KEY}",
        "base_url": "https://api.unusualwhales.com/api",
        "rate_limit_rpm": 60,
        "rate_limit_pd": 2000,
        "data_types": {
            "flow": {
                "enabled": True,
                "collection_interval_seconds": 300,
                "fields": [
                    "ticker", "strike", "expiry", "type", "size", "premium", "spot",
                    "iv", "openinterest", "volumeopeninterest", "sentiment"
                ],
                "filters": {
                    "min_premium": 10000,
                    "min_size": 50,
                    "min_volumeoi": 1.5,
                },
                "sentiment_threshold": 70,
            },
            "news": {
                "enabled": True,
                "collection_interval_seconds": 600,
                "fields": ["ticker", "title", "summary", "url", "sentiment", "timestamp"],
            },
        },
    },
    
    "reddit": {
        **BASE_COLLECTOR_CONFIG,
        "cache_ttl_seconds": 3600,
        "concurrent_requests": 3,
        "client_id": "${REDDIT_CLIENT_ID}",
        "client_secret": "${REDDIT_CLIENT_SECRET}",
        "user_agent": "system_trader:v1.0 (by /u/${REDDIT_USERNAME})",
        "username": "${REDDIT_USERNAME}",
        "password": "${REDDIT_PASSWORD}",
        "subreddits": [
            "wallstreetbets", "stocks", "investing", "options",
            "SecurityAnalysis", "StockMarket", "pennystocks"
        ],
        "post_limit": 100,
        "comment_limit": 50,
        "min_score": 5,
        "include_comments": True,
        "track_symbols": []  # Empty list means track all found symbols
    },
    
    # Data validation rules for all collectors
    "validation_rules": {
        "trades": {
            "price": {
                "type": "float",
                "min": 0.0001,
                "max": 100000,
            },
            "size": {
                "type": "int",
                "min": 1,
            },
            "timestamp": {
                "type": "timestamp",
                "max_age_seconds": 60,
            },
        },
        "quotes": {
            "bid_price": {
                "type": "float",
                "min": 0,
                "max": 100000,
            },
            "ask_price": {
                "type": "float",
                "min": 0,
                "max": 100000,
            },
            "bid_size": {
                "type": "int",
                "min": 0,
            },
            "ask_size": {
                "type": "int",
                "min": 0,
            },
            "timestamp": {
                "type": "timestamp",
                "max_age_seconds": 60,
            },
        },
        "bars": {
            "open": {
                "type": "float",
                "min": 0.0001,
                "max": 100000,
            },
            "high": {
                "type": "float",
                "min": 0.0001,
                "max": 100000,
            },
            "low": {
                "type": "float",
                "min": 0.0001,
                "max": 100000,
            },
            "close": {
                "type": "float",
                "min": 0.0001,
                "max": 100000,
            },
            "volume": {
                "type": "int",
                "min": 0,
            },
            "timestamp": {
                "type": "timestamp",
                "max_age_seconds": 3600,
            },
        },
    },
    
    # Anomaly detection rules
    "anomaly_rules": {
        "price_jump": {
            "threshold_pct": 10,
            "window_size": 5,
        },
        "volume_spike": {
            "threshold_multiple": 5,
            "window_size": 10,
        },
        "bid_ask_spread": {
            "max_spread_pct": 5,
        },
        "quote_age": {
            "max_age_seconds": 30,
        },
        "trading_halt": {
            "max_unchanged_periods": 5,
        },
        "zero_values": {
            "allowed_fields": ["volume", "bid_size", "ask_size"],
        },
    },
}


class CollectorConfig:
    """
    Configuration class for data collectors.
    
    This class handles loading and accessing configuration settings 
    from the default configurations and api_credentials.yaml.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize collector configuration.
        
        Args:
            config_dict: Dictionary with configuration parameters
        """
        self.config = config_dict or {}
        
    def __getattr__(self, name):
        """Get configuration attribute."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get(self, key, default=None):
        """
        Get a configuration value, with a default if not found.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CollectorConfig':
        """Create configuration from dictionary."""
        return cls(config_dict)
    
    @classmethod
    def load(cls, collector_name: str) -> 'CollectorConfig':
        """
        Load configuration for a specific collector from default configs.
        
        Args:
            collector_name: Name of the collector
            
        Returns:
            CollectorConfig instance
        """
        try:
            # Get default config for the collector
            collector_config = DEFAULT_CONFIGS.get(collector_name, {}).copy()
            if not collector_config:
                collector_config = BASE_COLLECTOR_CONFIG.copy()
                logger.warning(f"No default config found for {collector_name}, using base config")
            
            # Expand environment variables in API keys and secrets
            collector_config = cls._expand_env_vars(collector_config)
            
            # Try to load API keys from api_credentials.yaml
            try:
                api_credentials = cls._load_api_credentials()
                if api_credentials:
                    collector_config = cls._apply_api_credentials(collector_name, collector_config, api_credentials)
            except Exception as e:
                logger.warning(f"Could not load API credentials from file: {str(e)}")
            
            logger.info(f"Loaded configuration for {collector_name}")
            return cls(collector_config)
        except Exception as e:
            logger.error(f"Error loading configuration for {collector_name}: {str(e)}")
            return cls({})
    
    @staticmethod
    def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand environment variables in configuration values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with expanded environment variables
        """
        # Create a new dict to avoid modifying the original
        result = {}
        
        # Process each item in the config
        for key, value in config.items():
            # If value is a dictionary, process recursively
            if isinstance(value, dict):
                result[key] = CollectorConfig._expand_env_vars(value)
            # If value is a list, process each item
            elif isinstance(value, list):
                result[key] = [
                    CollectorConfig._expand_env_vars(item) if isinstance(item, dict) else item
                    for item in value
                ]
            # If value is a string that looks like an environment variable
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_name = value[2:-1]
                result[key] = os.environ.get(env_name, '')
            # Otherwise, keep the value as is
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def _load_api_credentials() -> Dict[str, Any]:
        """
        Load API credentials from file.
        
        Returns:
            API credentials dictionary
        """
        credentials_path = 'config/api_credentials.yaml'
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                credentials = yaml.safe_load(f)
            return credentials
        return {}
    
    @staticmethod
    def _apply_api_credentials(collector_name: str, config: Dict[str, Any], credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply API credentials to configuration.
        
        Args:
            collector_name: Name of the collector
            config: Configuration dictionary
            credentials: API credentials dictionary
            
        Returns:
            Updated configuration dictionary
        """
        # Create a new dict to avoid modifying the original
        result = config.copy()
        
        # Apply credentials based on collector type
        if collector_name == 'polygon' and 'Polygon' in credentials and 'API_Key' in credentials['Polygon']:
            polygon_key = credentials['Polygon']['API_Key']
            result['api_key'] = polygon_key
            logger.info(f"Using Polygon API key from credentials file: {polygon_key[:4]}...{polygon_key[-4:]}")
            
        elif collector_name == 'alpaca' and 'Alpaca' in credentials:
            if 'API_Key' in credentials['Alpaca']:
                result['api_key'] = credentials['Alpaca']['API_Key']
            if 'Secret_Key' in credentials['Alpaca']:
                result['api_secret'] = credentials['Alpaca']['Secret_Key']
            if 'Endpoint' in credentials['Alpaca']:
                result['base_url'] = credentials['Alpaca']['Endpoint']
                
        elif collector_name == 'unusual_whales' and 'Unusual_Whales' in credentials and 'API_Token' in credentials['Unusual_Whales']:
            result['api_key'] = credentials['Unusual_Whales']['API_Token']
            
        elif collector_name == 'reddit' and 'Reddit' in credentials:
            if 'Client_ID' in credentials['Reddit']:
                result['client_id'] = credentials['Reddit']['Client_ID']
            if 'Secret_Key' in credentials['Reddit']:
                result['client_secret'] = credentials['Reddit']['Secret_Key']
            if 'Username' in credentials['Reddit']:
                result['username'] = credentials['Reddit']['Username']
                # Construct user agent from username
                result['user_agent'] = f"system_trader:v1.0 (by /u/{credentials['Reddit']['Username']})"
            logger.info(f"Using Reddit credentials from credentials file")
        
        return result


# Helper functions
def get_collector_config(collector_name: str) -> CollectorConfig:
    """
    Get configuration for a specific collector.
    
    Args:
        collector_name: Name of the collector
        
    Returns:
        Collector configuration
    """
    return CollectorConfig.load(collector_name)


def get_validation_rules(data_type: str) -> Dict[str, Any]:
    """
    Get validation rules for a specific data type.
    
    Args:
        data_type: Type of data
        
    Returns:
        Validation rules for the data type
    """
    return DEFAULT_CONFIGS.get('validation_rules', {}).get(data_type, {})


def get_anomaly_rules() -> Dict[str, Any]:
    """
    Get anomaly detection rules.
    
    Returns:
        Anomaly detection rules
    """
    return DEFAULT_CONFIGS.get('anomaly_rules', {})


def is_collector_enabled(collector_name: str) -> bool:
    """
    Check if a collector is enabled.
    
    Args:
        collector_name: Name of the collector
        
    Returns:
        True if collector is enabled, False otherwise
    """
    config = get_collector_config(collector_name)
    return config.get("enabled", False)


def get_data_types_for_collector(collector_name: str) -> List[str]:
    """
    Get list of data types supported by a collector.
    
    Args:
        collector_name: Name of the collector
        
    Returns:
        List of supported data types
    """
    config = get_collector_config(collector_name)
    return list(config.get("data_types", {}).keys())


def get_collection_interval(collector_name: str, data_type: str) -> int:
    """
    Get collection interval for a specific data type.
    
    Args:
        collector_name: Name of the collector
        data_type: Type of data
        
    Returns:
        Collection interval in seconds
    """
    config = get_collector_config(collector_name)
    return config.get("data_types", {}).get(data_type, {}).get("collection_interval_seconds", 60)


def get_batch_size(collector_name: str, data_type: str) -> int:
    """
    Get batch size for a specific data type.
    
    Args:
        collector_name: Name of the collector
        data_type: Type of data
        
    Returns:
        Batch size
    """
    config = get_collector_config(collector_name)
    return config.get("data_types", {}).get(data_type, {}).get("batch_size", 50)


def get_fields(collector_name: str, data_type: str) -> List[str]:
    """
    Get fields for a specific data type.
    
    Args:
        collector_name: Name of the collector
        data_type: Type of data
        
    Returns:
        List of fields
    """
    config = get_collector_config(collector_name)
    return config.get("data_types", {}).get(data_type, {}).get("fields", [])
