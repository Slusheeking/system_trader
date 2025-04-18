#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collector Factory Module
-----------------------
This module provides factory classes and methods for creating
data collector instances based on configuration.
"""

from typing import Type, Dict, Optional, Any, Callable, List, Union
import logging
import importlib
import inspect

from utils.logging import setup_logger
from config.collector_config import CollectorConfig, get_collector_config
from data.collectors.base_collector import BaseCollector
from data.collectors.polygon_collector import PolygonCollector, create_polygon_collector
from data.collectors.polygon_websocket_collector import PolygonWebsocketCollector, create_polygon_websocket_collector
from data.collectors.yahoo_collector import YahooCollector, create_yahoo_collector
from data.collectors.alpaca_collector import AlpacaCollector, create_alpaca_collector
from data.collectors.alpaca_websocket_collector import AlpacaWebsocketCollector, create_alpaca_websocket_collector
from data.collectors.unusual_whales_collector import UnusualWhalesCollector, create_unusual_whales_collector
from data.collectors.reddit_collector import RedditCollector, create_reddit_collector

# Setup logging
logger = setup_logger('collector_factory', category='data')


class CollectorFactory:
    """
    Factory for creating data collector instances by name.
    
    This factory maintains a registry of collector types and provides
    methods for creating and registering collectors.
    """

    # Registry mapping collector identifiers to their classes or factory functions
    _registry: Dict[str, Union[Type[BaseCollector], Callable]] = {
        'polygon': create_polygon_collector,
        'polygon_websocket': create_polygon_websocket_collector,
        'yahoo': create_yahoo_collector,
        'alpaca': create_alpaca_collector,
        'alpaca_websocket': create_alpaca_websocket_collector,
        'unusual_whales': create_unusual_whales_collector,
        'reddit': create_reddit_collector
    }
    
    # Additional metadata for each collector
    _collector_info: Dict[str, Dict[str, Any]] = {
        'polygon': {
            'description': 'Polygon.io REST API collector for historical market data',
            'data_types': ['bars', 'trades', 'quotes', 'aggregates'],
            'subscription_required': True
        },
        'polygon_websocket': {
            'description': 'Polygon.io WebSocket collector for real-time market data',
            'data_types': ['trades', 'quotes', 'aggregate_minute', 'aggregate_second'],
            'subscription_required': True,
            'realtime': True,
            'batch_processing': True,
            'cache_integration': True,
            'data_validation': True
        },
        'yahoo': {
            'description': 'Yahoo Finance collector for historical market data',
            'data_types': ['bars'],
            'subscription_required': False
        },
        'alpaca': {
            'description': 'Alpaca Markets REST API collector for historical market data',
            'data_types': ['bars', 'trades', 'quotes'],
            'subscription_required': True
        },
        'alpaca_websocket': {
            'description': 'Alpaca Markets WebSocket collector for real-time market data and account updates',
            'data_types': ['trades', 'quotes', 'bars', 'account_updates', 'order_updates'],
            'subscription_required': True,
            'realtime': True,
            'batch_processing': True,
            'cache_integration': True,
            'data_validation': True
        },
        'unusual_whales': {
            'description': 'Unusual Whales API collector for options flow data',
            'data_types': ['options_flow'],
            'subscription_required': True
        },
        'reddit': {
            'description': 'Reddit API collector for social sentiment data',
            'data_types': ['posts', 'comments', 'sentiment'],
            'subscription_required': False
        }
    }

    @classmethod
    def register_collector(cls, name: str, collector_factory: Callable, collector_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new collector factory function under a given name.

        Args:
            name: Identifier for the collector (case-insensitive)
            collector_factory: Factory function that creates a collector instance
            collector_info: Optional metadata about the collector

        Raises:
            KeyError: If a collector is already registered under the name
        """
        key = name.lower()
        if key in cls._registry:
            raise KeyError(f"Collector '{name}' is already registered.")
        
        cls._registry[key] = collector_factory
        
        if collector_info:
            cls._collector_info[key] = collector_info
        else:
            # Add minimal info if not provided
            cls._collector_info[key] = {
                'description': f'Custom collector: {name}',
                'data_types': ['custom'],
                'subscription_required': False
            }
        
        logger.info(f"Registered collector: {name}")

    @classmethod
    def create(cls, name: str, config: Optional[CollectorConfig] = None, **kwargs) -> BaseCollector:
        """
        Instantiate a collector by its registered name using the provided config.

        Args:
            name: Identifier of the collector (case-insensitive)
            config: Optional CollectorConfig instance for initializing the collector.
                   If None, the config will be loaded using the collector name.
            **kwargs: Additional arguments to pass to the collector factory

        Returns:
            An instance of the requested BaseCollector subclass

        Raises:
            ValueError: If the name is not found in the registry
            TypeError: If the instantiated collector does not subclass BaseCollector
        """
        key = name.lower()
        if key not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Collector '{name}' is not registered. Available collectors: {available}"
            )

        # Load config if not provided
        if config is None:
            config = get_collector_config(name)
        
        # Get factory or class from registry
        factory_or_class = cls._registry[key]
        
        try:
            # If it's a factory function, call it with config and kwargs
            if callable(factory_or_class) and not inspect.isclass(factory_or_class):
                instance = factory_or_class(config, **kwargs)
            # If it's a class, instantiate it with config and kwargs
            elif inspect.isclass(factory_or_class) and issubclass(factory_or_class, BaseCollector):
                instance = factory_or_class(config, **kwargs)
            else:
                instance = factory_or_class(config)
            
            # Verify the instance is a BaseCollector
            if not isinstance(instance, BaseCollector):
                raise TypeError(
                    f"Collector '{name}' must subclass BaseCollector, got {type(instance).__name__}."
                )
            
            logger.debug(f"Created {instance.__class__.__name__} instance via factory")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create collector '{name}': {str(e)}")
            raise

    @classmethod
    def get_available_collectors(cls) -> List[Dict[str, Any]]:
        """
        Get information about all available collectors.
        
        Returns:
            List of dictionaries with collector information
        """
        collectors = []
        for name, factory in cls._registry.items():
            info = cls._collector_info.get(name, {})
            collectors.append({
                'name': name,
                'description': info.get('description', 'No description available'),
                'data_types': info.get('data_types', []),
                'subscription_required': info.get('subscription_required', False),
                'realtime': info.get('realtime', False),
                'batch_processing': info.get('batch_processing', False),
                'cache_integration': info.get('cache_integration', False),
                'data_validation': info.get('data_validation', False)
            })
        return collectors
    
    @classmethod
    def get_collector_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a specific collector.
        
        Args:
            name: Collector name
            
        Returns:
            Dictionary with collector information
            
        Raises:
            ValueError: If the collector is not registered
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Collector '{name}' is not registered.")
        
        info = cls._collector_info.get(key, {})
        return {
            'name': key,
            'description': info.get('description', 'No description available'),
            'data_types': info.get('data_types', []),
            'subscription_required': info.get('subscription_required', False),
            'realtime': info.get('realtime', False),
            'batch_processing': info.get('batch_processing', False),
            'cache_integration': info.get('cache_integration', False),
            'data_validation': info.get('data_validation', False)
        }
    
    @classmethod
    def unregister_collector(cls, name: str) -> None:
        """
        Unregister a collector.
        
        Args:
            name: Collector name
            
        Raises:
            ValueError: If the collector is not registered
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Collector '{name}' is not registered.")
        
        del cls._registry[key]
        if key in cls._collector_info:
            del cls._collector_info[key]
        
        logger.info(f"Unregistered collector: {name}")
    
    @classmethod
    def load_custom_collector(cls, module_path: str, collector_name: str, register_as: Optional[str] = None) -> None:
        """
        Load a custom collector from a Python module.
        
        Args:
            module_path: Path to the module (dot notation)
            collector_name: Name of the collector class or factory function in the module
            register_as: Optional alternative name to register the collector under
            
        Raises:
            ImportError: If the module or collector cannot be imported
            ValueError: If the collector doesn't implement BaseCollector
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the collector factory or class
            if hasattr(module, collector_name):
                collector = getattr(module, collector_name)
            else:
                raise ImportError(f"Module {module_path} does not contain {collector_name}")
            
            # Check if it's a valid collector factory or class
            if callable(collector):
                # Register it
                name = register_as or collector_name.lower()
                cls.register_collector(name, collector)
                logger.info(f"Loaded and registered custom collector: {name}")
            else:
                raise ValueError(f"{collector_name} must be a callable factory function or collector class")
            
        except Exception as e:
            logger.error(f"Failed to load custom collector: {str(e)}")
            raise

    @classmethod
    def get_websocket_collectors(cls) -> List[Dict[str, Any]]:
        """
        Get information about all available WebSocket collectors.
        
        Returns:
            List of dictionaries with collector information for WebSocket collectors
        """
        return [
            collector for collector in cls.get_available_collectors()
            if collector.get('realtime', False)
        ]


# Helper function to get a collector instance
def get_collector(name: str, config: Optional[CollectorConfig] = None, **kwargs) -> BaseCollector:
    """
    Get a collector instance by name.
    
    Args:
        name: Collector name
        config: Optional configuration (if None, loads from config file)
        **kwargs: Additional arguments to pass to the collector factory
        
    Returns:
        Collector instance
    """
    return CollectorFactory.create(name, config, **kwargs)


# Helper functions for WebSocket collectors
def get_websocket_collector(name: str, config: Optional[CollectorConfig] = None, **kwargs) -> BaseCollector:
    """
    Get a WebSocket collector instance by name.
    
    Args:
        name: WebSocket collector name (e.g., 'polygon_websocket', 'alpaca_websocket')
        config: Optional configuration
        **kwargs: Additional arguments
        
    Returns:
        WebSocket collector instance
    """
    if name.lower() not in ['polygon_websocket', 'alpaca_websocket']:
        raise ValueError(f"'{name}' is not a recognized WebSocket collector")
    
    return get_collector(name, config, **kwargs)


if __name__ == "__main__":
    # List available collectors for testing
    available = CollectorFactory.get_available_collectors()
    print("Available collectors:")
    for collector in available:
        print(f"- {collector['name']}: {collector['description']}")
        print(f"  Data types: {', '.join(collector['data_types'])}")
        print(f"  Subscription required: {'Yes' if collector['subscription_required'] else 'No'}")
        if collector.get('realtime'):
            print("  Realtime data: Yes")
        if collector.get('batch_processing'):
            print("  Batch processing: Yes")
        if collector.get('cache_integration'):
            print("  Cache integration: Yes")
        if collector.get('data_validation'):
            print("  Data validation: Yes")
        print()
    
    # List WebSocket collectors
    print("\nWebSocket collectors:")
    websocket_collectors = CollectorFactory.get_websocket_collectors()
    for collector in websocket_collectors:
        print(f"- {collector['name']}: {collector['description']}")
        print(f"  Data types: {', '.join(collector['data_types'])}")
        print()
    
    # Test creating a collector
    print("Creating a Yahoo collector...")
    yahoo = get_collector('yahoo', symbol='AAPL')
    print(f"Created: {yahoo.__class__.__name__}")
