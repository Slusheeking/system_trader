#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Collectors Package
----------------------
This package contains data collectors for various data sources.

The collectors follow a standardized interface through the BaseCollector
abstract class, making it easy to collect and process data from different
sources in a consistent way.

Each collector implements specific methods for its data source and provides
cached data access and database integration.

Factory functions are provided for easy collector instantiation.
"""

# Base classes and schemas
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType

# Collector classes
from data.collectors.polygon_collector import PolygonCollector
from data.collectors.polygon_websocket_collector import PolygonWebsocketCollector
from data.collectors.alpaca_collector import AlpacaCollector
from data.collectors.alpaca_websocket_collector import AlpacaWsCollector
from data.collectors.yahoo_collector import YahooCollector
from data.collectors.unusual_whales_collector import UnusualWhalesCollector
from data.collectors.reddit_collector import RedditCollector

# Factory for collector creation
from data.collectors.factory import CollectorFactory, get_collector

# Factory functions for individual collectors
from data.collectors.polygon_collector import create_polygon_collector
from data.collectors.polygon_websocket_collector import create_polygon_websocket_collector
from data.collectors.alpaca_collector import create_alpaca_collector
from data.collectors.alpaca_websocket_collector import create_alpaca_ws_collector
from data.collectors.yahoo_collector import create_yahoo_collector
from data.collectors.unusual_whales_collector import create_unusual_whales_collector
from data.collectors.reddit_collector import create_reddit_collector

__all__ = [
    # Base classes
    'BaseCollector',
    'CollectorError',
    'StandardRecord',
    'RecordType',
    
    # Collector classes
    'PolygonCollector',
    'PolygonWebsocketCollector',
    'AlpacaCollector',
    'AlpacaWsCollector',
    'YahooCollector',
    'RedditCollector',
    'UnusualWhalesCollector',
    
    # Factory
    'CollectorFactory',
    'get_collector',
    
    # Factory functions
    'create_polygon_collector',
    'create_polygon_websocket_collector',
    'create_alpaca_collector',
    'create_alpaca_ws_collector',
    'create_yahoo_collector',
    'create_unusual_whales_collector',
    'create_reddit_collector'
]
