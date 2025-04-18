#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polygon WebSocket Collector Test
--------------------------------
This script tests the Polygon WebSocket collector with enhanced data processing.
It demonstrates how to use the collector to receive real-time market data and
how the data flows through our processing pipeline.
"""

import argparse
import json
import logging
import time
import sys
import signal
from typing import Dict, List, Set, Any
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors.polygon_websocket_collector import PolygonWebsocketCollector
from data.collectors.schema import RecordType
from data.collectors.api_key_manager import get_api_key_manager, APIKeyException
from data.processors.data_cache import get_data_cache
from data.processors.realtime_data_provider import RealtimeDataProvider, DataSource
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('test_polygon_websocket', 'console')

# Initialize components
api_key_manager = get_api_key_manager()
data_cache = get_data_cache()

# Global variables
running = True
collector = None


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down the collector."""
    global running
    logger.info("Shutting down...")
    running = False
    if collector:
        collector.stop()
    sys.exit(0)


def validate_api_key():
    """Validate that the Polygon API key is configured correctly."""
    try:
        credentials = api_key_manager.get_api_key('polygon', validate=True)
        logger.info(f"✅ API key validated: {credentials['api_key'][:4]}...{credentials['api_key'][-4:]}")
        return True
    except APIKeyException as e:
        logger.error(f"❌ API key validation failed: {str(e)}")
        logger.error("Please check your config/api_credentials.yaml file or set the POLYGON_API_KEY environment variable")
        return False


def monitor_real_time_data(symbols: List[str], duration_seconds: int = 60):
    """
    Monitor real-time data for the specified symbols.
    
    Args:
        symbols: List of stock symbols to monitor
        duration_seconds: How long to monitor (in seconds)
    """
    global collector, running
    
    # Validate API key first
    if not validate_api_key():
        return
    
    # Create collector
    collector = PolygonWebsocketCollector()
    
    # Add symbols
    for symbol in symbols:
        collector.add_symbol(symbol)
    
    # Start collector
    collector.start()
    logger.info(f"Started WebSocket collector for {len(symbols)} symbols")
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Monitor data
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Setup counters
    message_counts = {symbol: {'trades': 0, 'quotes': 0, 'bars': 0} for symbol in symbols}
    last_status_time = start_time
    status_interval = 5  # Print status every 5 seconds
    
    try:
        while running and time.time() < end_time:
            current_time = time.time()
            
            # Check for new data
            for symbol in symbols:
                # Check for trades
                trades = data_cache.get_latest_websocket_data(symbol, 'trades', 1)
                if trades:
                    message_counts[symbol]['trades'] += 1
                
                # Check for quotes
                quotes = data_cache.get_latest_websocket_data(symbol, 'quotes', 1)
                if quotes:
                    message_counts[symbol]['quotes'] += 1
                
                # Check for bars
                bars = data_cache.get_latest_websocket_data(symbol, 'bars', 1)
                if bars:
                    message_counts[symbol]['bars'] += 1
            
            # Print status periodically
            if current_time - last_status_time >= status_interval:
                last_status_time = current_time
                elapsed = current_time - start_time
                remaining = max(0, duration_seconds - elapsed)
                
                # Get collector status
                status = collector.get_status()
                
                logger.info(f"Status after {elapsed:.1f}s (remaining: {remaining:.1f}s):")
                logger.info(f"Collector: {status['status']}, Messages: {status['message_count']}")
                
                # Print message counts by symbol
                for symbol in symbols:
                    counts = message_counts[symbol]
                    logger.info(f"{symbol}: Trades={counts['trades']}, Quotes={counts['quotes']}, Bars={counts['bars']}")
                
                logger.info("---")
            
            # Small delay to avoid CPU spinning
            time.sleep(0.1)
        
        # Final status
        logger.info(f"\nTest completed after {duration_seconds} seconds")
        
        # Get final counts
        for symbol in symbols:
            counts = message_counts[symbol]
            logger.info(f"{symbol} final counts: Trades={counts['trades']}, Quotes={counts['quotes']}, Bars={counts['bars']}")
        
        # Test RealtimeDataProvider access
        test_realtime_data_provider(symbols)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
    finally:
        # Stop collector
        if collector:
            collector.stop()
            logger.info("Stopped WebSocket collector")


def test_realtime_data_provider(symbols: List[str]):
    """
    Test the RealtimeDataProvider interface with collected data.
    
    Args:
        symbols: List of symbols to test
    """
    logger.info("\nTesting RealtimeDataProvider interface:")
    
    for symbol in symbols:
        logger.info(f"\nData for {symbol}:")
        
        # Get latest price
        price = RealtimeDataProvider.get_latest_price(symbol, DataSource.POLYGON)
        logger.info(f"Latest price: {price}")
        
        # Get latest OHLCV
        ohlcv = RealtimeDataProvider.get_latest_ohlcv(symbol, DataSource.POLYGON)
        if ohlcv:
            logger.info(f"Latest OHLCV: open={ohlcv.get('open')}, high={ohlcv.get('high')}, "
                        f"low={ohlcv.get('low')}, close={ohlcv.get('close')}, "
                        f"volume={ohlcv.get('volume')}")
        else:
            logger.info("No OHLCV data available")
        
        # Get recent trades
        trades_df = RealtimeDataProvider.get_recent_dataframe(symbol, RecordType.TRADE, 5, DataSource.POLYGON)
        if not trades_df.empty:
            logger.info(f"Recent trades:\n{trades_df[['timestamp', 'price', 'volume']].head()}")
        else:
            logger.info("No recent trades available")
        
        # Get recent quotes
        quotes_df = RealtimeDataProvider.get_recent_dataframe(symbol, RecordType.QUOTE, 5, DataSource.POLYGON)
        if not quotes_df.empty:
            logger.info(f"Recent quotes:\n{quotes_df[['timestamp', 'bid_price', 'ask_price']].head()}")
        else:
            logger.info("No recent quotes available")
        
        # Get OHLCV dataframe
        ohlcv_df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, 5, DataSource.POLYGON)
        if not ohlcv_df.empty:
            logger.info(f"OHLCV dataframe:\n{ohlcv_df.head()}")
        else:
            logger.info("No OHLCV dataframe available")


def check_data_quality(symbols: List[str]):
    """
    Check the quality of collected data.
    
    Args:
        symbols: List of symbols to check
    """
    logger.info("\nChecking data quality:")
    
    for symbol in symbols:
        # Get OHLCV dataframe
        ohlcv_df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, 100, DataSource.POLYGON)
        
        if ohlcv_df.empty:
            logger.info(f"{symbol}: No data available for quality check")
            continue
        
        # Check for missing values
        missing = ohlcv_df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"{symbol}: Missing values detected: {missing}")
        else:
            logger.info(f"{symbol}: No missing values")
        
        # Check for price anomalies
        if 'close' in ohlcv_df.columns:
            price_mean = ohlcv_df['close'].mean()
            price_std = ohlcv_df['close'].std()
            outliers = ohlcv_df[abs(ohlcv_df['close'] - price_mean) > 3 * price_std]
            
            if not outliers.empty:
                logger.warning(f"{symbol}: {len(outliers)} price outliers detected")
            else:
                logger.info(f"{symbol}: No price outliers detected")
        
        # Check for timestamp consistency
        if 'timestamp' in ohlcv_df.columns and len(ohlcv_df) > 1:
            ohlcv_df = ohlcv_df.sort_values('timestamp')
            diffs = ohlcv_df['timestamp'].diff().dropna()
            irregular = diffs.describe()
            
            logger.info(f"{symbol}: Timestamp intervals - min={irregular['min']}, "
                       f"max={irregular['max']}, mean={irregular['mean']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Polygon WebSocket Collector')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,AMZN,TSLA,GOOGL',
                      help='Comma-separated list of symbols to monitor')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration to run the test in seconds')
    parser.add_argument('--check-quality', action='store_true',
                      help='Perform data quality checks')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbol_list = [s.strip().upper() for s in args.symbols.split(',')]
    
    logger.info(f"Testing Polygon WebSocket Collector with {len(symbol_list)} symbols for {args.duration} seconds")
    logger.info(f"Symbols: {', '.join(symbol_list)}")
    
    # Run the test
    monitor_real_time_data(symbol_list, args.duration)
    
    # Data quality check if requested
    if args.check_quality:
        check_data_quality(symbol_list)
    
    logger.info("Test completed")