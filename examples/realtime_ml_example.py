#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time ML Example
------------------
This example demonstrates how to use the Polygon websocket collector
to provide real-time data to ML models.
"""

import argparse
import logging
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import threading
import signal
import sys

from config.collector_config import CollectorConfig
from data.collectors.factory import CollectorFactory
from data.collectors.polygon_websocket_collector import PolygonWebsocketCollector
from data.collectors.schema import RecordType
from data.processors.realtime_data_provider import RealtimeDataProvider
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('realtime_ml_example')

# Global flag for controlling the main loop
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit."""
    global running
    logger.info("Stopping example...")
    running = False


def start_websocket_collector(symbols: List[str]) -> PolygonWebsocketCollector:
    """
    Start the Polygon websocket collector.
    
    Args:
        symbols: List of symbols to subscribe to
        
    Returns:
        PolygonWebsocketCollector instance
    """
    logger.info(f"Starting Polygon websocket collector for symbols: {symbols}")
    
    # Create collector
    config = CollectorConfig.load('polygon')
    collector = CollectorFactory.create('polygon_websocket', config)
    
    # Add symbols
    for symbol in symbols:
        collector.add_symbol(symbol)
    
    # Start collector
    collector.start()
    
    return collector


def simulate_peak_detection_model(symbol: str) -> Dict[str, Any]:
    """
    Simulate the peak detection model using real-time data.
    
    Args:
        symbol: Symbol to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Get OHLCV data
    df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, limit=30)
    
    if df.empty:
        logger.warning(f"No data available for {symbol}")
        return {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'peak_probability': 0.0,
            'assessment': "No data available"
        }
    
    # Simple peak detection logic (this would be replaced by the actual ML model)
    # Check if price is near recent high and volume is increasing
    if len(df) < 5:
        return {
            'symbol': symbol,
            'timestamp': df['timestamp'].iloc[-1],
            'peak_probability': 0.0,
            'assessment': "Insufficient data points"
        }
    
    # Calculate some basic indicators
    df['price_pct_of_high'] = df['close'] / df['high'].rolling(10).max()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # Get latest values
    latest = df.iloc[-1]
    price_pct_of_high = latest.get('price_pct_of_high', 0)
    volume_ratio = latest.get('volume_ratio', 0)
    
    # Simple peak probability calculation
    peak_probability = 0.0
    
    if price_pct_of_high > 0.95:  # Price is near recent high
        peak_probability += 0.4
        
    if volume_ratio > 1.5:  # Volume is increasing
        peak_probability += 0.3
        
    if price_pct_of_high > 0.98 and volume_ratio > 2.0:
        peak_probability += 0.3
    
    # Determine assessment
    if peak_probability > 0.8:
        assessment = "Strong peak signal - consider exit"
    elif peak_probability > 0.6:
        assessment = "Moderate peak signal - monitor closely"
    elif peak_probability > 0.4:
        assessment = "Possible peak forming - be cautious"
    else:
        assessment = "No significant peak indication"
    
    return {
        'symbol': symbol,
        'timestamp': latest.get('timestamp', pd.Timestamp.now()),
        'close': latest.get('close', 0),
        'peak_probability': peak_probability,
        'price_pct_of_high': price_pct_of_high,
        'volume_ratio': volume_ratio,
        'assessment': assessment
    }


def simulate_entry_timing_model(symbol: str) -> Dict[str, Any]:
    """
    Simulate the entry timing model using real-time data.
    
    Args:
        symbol: Symbol to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Get OHLCV data
    df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, limit=30)
    
    if df.empty:
        logger.warning(f"No data available for {symbol}")
        return {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'entry_confidence': 0.0,
            'assessment': "No data available"
        }
    
    # Simple entry timing logic (this would be replaced by the actual ML model)
    if len(df) < 5:
        return {
            'symbol': symbol,
            'timestamp': df['timestamp'].iloc[-1],
            'entry_confidence': 0.0,
            'assessment': "Insufficient data points"
        }
    
    # Calculate some basic indicators
    df['price_pct_of_low'] = df['close'] / df['low'].rolling(10).min()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # Get latest values
    latest = df.iloc[-1]
    price_pct_of_low = latest.get('price_pct_of_low', 0)
    volume_ratio = latest.get('volume_ratio', 0)
    
    # Simple entry confidence calculation
    entry_confidence = 0.0
    
    if price_pct_of_low < 1.05:  # Price is near recent low
        entry_confidence += 0.4
        
    if volume_ratio > 1.5:  # Volume is increasing
        entry_confidence += 0.3
        
    if price_pct_of_low < 1.02 and volume_ratio > 2.0:
        entry_confidence += 0.3
    
    # Determine assessment
    if entry_confidence > 0.8:
        assessment = "Strong entry signal - consider entry"
    elif entry_confidence > 0.6:
        assessment = "Moderate entry signal - prepare for entry"
    elif entry_confidence > 0.4:
        assessment = "Possible entry point forming - monitor closely"
    else:
        assessment = "No significant entry indication"
    
    return {
        'symbol': symbol,
        'timestamp': latest.get('timestamp', pd.Timestamp.now()),
        'close': latest.get('close', 0),
        'entry_confidence': entry_confidence,
        'price_pct_of_low': price_pct_of_low,
        'volume_ratio': volume_ratio,
        'assessment': assessment
    }


def simulate_market_regime_model(symbols: List[str]) -> Dict[str, Any]:
    """
    Simulate the market regime model using real-time data.
    
    Args:
        symbols: List of symbols to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Get latest prices for all symbols
    prices = RealtimeDataProvider.get_multi_symbol_prices(symbols)
    
    if not prices:
        logger.warning("No price data available")
        return {
            'timestamp': pd.Timestamp.now(),
            'regime': "unknown",
            'confidence': 0.0,
            'assessment': "No data available"
        }
    
    # Simple market regime logic (this would be replaced by the actual ML model)
    # Calculate average price change
    price_changes = []
    
    for symbol in symbols:
        # Get OHLCV data
        df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, limit=10)
        
        if not df.empty and len(df) > 1:
            # Calculate price change
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            price_change = (last_price / first_price) - 1
            price_changes.append(price_change)
    
    if not price_changes:
        return {
            'timestamp': pd.Timestamp.now(),
            'regime': "unknown",
            'confidence': 0.0,
            'assessment': "Insufficient data points"
        }
    
    # Calculate average price change
    avg_price_change = sum(price_changes) / len(price_changes)
    
    # Calculate volatility
    volatility = np.std(price_changes)
    
    # Determine market regime
    regime = "unknown"
    confidence = 0.0
    
    if avg_price_change > 0.01:  # Trending up
        regime = "trending_up"
        confidence = min(abs(avg_price_change) * 50, 1.0)
    elif avg_price_change < -0.01:  # Trending down
        regime = "trending_down"
        confidence = min(abs(avg_price_change) * 50, 1.0)
    elif volatility > 0.01:  # High volatility
        regime = "high_volatility"
        confidence = min(volatility * 50, 1.0)
    else:  # Low volatility
        regime = "low_volatility"
        confidence = min(1.0 - volatility * 50, 1.0)
    
    # Determine assessment
    if regime == "trending_up":
        assessment = "Market is trending up - favor long positions"
    elif regime == "trending_down":
        assessment = "Market is trending down - favor short positions or cash"
    elif regime == "high_volatility":
        assessment = "Market is highly volatile - reduce position sizes"
    else:
        assessment = "Market has low volatility - normal position sizing"
    
    return {
        'timestamp': pd.Timestamp.now(),
        'regime': regime,
        'confidence': confidence,
        'avg_price_change': avg_price_change,
        'volatility': volatility,
        'assessment': assessment
    }


def run_ml_models(symbols: List[str], interval: int = 5):
    """
    Run ML models on real-time data.
    
    Args:
        symbols: List of symbols to analyze
        interval: Interval in seconds between model runs
    """
    logger.info(f"Starting ML models with {interval} second interval")
    
    while running:
        try:
            # Run peak detection model for each symbol
            for symbol in symbols:
                peak_result = simulate_peak_detection_model(symbol)
                logger.info(f"Peak Detection ({symbol}): {peak_result['assessment']} "
                           f"(probability: {peak_result['peak_probability']:.2f})")
                
                entry_result = simulate_entry_timing_model(symbol)
                logger.info(f"Entry Timing ({symbol}): {entry_result['assessment']} "
                           f"(confidence: {entry_result['entry_confidence']:.2f})")
            
            # Run market regime model
            regime_result = simulate_market_regime_model(symbols)
            logger.info(f"Market Regime: {regime_result['assessment']} "
                       f"(confidence: {regime_result['confidence']:.2f})")
            
            # Wait for next interval
            time.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error running ML models: {str(e)}")
            time.sleep(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time ML Example')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL',
                      help='Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)')
    parser.add_argument('--interval', type=int, default=5,
                      help='Interval in seconds between model runs (default: 5)')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start websocket collector
        collector = start_websocket_collector(symbols)
        
        # Wait for collector to connect and receive some data
        logger.info("Waiting for initial data collection...")
        time.sleep(5)
        
        # Run ML models
        run_ml_models(symbols, args.interval)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Stop collector
        if 'collector' in locals():
            collector.stop()
        
        logger.info("Example completed")


if __name__ == "__main__":
    main()
