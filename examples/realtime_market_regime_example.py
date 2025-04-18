#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time Market Regime Detection Example
----------------------------------------
This example demonstrates how to use the Enhanced Market Regime Model
with real-time data from Polygon websocket and Unusual Whales.
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

from models.market_regime.model import EnhancedMarketRegimeModel
from data.processors.realtime_data_provider import RealtimeDataProvider
from data.collectors.factory import CollectorFactory
from config.collector_config import CollectorConfig
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('realtime_market_regime_example')


def load_model(model_path: str = None):
    """
    Load a pre-trained market regime model.
    
    Args:
        model_path: Path to the model files
        
    Returns:
        Loaded model
    """
    # Create model with default config
    config = {
        'n_regimes': 4,
        'lookback_window': 60,
        'smooth_window': 5,
        'hmm_n_iter': 100,
        'xgb_n_estimators': 100,
        'xgb_learning_rate': 0.1,
        'xgb_max_depth': 5,
        'feature_groups': {
            'returns': True,
            'volatility': True,
            'trend': True,
            'breadth': True,
            'sentiment': True,
            'options_flow': True
        }
    }
    
    model = EnhancedMarketRegimeModel(config)
    
    # Load model if path provided
    if model_path:
        success = model.load(model_path)
        if success:
            logger.info(f"Successfully loaded model from {model_path}")
        else:
            logger.warning(f"Failed to load model from {model_path}, using untrained model")
    else:
        logger.warning("No model path provided, using untrained model")
    
    return model


def start_data_collectors(symbols):
    """
    Start the Polygon websocket collector and ensure it's running.
    
    Args:
        symbols: List of symbols to collect data for
    """
    # Load Polygon websocket collector config
    config = CollectorConfig.load('polygon')
    
    # Create and start websocket collector
    collector = CollectorFactory.create('polygon_websocket', config)
    
    # Add symbols
    for symbol in symbols:
        collector.add_symbol(symbol)
    
    # Start collector
    collector.start()
    logger.info(f"Started Polygon websocket collector for symbols: {symbols}")
    
    # Wait for initial data collection
    logger.info("Waiting for initial data collection...")
    time.sleep(5)


def visualize_regime(regime_history, window=20):
    """
    Visualize the market regime history.
    
    Args:
        regime_history: List of regime prediction results
        window: Number of recent predictions to show
    """
    if len(regime_history) == 0:
        logger.warning("No regime history to visualize")
        return
    
    # Extract data for plotting
    recent = regime_history[-window:] if len(regime_history) > window else regime_history
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Regime timeline
    plt.subplot(3, 1, 1)
    
    # Extract data
    timestamps = [r['timestamp'] for r in recent]
    regimes = [r['regime'] for r in recent]
    
    # Create color map
    regime_colors = {
        'trending_up': 'green',
        'trending_down': 'red',
        'high_volatility': 'orange',
        'low_volatility': 'blue',
        'unknown': 'gray'
    }
    
    # Plot as colored background
    unique_regimes = list(set(regimes))
    for i, regime in enumerate(unique_regimes):
        mask = [r == regime for r in regimes]
        if any(mask):
            plt.fill_between(range(len(timestamps)), 0, 1, where=mask, 
                             color=regime_colors.get(regime, 'gray'), alpha=0.3, label=regime)
    
    plt.title('Market Regime Timeline')
    plt.ylabel('Regime')
    plt.xticks(range(len(timestamps)), [t.strftime('%H:%M') for t in timestamps], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Regime probabilities
    plt.subplot(3, 1, 2)
    
    # Extract probability data if available
    prob_data = {}
    for regime in ['trending_up', 'trending_down', 'high_volatility', 'low_volatility']:
        prob_key = f'prob_{regime}'
        if prob_key in recent[0]:
            prob_data[regime] = [r.get(prob_key, 0) for r in recent]
    
    # Plot probabilities
    for regime, probs in prob_data.items():
        plt.plot(range(len(timestamps)), probs, 
                 color=regime_colors.get(regime, 'gray'), 
                 marker='o', markersize=3, label=regime)
    
    plt.title('Regime Probabilities')
    plt.ylabel('Probability')
    plt.xticks(range(len(timestamps)), [t.strftime('%H:%M') for t in timestamps], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Options flow metrics if available
    plt.subplot(3, 1, 3)
    
    # Check which metrics are available
    metrics = []
    if 'put_call_ratio' in recent[0]:
        metrics.append('put_call_ratio')
    if 'smart_money_direction' in recent[0]:
        metrics.append('smart_money_direction')
    if 'unusual_activity_score' in recent[0]:
        metrics.append('unusual_activity_score')
    
    if metrics:
        for metric in metrics:
            values = [r.get(metric, 0) for r in recent]
            plt.plot(range(len(timestamps)), values, marker='o', markersize=3, label=metric)
        
        plt.title('Options Flow Metrics')
        plt.ylabel('Value')
        plt.xticks(range(len(timestamps)), [t.strftime('%H:%M') for t in timestamps], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title('No Options Flow Data Available')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description='Real-time Market Regime Detection Example')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ,IWM',
                        help='Comma-separated list of symbols to monitor')
    parser.add_argument('--interval', type=int, default=60,
                        help='Prediction interval in seconds')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pre-trained model')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize regime predictions')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Duration to run in seconds (default: 1 hour)')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Load model
    model = load_model(args.model)
    
    # Start data collectors
    start_data_collectors(symbols)
    
    # Main loop
    logger.info(f"Starting real-time market regime detection for symbols: {symbols}")
    logger.info(f"Prediction interval: {args.interval} seconds")
    
    start_time = time.time()
    end_time = start_time + args.duration
    
    # Store regime history for visualization
    regime_history = []
    
    try:
        while time.time() < end_time:
            # Make predictions for each symbol
            for symbol in symbols:
                # Get real-time prediction
                prediction = model.predict_realtime(symbol)
                
                # Check for errors
                if 'error' in prediction:
                    logger.warning(f"Error predicting for {symbol}: {prediction['error']}")
                    continue
                
                # Log prediction
                regime = prediction['regime']
                timestamp = prediction['timestamp']
                
                # Format confidence if available
                confidence_str = ""
                if f'prob_{regime}' in prediction:
                    confidence = prediction[f'prob_{regime}']
                    confidence_str = f" (confidence: {confidence:.2f})"
                
                logger.info(f"{symbol} at {timestamp}: {regime}{confidence_str}")
                
                # Log trading implications
                if 'trading_bias' in prediction:
                    logger.info(f"Trading implications: {prediction['trading_bias']} bias, "
                                f"{prediction['volatility_expectation']} volatility, "
                                f"suggested strategy: {prediction['suggested_strategy']}")
                
                # Add to history
                regime_history.append(prediction)
            
            # Visualize if requested and we have enough data
            if args.visualize and len(regime_history) > 0 and len(regime_history) % 5 == 0:
                visualize_regime(regime_history)
            
            # Wait for next interval
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    # Final visualization
    if args.visualize and len(regime_history) > 0:
        visualize_regime(regime_history)
    
    logger.info("Example completed")


if __name__ == "__main__":
    main()
