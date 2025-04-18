#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest fixtures for system_trader tests.

This module provides common fixtures used across test modules:
- fake_config: Creates minimal configuration files for testing
- dummy_tracker: Provides a stub MLflow tracker
- fake_strategy_composer: Returns a stub strategy composer with signal generation
"""

import os
import json
import yaml
import pytest
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture
def fake_config(tmp_path):
    """
    Create minimal configuration files for testing.
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory path
        
    Returns:
        dict: Dictionary containing paths to generated config files
    """
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create scheduler config
    scheduler_config = {
        "training": {
            "frequency": "daily",
            "time": "01:00"
        },
        "backtesting": {
            "schedule": {
                "frequency": "daily",
                "time": "02:00"
            },
            "params": {
                "data_path": str(tmp_path / "data" / "market_data.csv")
            }
        }
    }
    
    scheduler_config_path = config_dir / "scheduler_config.yaml"
    with open(scheduler_config_path, "w") as f:
        yaml.dump(scheduler_config, f)
    
    # Create backtesting config
    backtesting_config = {
        "data": {
            "path": str(tmp_path / "data" / "market_data.csv")
        },
        "strategy": {
            "name": "test_strategy",
            "config_path": str(config_dir / "strategy_config.yaml")
        },
        "parameters": {
            "initial_capital": 100000,
            "commission": 0.001
        }
    }
    
    backtesting_config_path = config_dir / "backtesting_config.yaml"
    with open(backtesting_config_path, "w") as f:
        yaml.dump(backtesting_config, f)
    
    # Create strategy config
    strategy_config = {
        "strategies": {
            "test_strategy": {
                "type": "trend_following",
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            }
        }
    }
    
    strategy_config_path = config_dir / "strategy_config.yaml"
    with open(strategy_config_path, "w") as f:
        yaml.dump(strategy_config, f)
    
    # Create data directory and sample market data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create a minimal market data CSV file
    market_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    market_data_path = data_dir / "market_data.csv"
    market_data.to_csv(market_data_path, index=False)
    
    return {
        "scheduler_config_path": str(scheduler_config_path),
        "backtesting_config_path": str(backtesting_config_path),
        "strategy_config_path": str(strategy_config_path),
        "market_data_path": str(market_data_path),
        "config_dir": str(config_dir),
        "data_dir": str(data_dir)
    }


@pytest.fixture
def dummy_tracker():
    """
    Create a stub MLflow tracker for testing.
    
    Returns:
        MagicMock: A mock object with MLflow tracker interface
    """
    tracker = MagicMock()
    
    # Mock the common MLflow tracking methods
    tracker.log_metric = MagicMock(return_value=None)
    tracker.log_param = MagicMock(return_value=None)
    tracker.log_artifact = MagicMock(return_value=None)
    tracker.set_tag = MagicMock(return_value=None)
    
    # Mock the start_run context manager
    run = MagicMock()
    run.__enter__ = MagicMock(return_value=run)
    run.__exit__ = MagicMock(return_value=None)
    run.info = MagicMock()
    run.info.run_id = "test_run_id"
    tracker.start_run = MagicMock(return_value=run)
    
    # Add a method to verify calls
    tracker.get_call_count = lambda method_name: getattr(tracker, method_name).call_count
    
    return tracker


@pytest.fixture
def fake_strategy_composer(monkeypatch):
    """
    Create a stub strategy composer with signal generation capabilities.
    
    Args:
        monkeypatch: Pytest fixture for patching objects
        
    Returns:
        MagicMock: A mock strategy composer object
    """
    composer = MagicMock()
    
    # Mock the generate_signals method to return a DataFrame with signals
    def generate_signals(data_df):
        signals_df = data_df.copy()
        # Add signal columns
        signals_df['regime'] = 'bullish'  # Default regime
        signals_df['positions'] = [['AAPL_LONG'] if i % 2 == 0 else [] for i in range(len(signals_df))]
        signals_df['exits'] = [[] if i % 3 == 0 else ['AAPL_LONG'] for i in range(len(signals_df))]
        return signals_df
    
    composer.generate_signals = MagicMock(side_effect=generate_signals)
    
    # Mock the process_signals method
    def process_signals(signals):
        # Return a dictionary with entries and exits
        return {
            'entries': [{'symbol': 'AAPL', 'direction': 'LONG', 'price': 150.0, 'size': 10}] 
                      if signals.get('positions') else [],
            'exits': [{'symbol': 'AAPL', 'direction': 'LONG', 'price': 160.0, 'size': 10}] 
                    if signals.get('exits') else []
        }
    
    # Create a strategy mock with process_signals
    strategy = MagicMock()
    strategy.process_signals = MagicMock(side_effect=process_signals)
    composer.get_strategy = MagicMock(return_value=strategy)
    
    # Patch the import if needed
    if monkeypatch:
        monkeypatch.setattr('trading.strategy.get_strategy_composer', 
                           lambda config_path: composer)
    
    return composer