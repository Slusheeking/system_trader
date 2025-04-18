#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Thresholds
------------------
Dynamically adjusts decision thresholds based on market conditions,
model performance, and trading results.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import threading
from collections import deque

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('adaptive_thresholds')


class ThresholdConfig:
    """
    Configuration for an adaptive threshold.
    """
    
    def __init__(self, name: str, initial_value: float, min_value: Optional[float] = None,
               max_value: Optional[float] = None, increment: Optional[float] = None,
               mode: str = 'market_regime', learning_rate: float = 0.1):
        """
        Initialize a threshold configuration.
        
        Args:
            name: Threshold name
            initial_value: Initial threshold value
            min_value: Minimum threshold value
            max_value: Maximum threshold value
            increment: Increment for manual adjustments
            mode: Adaptation mode ('market_regime', 'performance', 'manual', 'reinforcement')
            learning_rate: Learning rate for adaptive adjustments
        """
        self.name = name
        self.initial_value = initial_value
        self.min_value = min_value if min_value is not None else initial_value * 0.5
        self.max_value = max_value if max_value is not None else initial_value * 1.5
        self.increment = increment if increment is not None else (self.max_value - self.min_value) / 20
        self.mode = mode
        self.learning_rate = learning_rate


class ModelThresholds:
    """
    Manages thresholds for a specific model.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize model thresholds.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.thresholds = {}
        self.history = {}
        self.performance = {}
        
        # Initialize from config
        self._init_from_config(config or {})
    
    def _init_from_config(self, config: Dict[str, Any]):
        """
        Initialize thresholds from configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Get threshold configurations
        thresholds_config = config.get('thresholds', {})
        
        # Initialize thresholds based on model type
        if self.model_name == 'market_regime':
            # Market regime model thresholds
            self._add_threshold('bull_confidence', thresholds_config.get('bull_confidence', 0.7))
            self._add_threshold('bear_confidence', thresholds_config.get('bear_confidence', 0.7))
            self._add_threshold('neutral_confidence', thresholds_config.get('neutral_confidence', 0.6))
        
        elif self.model_name == 'stock_selection':
            # Stock selection model thresholds
            self._add_threshold('selection_confidence', thresholds_config.get('selection_confidence', 0.6))
            self._add_threshold('max_selections', thresholds_config.get('max_selections', 10))
            self._add_threshold('min_selections', thresholds_config.get('min_selections', 3))
        
        elif self.model_name == 'entry_timing':
            # Entry timing model thresholds
            self._add_threshold('entry_confidence', thresholds_config.get('entry_confidence', 0.75))
            self._add_threshold('max_entries_per_day', thresholds_config.get('max_entries_per_day', 5))
            self._add_threshold('min_time_between_entries', thresholds_config.get('min_time_between_entries', 30))  # minutes
        
        elif self.model_name == 'peak_detection':
            # Peak detection model thresholds
            self._add_threshold('exit_confidence', thresholds_config.get('exit_confidence', 0.7))
            self._add_threshold('trailing_stop', thresholds_config.get('trailing_stop', 0.03))  # 3%
            self._add_threshold('profit_target', thresholds_config.get('profit_target', 0.05))  # 5%
        
        elif self.model_name == 'risk_management':
            # Risk management model thresholds
            self._add_threshold('max_position_size', thresholds_config.get('max_position_size', 0.1))  # 10% of portfolio
            self._add_threshold('max_portfolio_risk', thresholds_config.get('max_portfolio_risk', 0.02))  # 2% portfolio risk
            self._add_threshold('correlation_threshold', thresholds_config.get('correlation_threshold', 0.7))
        
        else:
            # Generic thresholds for unknown model
            self._add_threshold('confidence', thresholds_config.get('confidence', 0.7))
            self._add_threshold('max_actions', thresholds_config.get('max_actions', 10))
    
    def _add_threshold(self, name: str, value_or_config: Union[float, Dict[str, Any]]):
        """
        Add a threshold.
        
        Args:
            name: Threshold name
            value_or_config: Threshold value or configuration
        """
        # Create threshold configuration
        if isinstance(value_or_config, (int, float)):
            config = ThresholdConfig(name=name, initial_value=float(value_or_config))
        else:
            config = ThresholdConfig(
                name=name,
                initial_value=value_or_config.get('initial_value', 0.7),
                min_value=value_or_config.get('min_value'),
                max_value=value_or_config.get('max_value'),
                increment=value_or_config.get('increment'),
                mode=value_or_config.get('mode', 'market_regime'),
                learning_rate=value_or_config.get('learning_rate', 0.1)
            )
        
        # Add threshold
        self.thresholds[name] = {
            'config': config,
            'value': config.initial_value,
            'last_updated': datetime.now(),
            'update_count': 0
        }
        
        # Initialize history
        self.history[name] = deque(maxlen=100)
        self.history[name].append((datetime.now(), config.initial_value))
        
        # Initialize performance
        self.performance[name] = {
            'regime_performance': {},
            'success_rate': 0.0,
            'sample_size': 0
        }
    
    def get_threshold(self, name: str) -> float:
        """
        Get a threshold value.
        
        Args:
            name: Threshold name
            
        Returns:
            Threshold value or None if not found
        """
        if name not in self.thresholds:
            logger.warning(f"Threshold {name} not found for model {self.model_name}")
            return None
        
        return self.thresholds[name]['value']
    
    def update_threshold(self, name: str, value: Optional[float] = None, 
                       direction: Optional[str] = None, increment: Optional[float] = None,
                       market_regime: Optional[str] = None, success: Optional[bool] = None) -> float:
        """
        Update a threshold value.
        
        Args:
            name: Threshold name
            value: New threshold value (for direct updates)
            direction: Update direction ('increase', 'decrease') for incremental updates
            increment: Custom increment value
            market_regime: Current market regime (for regime-based updates)
            success: Whether the last action was successful (for reinforcement updates)
            
        Returns:
            Updated threshold value
        """
        if name not in self.thresholds:
            logger.warning(f"Threshold {name} not found for model {self.model_name}")
            return None
        
        threshold = self.thresholds[name]
        config = threshold['config']
        current_value = threshold['value']
        
        # Calculate new value
        new_value = None
        
        if value is not None:
            # Direct update
            new_value = value
        
        elif direction is not None:
            # Incremental update
            inc = increment if increment is not None else config.increment
            if direction == 'increase':
                new_value = current_value + inc
            elif direction == 'decrease':
                new_value = current_value - inc
        
        elif market_regime is not None and config.mode == 'market_regime':
            # Market regime-based update
            # This is a simplified approach - a more sophisticated implementation could
            # use a formula based on market regime characteristics
            if market_regime == 'bull':
                # In bull markets, we might lower entry thresholds
                if 'entry' in name.lower() or 'selection' in name.lower():
                    new_value = current_value * 0.95  # 5% decrease
                # And increase exit thresholds
                elif 'exit' in name.lower() or 'stop' in name.lower():
                    new_value = current_value * 1.05  # 5% increase
            
            elif market_regime == 'bear':
                # In bear markets, we might increase entry thresholds
                if 'entry' in name.lower() or 'selection' in name.lower():
                    new_value = current_value * 1.1  # 10% increase
                # And lower exit thresholds
                elif 'exit' in name.lower() or 'stop' in name.lower():
                    new_value = current_value * 0.9  # 10% decrease
        
        elif success is not None and config.mode == 'reinforcement':
            # Reinforcement learning update
            if success:
                # Successful action - reinforce current threshold
                # No change or small adjustment
                new_value = current_value * (1 + 0.01 * config.learning_rate)
            else:
                # Unsuccessful action - adjust threshold
                if 'entry' in name.lower() or 'selection' in name.lower():
                    # For entry thresholds, increase after failure
                    new_value = current_value * (1 + 0.1 * config.learning_rate)
                elif 'exit' in name.lower() or 'stop' in name.lower():
                    # For exit thresholds, decrease after failure
                    new_value = current_value * (1 - 0.1 * config.learning_rate)
        
        # Apply new value if calculated
        if new_value is not None:
            # Ensure value is within bounds
            new_value = max(config.min_value, min(config.max_value, new_value))
            
            # Update threshold
            threshold['value'] = new_value
            threshold['last_updated'] = datetime.now()
            threshold['update_count'] += 1
            
            # Update history
            self.history[name].append((datetime.now(), new_value))
            
            logger.info(f"Updated threshold {name} for model {self.model_name}: {current_value:.4f} -> {new_value:.4f}")
        
        return threshold['value']
    
    def update_performance(self, name: str, success: bool, market_regime: Optional[str] = None):
        """
        Update performance metrics for a threshold.
        
        Args:
            name: Threshold name
            success: Whether the action was successful
            market_regime: Current market regime
        """
        if name not in self.thresholds:
            logger.warning(f"Threshold {name} not found for model {self.model_name}")
            return
        
        # Update overall performance
        performance = self.performance[name]
        current_success_count = performance['success_rate'] * performance['sample_size']
        performance['sample_size'] += 1
        
        if success:
            current_success_count += 1
        
        performance['success_rate'] = current_success_count / performance['sample_size']
        
        # Update regime-specific performance
        if market_regime:
            if market_regime not in performance['regime_performance']:
                performance['regime_performance'][market_regime] = {
                    'success_rate': 0.0,
                    'sample_size': 0
                }
            
            regime_perf = performance['regime_performance'][market_regime]
            current_regime_success = regime_perf['success_rate'] * regime_perf['sample_size']
            regime_perf['sample_size'] += 1
            
            if success:
                current_regime_success += 1
            
            regime_perf['success_rate'] = current_regime_success / regime_perf['sample_size']
    
    def reset_thresholds(self):
        """
        Reset all thresholds to their initial values.
        """
        for name, threshold in self.thresholds.items():
            config = threshold['config']
            threshold['value'] = config.initial_value
            threshold['last_updated'] = datetime.now()
            threshold['update_count'] += 1
            
            # Update history
            self.history[name].append((datetime.now(), config.initial_value))
            
            logger.info(f"Reset threshold {name} for model {self.model_name} to {config.initial_value:.4f}")
    
    def get_threshold_history(self, name: str) -> List[Dict[str, Any]]:
        """
        Get history for a threshold.
        
        Args:
            name: Threshold name
            
        Returns:
            List of (timestamp, value) tuples
        """
        if name not in self.history:
            logger.warning(f"Threshold {name} not found for model {self.model_name}")
            return []
        
        return [
            {
                'timestamp': timestamp.isoformat(),
                'value': value
            }
            for timestamp, value in self.history[name]
        ]
    
    def get_all_thresholds(self) -> Dict[str, Any]:
        """
        Get all thresholds.
        
        Returns:
            Dictionary with all threshold information
        """
        result = {}
        
        for name, threshold in self.thresholds.items():
            result[name] = {
                'value': threshold['value'],
                'initial_value': threshold['config'].initial_value,
                'min_value': threshold['config'].min_value,
                'max_value': threshold['config'].max_value,
                'mode': threshold['config'].mode,
                'last_updated': threshold['last_updated'].isoformat(),
                'update_count': threshold['update_count']
            }
        
        return result


class AdaptiveThresholds:
    """
    Dynamically adjusts decision thresholds based on market conditions,
    model performance, and trading results.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize adaptive thresholds.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model_thresholds = {}
        
        # Initialize model thresholds
        self._init_model_thresholds()
        
        # Track market regime
        self.current_regime = 'neutral'
        self.regime_confidence = 0.5
        
        # Track performance
        self.performance_history = {}
        
        # Auto adaptation
        self.auto_adapt_enabled = self.config.get('auto_adapt', {}).get('enabled', True)
        self.auto_adapt_interval = self.config.get('auto_adapt', {}).get('interval', 86400)  # 1 day
        self._stop_auto_adapt = threading.Event()
        self._auto_adapt_thread = None
        
        # Start auto-adapt thread if enabled
        if self.auto_adapt_enabled:
            self._start_auto_adapt_thread()
        
        logger.info("Adaptive thresholds initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config dictionary
        """
        if config_path is None:
            logger.info("No config path provided, using default configuration")
            return {}
        
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _init_model_thresholds(self):
        """
        Initialize thresholds for all models.
        """
        # Get model-specific configurations
        models_config = self.config.get('models', {})
        
        # Initialize common models
        model_names = [
            'market_regime',
            'stock_selection',
            'entry_timing',
            'peak_detection',
            'risk_management'
        ]
        
        for model_name in model_names:
            config = models_config.get(model_name, {})
            self.model_thresholds[model_name] = ModelThresholds(model_name, config)
            logger.info(f"Initialized thresholds for model {model_name}")
    
    def update_market_regime(self, regime: str, confidence: float):
        """
        Update current market regime.
        
        Args:
            regime: Market regime ('bull', 'bear', 'neutral')
            confidence: Confidence in regime classification
        """
        self.current_regime = regime
        self.regime_confidence = confidence
        logger.info(f"Updated market regime: {regime} (confidence: {confidence:.4f})")
    
    def get_threshold(self, model_name: str, threshold_name: str) -> Optional[float]:
        """
        Get a threshold value.
        
        Args:
            model_name: Name of the model
            threshold_name: Name of the threshold
            
        Returns:
            Threshold value or None if not found
        """
        if model_name not in self.model_thresholds:
            logger.warning(f"Model {model_name} not found")
            return None
        
        return self.model_thresholds[model_name].get_threshold(threshold_name)
    
    def update_threshold(self, model_name: str, threshold_name: str, value: Optional[float] = None,
                      direction: Optional[str] = None, increment: Optional[float] = None) -> Optional[float]:
        """
        Update a threshold value.
        
        Args:
            model_name: Name of the model
            threshold_name: Name of the threshold
            value: New threshold value (for direct updates)
            direction: Update direction ('increase', 'decrease') for incremental updates
            increment: Custom increment value
            
        Returns:
            Updated threshold value or None if update failed
        """
        if model_name not in self.model_thresholds:
            logger.warning(f"Model {model_name} not found")
            return None
        
        return self.model_thresholds[model_name].update_threshold(
            threshold_name, value, direction, increment
        )
    
    def record_performance(self, model_name: str, threshold_name: str, success: bool):
        """
        Record performance for a threshold.
        
        Args:
            model_name: Name of the model
            threshold_name: Name of the threshold
            success: Whether the action was successful
        """
        if model_name not in self.model_thresholds:
            logger.warning(f"Model {model_name} not found")
            return
        
        self.model_thresholds[model_name].update_performance(
            threshold_name, success, self.current_regime
        )
        
        # Track in overall performance history
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {}
        
        if threshold_name not in self.performance_history[model_name]:
            self.performance_history[model_name][threshold_name] = []
        
        self.performance_history[model_name][threshold_name].append({
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'regime': self.current_regime,
            'threshold_value': self.get_threshold(model_name, threshold_name)
        })
    
    def adapt_thresholds(self, model_name: Optional[str] = None):
        """
        Adapt thresholds based on market regime and performance.
        
        Args:
            model_name: Optional name of the model to adapt thresholds for
        """
        # Determine which models to adapt
        if model_name:
            models_to_adapt = [model_name] if model_name in self.model_thresholds else []
        else:
            models_to_adapt = list(self.model_thresholds.keys())
        
        for model in models_to_adapt:
            model_thresholds = self.model_thresholds[model]
            
            for name, threshold in model_thresholds.thresholds.items():
                config = threshold['config']
                
                # Skip manual thresholds
                if config.mode == 'manual':
                    continue
                
                # Adapt based on mode
                if config.mode == 'market_regime':
                    model_thresholds.update_threshold(
                        name, market_regime=self.current_regime
                    )
                elif config.mode == 'reinforcement':
                    # Use performance data to adapt
                    performance = model_thresholds.performance.get(name, {})
                    success_rate = performance.get('success_rate', 0.0)
                    sample_size = performance.get('sample_size', 0)
                    
                    if sample_size > 10:  # Only adapt after sufficient samples
                        success = success_rate > 0.5  # Consider successful if success rate > 50%
                        model_thresholds.update_threshold(
                            name, success=success
                        )
            
            logger.info(f"Adapted thresholds for model {model}")
    
    def reset_thresholds(self, model_name: Optional[str] = None):
        """
        Reset thresholds to their initial values.
        
        Args:
            model_name: Optional name of the model to reset thresholds for
        """
        # Determine which models to reset
        if model_name:
            models_to_reset = [model_name] if model_name in self.model_thresholds else []
        else:
            models_to_reset = list(self.model_thresholds.keys())
        
        for model in models_to_reset:
            self.model_thresholds[model].reset_thresholds()
            logger.info(f"Reset thresholds for model {model}")
    
    def get_all_thresholds(self) -> Dict[str, Any]:
        """
        Get all thresholds for all models.
        
        Returns:
            Dictionary with all threshold information
        """
        result = {}
        
        for model_name, model_thresholds in self.model_thresholds.items():
            result[model_name] = model_thresholds.get_all_thresholds()
        
        return result
    
    def get_threshold_history(self, model_name: str, threshold_name: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific threshold.
        
        Args:
            model_name: Name of the model
            threshold_name: Name of the threshold
            
        Returns:
            List of threshold history entries
        """
        if model_name not in self.model_thresholds:
            logger.warning(f"Model {model_name} not found")
            return []
        
        return self.model_thresholds[model_name].get_threshold_history(threshold_name)
    
    def adjust_results(self, results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Adjust model results based on thresholds.
        
        Args:
            results: Model results to adjust
            model_name: Name of the model
            
        Returns:
            Adjusted results
        """
        if model_name not in self.model_thresholds:
            logger.warning(f"Model {model_name} not found, returning original results")
            return results
        
        adjusted_results = results.copy()
        
        try:
            # Apply model-specific adjustments
            if model_name == 'market_regime':
                self._adjust_market_regime_results(adjusted_results)
            elif model_name == 'stock_selection':
                self._adjust_stock_selection_results(adjusted_results)
            elif model_name == 'entry_timing':
                self._adjust_entry_timing_results(adjusted_results)
            elif model_name == 'peak_detection':
                self._adjust_peak_detection_results(adjusted_results)
            elif model_name == 'risk_management':
                self._adjust_risk_management_results(adjusted_results)
        
        except Exception as e:
            logger.error(f"Error adjusting results for {model_name}: {str(e)}")
        
        # Add threshold information to the results
        if 'thresholds_applied' not in adjusted_results:
            adjusted_results['thresholds_applied'] = {}
        
        model_thresholds = self.model_thresholds[model_name]
        for name, threshold in model_thresholds.thresholds.items():
            adjusted_results['thresholds_applied'][name] = threshold['value']
        
        return adjusted_results
    
    def _adjust_market_regime_results(self, results: Dict[str, Any]):
        """
        Adjust market regime model results based on thresholds.
        
        Args:
            results: Market regime model results
        """
        # Get thresholds
        model_thresholds = self.model_thresholds['market_regime']
        bull_confidence = model_thresholds.get_threshold('bull_confidence')
        bear_confidence = model_thresholds.get_threshold('bear_confidence')
        neutral_confidence = model_thresholds.get_threshold('neutral_confidence')
        
        # Get regime and confidence from results
        regime = results.get('regime', 'unknown')
        confidence = results.get('confidence', 0.0)
        
        # Apply thresholds
        adjusted_regime = regime
        
        if regime == 'bull' and confidence < bull_confidence:
            adjusted_regime = 'neutral'
        elif regime == 'bear' and confidence < bear_confidence:
            adjusted_regime = 'neutral'
        elif regime == 'neutral' and confidence < neutral_confidence:
            # Default to neutral with low confidence
            pass
        
        # Update results
        if adjusted_regime != regime:
            results['original_regime'] = regime
            results['regime'] = adjusted_regime
            logger.info(f"Adjusted market regime: {regime} -> {adjusted_regime} (confidence: {confidence:.4f})")
        
        # Update current regime
        self.update_market_regime(adjusted_regime, confidence)
    
    def _adjust_stock_selection_results(self, results: Dict[str, Any]):
        """
        Adjust stock selection model results based on thresholds.
        
        Args:
            results: Stock selection model results
        """
        # Get thresholds
        model_thresholds = self.model_thresholds['stock_selection']
        selection_confidence = model_thresholds.get_threshold('selection_confidence')
        max_selections = int(model_thresholds.get_threshold('max_selections'))
        min_selections = int(model_thresholds.get_threshold('min_selections'))
        
        # Get selected symbols and data
        selected_symbols = results.get('selected_symbols', [])
        selection_data = results.get('selection_data', [])
        
        if not selected_symbols or not selection_data:
            return
        
        # If selection_data is a string (too large to include), skip detailed adjustments
        if isinstance(selection_data, str) and 'too large' in selection_data:
            # Just apply the max/min limits
            if len(selected_symbols) > max_selections:
                results['original_count'] = len(selected_symbols)
                results['selected_symbols'] = selected_symbols[:max_selections]
                results['selected_count'] = len(results['selected_symbols'])
                logger.info(f"Limited stock selections to {max_selections} (original: {results['original_count']})")
            return
        
        # Apply confidence threshold
        confident_selections = []
        if isinstance(selection_data, list):
            for selection in selection_data:
                if 'symbol' in selection and 'confidence' in selection:
                    if selection['confidence'] >= selection_confidence:
                        confident_selections.append(selection['symbol'])
        
        # Apply max/min selection limits
        if confident_selections:
            # Ensure we have at least min_selections
            if len(confident_selections) < min_selections:
                # If we don't have enough confident selections, use original selections up to min
                if len(selected_symbols) >= min_selections:
                    confident_selections = selected_symbols[:min_selections]
                else:
                    # Not enough symbols in either list
                    confident_selections = selected_symbols
            
            # Limit to max_selections
            if len(confident_selections) > max_selections:
                confident_selections = confident_selections[:max_selections]
            
            if set(confident_selections) != set(selected_symbols):
                results['original_symbols'] = selected_symbols
                results['original_count'] = len(selected_symbols)
                results['selected_symbols'] = confident_selections
                results['selected_count'] = len(confident_selections)
                logger.info(f"Adjusted stock selections: {len(selected_symbols)} -> {len(confident_selections)}")
    
    def _adjust_entry_timing_results(self, results: Dict[str, Any]):
        """
        Adjust entry timing model results based on thresholds.
        
        Args:
            results: Entry timing model results
        """
        # Get thresholds
        model_thresholds = self.model_thresholds['entry_timing']
        entry_confidence = model_thresholds.get_threshold('entry_confidence')
        max_entries_per_day = int(model_thresholds.get_threshold('max_entries_per_day'))
        min_time_between_entries = int(model_thresholds.get_threshold('min_time_between_entries'))
        
        # Get entry symbols and signals
        entry_symbols = results.get('entry_symbols', [])
        signals = results.get('signals', [])
        signals_by_symbol = results.get('signals_by_symbol', {})
        
        if not entry_symbols:
            return
        
        # If signals is a string (too large to include), skip detailed adjustments
        if isinstance(signals, str) and 'too large' in signals:
            # Just apply the max entries limit
            if len(entry_symbols) > max_entries_per_day:
                results['original_entry_count'] = len(entry_symbols)
                results['entry_symbols'] = entry_symbols[:max_entries_per_day]
                results['entry_count'] = len(results['entry_symbols'])
                logger.info(f"Limited entry signals to {max_entries_per_day} (original: {results['original_entry_count']})")
            return
        
        # Apply confidence threshold to signals
        confident_entries = []
        entry_signals = []
        
        if isinstance(signals, list):
            # Apply threshold to each signal
            for signal in signals:
                if 'symbol' in signal and 'entry_confidence' in signal:
                    if signal['entry_confidence'] >= entry_confidence:
                        if signal['symbol'] not in confident_entries:
                            confident_entries.append(signal['symbol'])
                            entry_signals.append(signal)
        elif isinstance(signals_by_symbol, dict):
            # Apply threshold to signals_by_symbol
            for symbol, symbol_signals in signals_by_symbol.items():
                if isinstance(symbol_signals, list) and symbol_signals:
                    top_signal = symbol_signals[0]  # Assume first signal is the strongest
                    if 'entry_confidence' in top_signal and top_signal['entry_confidence'] >= entry_confidence:
                        confident_entries.append(symbol)
                        entry_signals.append(top_signal)
        
        # Apply max entries limit
        if len(confident_entries) > max_entries_per_day:
            # Sort by confidence (if available)
            if entry_signals and 'entry_confidence' in entry_signals[0]:
                entry_signals.sort(key=lambda x: x['entry_confidence'], reverse=True)
                confident_entries = [signal['symbol'] for signal in entry_signals[:max_entries_per_day]]
            else:
                confident_entries = confident_entries[:max_entries_per_day]
        
        # Apply time between entries constraint
        # This would require tracking the times of previous entries
        # For simplicity, we'll skip this here
        
        # Update results if changed
        if set(confident_entries) != set(entry_symbols):
            results['original_entry_symbols'] = entry_symbols
            results['original_entry_count'] = len(entry_symbols)
            results['entry_symbols'] = confident_entries
            results['entry_count'] = len(confident_entries)
            logger.info(f"Adjusted entry signals: {len(entry_symbols)} -> {len(confident_entries)}")
    
    def _adjust_peak_detection_results(self, results: Dict[str, Any]):
        """
        Adjust peak detection model results based on thresholds.
        
        Args:
            results: Peak detection model results
        """
        # Get thresholds
        model_thresholds = self.model_thresholds['peak_detection']
        exit_confidence = model_thresholds.get_threshold('exit_confidence')
        trailing_stop = model_thresholds.get_threshold('trailing_stop')
        profit_target = model_thresholds.get_threshold('profit_target')
        
        # Get exit symbols and signals
        exit_symbols = results.get('exit_symbols', [])
        signals = results.get('signals', [])
        
        if not exit_symbols:
            return
        
        # If signals is a string (too large to include), skip detailed adjustments
        if isinstance(signals, str) and 'too large' in signals:
            return
        
        # Apply confidence threshold to signals
        confident_exits = []
        
        if isinstance(signals, list):
            # Apply threshold to each signal
            for signal in signals:
                if 'symbol' in signal and 'exit_confidence' in signal:
                    if signal['exit_confidence'] >= exit_confidence:
                        if signal['symbol'] not in confident_exits:
                            confident_exits.append(signal['symbol'])
        
        # Update results if changed
        if set(confident_exits) != set(exit_symbols):
            results['original_exit_symbols'] = exit_symbols
            results['original_exit_count'] = len(exit_symbols)
            results['exit_symbols'] = confident_exits
            results['exit_count'] = len(confident_exits)
            logger.info(f"Adjusted exit signals: {len(exit_symbols)} -> {len(confident_exits)}")
    
    def _adjust_risk_management_results(self, results: Dict[str, Any]):
        """
        Adjust risk management model results based on thresholds.
        
        Args:
            results: Risk management model results
        """
        # Get thresholds
        model_thresholds = self.model_thresholds['risk_management']
        max_position_size = model_thresholds.get_threshold('max_position_size')
        max_portfolio_risk = model_thresholds.get_threshold('max_portfolio_risk')
        
        # Check portfolio risk
        if 'portfolio_risk' in results:
            portfolio_risk = results['portfolio_risk']
            if portfolio_risk > max_portfolio_risk:
                results['original_portfolio_risk'] = portfolio_risk
                results['portfolio_risk'] = max_portfolio_risk
                logger.info(f"Adjusted portfolio risk: {portfolio_risk:.4f} -> {max_portfolio_risk:.4f}")
        
        # Check allocations
        if 'allocations' in results:
            allocations = results['allocations']
            adjusted_allocations = {}
            
            for symbol, allocation in allocations.items():
                if allocation > max_position_size:
                    adjusted_allocations[symbol] = max_position_size
                else:
                    adjusted_allocations[symbol] = allocation
            
            if adjusted_allocations != allocations:
                results['original_allocations'] = allocations
                results['allocations'] = adjusted_allocations
                logger.info(f"Adjusted position allocations to maximum of {max_position_size:.4f}")
    
    def _start_auto_adapt_thread(self):
        """
        Start the auto-adapt thread.
        """
        if self._auto_adapt_thread is not None and self._auto_adapt_thread.is_alive():
            return
        
        self._stop_auto_adapt.clear()
        self._auto_adapt_thread = threading.Thread(target=self._auto_adapt_loop)
        self._auto_adapt_thread.daemon = True
        self._auto_adapt_thread.start()
        logger.info("Auto-adapt thread started")
    
    def _stop_auto_adapt_thread(self):
        """
        Stop the auto-adapt thread.
        """
        if self._auto_adapt_thread is None or not self._auto_adapt_thread.is_alive():
            return
        
        self._stop_auto_adapt.set()
        self._auto_adapt_thread.join(timeout=1.0)
        logger.info("Auto-adapt thread stopped")
    
    def _auto_adapt_loop(self):
        """
        Auto-adapt loop.
        """
        while not self._stop_auto_adapt.is_set():
            try:
                # Adapt thresholds
                self.adapt_thresholds()
            except Exception as e:
                logger.error(f"Error in auto-adapt loop: {str(e)}")
            
            # Sleep for the interval
            self._stop_auto_adapt.wait(self.auto_adapt_interval)
    
    def __del__(self):
        """
        Cleanup on deletion.
        """
        self._stop_auto_adapt_thread()


# Default adaptive thresholds instance
default_adaptive_thresholds = None


def get_adaptive_thresholds(config_path: Optional[str] = None) -> AdaptiveThresholds:
    """
    Get or create the default adaptive thresholds.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AdaptiveThresholds instance
    """
    global default_adaptive_thresholds
    
    if default_adaptive_thresholds is None:
        default_adaptive_thresholds = AdaptiveThresholds(config_path)
    
    return default_adaptive_thresholds


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Thresholds for Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--thresholds', action='store_true', help='Get all thresholds')
    parser.add_argument('--adapt', action='store_true', help='Adapt thresholds')
    parser.add_argument('--reset', action='store_true', help='Reset thresholds')
    parser.add_argument('--model', type=str, help='Specific model')
    parser.add_argument('--threshold', type=str, help='Specific threshold')
    parser.add_argument('--value', type=float, help='Set threshold value')
    parser.add_argument('--direction', type=str, choices=['increase', 'decrease'], help='Adjust threshold direction')
    
    args = parser.parse_args()
    
    # Create adaptive thresholds
    adaptive_thresholds = AdaptiveThresholds(args.config)
    
    if args.thresholds:
        # Get thresholds
        if args.model and args.threshold:
            # Get specific threshold
            threshold_value = adaptive_thresholds.get_threshold(args.model, args.threshold)
            print(f"Threshold {args.threshold} for model {args.model}: {threshold_value}")
        else:
            # Get all thresholds
            thresholds = adaptive_thresholds.get_all_thresholds()
            import json
            print(json.dumps(thresholds, indent=2))
    
    elif args.adapt:
        # Adapt thresholds
        adaptive_thresholds.adapt_thresholds(args.model)
        print(f"Adapted thresholds for{f' model {args.model}' if args.model else ' all models'}")
    
    elif args.reset:
        # Reset thresholds
        adaptive_thresholds.reset_thresholds(args.model)
        print(f"Reset thresholds for{f' model {args.model}' if args.model else ' all models'}")
    
    elif args.model and args.threshold and (args.value is not None or args.direction):
        # Update specific threshold
        if args.value is not None:
            # Set value
            new_value = adaptive_thresholds.update_threshold(args.model, args.threshold, value=args.value)
            print(f"Updated threshold {args.threshold} for model {args.model} to {new_value}")
        elif args.direction:
            # Adjust in direction
            new_value = adaptive_thresholds.update_threshold(args.model, args.threshold, direction=args.direction)
            print(f"Adjusted threshold {args.threshold} for model {args.model} to {new_value}")
    
    else:
        parser.print_help()