#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decision Framework
----------------
Reconciles signals from multiple models to produce final trading decisions.
Implements rules, priority systems, and conflict resolution strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('decision_framework')


class SignalPriority:
    """
    Defines the priority and weighting of signals from different models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize signal priority system.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default priorities (higher value = higher priority)
        self.priorities = {
            'market_regime': 10,
            'stock_selection': 7,
            'entry_timing': 8,
            'peak_detection': 9,
            'risk_management': 6
        }
        
        # Default weights for weighted combining
        self.weights = {
            'market_regime': 0.3,
            'stock_selection': 0.15,
            'entry_timing': 0.25,
            'peak_detection': 0.2,
            'risk_management': 0.1
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'market_regime': 0.7,
            'stock_selection': 0.6,
            'entry_timing': 0.75,
            'peak_detection': 0.7,
            'risk_management': 0.6
        }
        
        # Override defaults with provided config
        if config:
            if 'priorities' in config:
                self.priorities.update(config['priorities'])
            if 'weights' in config:
                self.weights.update(config['weights'])
            if 'confidence_thresholds' in config:
                self.confidence_thresholds.update(config['confidence_thresholds'])
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for model in self.weights:
                self.weights[model] /= total_weight
    
    def get_priority(self, model_name: str) -> int:
        """
        Get the priority of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Priority value (higher value = higher priority)
        """
        return self.priorities.get(model_name, 0)
    
    def get_weight(self, model_name: str) -> float:
        """
        Get the weight of a model for weighted combining.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Weight value (0.0 to 1.0)
        """
        return self.weights.get(model_name, 0.0)
    
    def get_confidence_threshold(self, model_name: str) -> float:
        """
        Get the confidence threshold for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Confidence threshold value (0.0 to 1.0)
        """
        return self.confidence_thresholds.get(model_name, 0.5)
    
    def is_above_threshold(self, model_name: str, confidence: float) -> bool:
        """
        Check if a confidence value is above the threshold for a model.
        
        Args:
            model_name: Name of the model
            confidence: Confidence value to check
            
        Returns:
            Boolean indicating if confidence is above threshold
        """
        threshold = self.get_confidence_threshold(model_name)
        return confidence >= threshold


class ConflictResolution:
    """
    Resolves conflicts between model signals.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize conflict resolution system.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default conflict resolution modes
        self.resolution_modes = {
            'entry_exit': 'exit_priority',  # When entry and exit signals conflict, prioritize exit
            'regime_entry': 'regime_veto',  # Market regime can veto entry decisions
            'allocation': 'risk_priority'   # Risk model overrides allocation decisions
        }
        
        # Override defaults with provided config
        if config and 'resolution_modes' in config:
            self.resolution_modes.update(config['resolution_modes'])
    
    def resolve_entry_exit_conflict(self, entry_signals: Dict[str, Any], 
                                   exit_signals: Dict[str, Any],
                                   market_regime: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Resolve conflicts between entry and exit signals.
        
        Args:
            entry_signals: Dictionary with entry signals
            exit_signals: Dictionary with exit signals
            market_regime: Optional market regime information
            
        Returns:
            Dictionary with resolved signals
        """
        resolution_mode = self.resolution_modes.get('entry_exit', 'exit_priority')
        
        # Get symbols with entry and exit signals
        entry_symbols = entry_signals.get('entry_symbols', [])
        exit_symbols = exit_signals.get('exit_symbols', [])
        
        # Find conflicts (symbols with both entry and exit signals)
        conflicts = set(entry_symbols).intersection(set(exit_symbols))
        
        resolved_entry = []
        resolved_exit = []
        
        if resolution_mode == 'exit_priority':
            # Prioritize exit signals over entry signals
            resolved_entry = [s for s in entry_symbols if s not in conflicts]
            resolved_exit = exit_symbols
            
        elif resolution_mode == 'entry_priority':
            # Prioritize entry signals over exit signals
            resolved_entry = entry_symbols
            resolved_exit = [s for s in exit_symbols if s not in conflicts]
            
        elif resolution_mode == 'regime_dependent':
            # Use market regime to resolve conflicts
            if market_regime and 'regime' in market_regime:
                regime = market_regime['regime']
                
                if regime == 'bull':
                    # In bull markets, favor entries
                    resolved_entry = entry_symbols
                    resolved_exit = [s for s in exit_symbols if s not in conflicts]
                elif regime == 'bear':
                    # In bear markets, favor exits
                    resolved_entry = [s for s in entry_symbols if s not in conflicts]
                    resolved_exit = exit_symbols
                else:  # neutral or unknown
                    # In neutral markets, use exit_priority
                    resolved_entry = [s for s in entry_symbols if s not in conflicts]
                    resolved_exit = exit_symbols
            else:
                # Default to exit_priority if no regime information
                resolved_entry = [s for s in entry_symbols if s not in conflicts]
                resolved_exit = exit_symbols
        
        return {
            'entry_symbols': resolved_entry,
            'exit_symbols': resolved_exit,
            'conflicts': list(conflicts),
            'resolution_mode': resolution_mode
        }
    
    def apply_regime_veto(self, entry_signals: Dict[str, Any], 
                         market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply market regime veto to entry signals.
        
        Args:
            entry_signals: Dictionary with entry signals
            market_regime: Market regime information
            
        Returns:
            Dictionary with vetoed entry signals
        """
        resolution_mode = self.resolution_modes.get('regime_entry', 'regime_veto')
        
        if resolution_mode != 'regime_veto':
            # No veto applied
            return entry_signals
        
        # Get regime information
        regime = market_regime.get('regime', 'unknown')
        regime_confidence = market_regime.get('confidence', 0.5)
        
        # Get entry symbols
        entry_symbols = entry_signals.get('entry_symbols', [])
        entry_count = len(entry_symbols)
        
        # Initialize vetoed signals
        vetoed_entry = entry_symbols.copy()
        vetoed_symbols = []
        
        # Apply veto based on regime
        if regime == 'bear' and regime_confidence > 0.7:
            # In strong bear markets, veto all entry signals
            vetoed_entry = []
            vetoed_symbols = entry_symbols
        elif regime == 'bear' and regime_confidence > 0.5:
            # In moderate bear markets, reduce entry signals
            vetoed_count = int(entry_count * 0.7)  # Veto 70% of entries
            if vetoed_count > 0 and entry_count > 0:
                vetoed_entry = entry_symbols[:(entry_count - vetoed_count)]
                vetoed_symbols = entry_symbols[(entry_count - vetoed_count):]
        elif regime == 'neutral':
            # In neutral markets, moderate reduction
            vetoed_count = int(entry_count * 0.3)  # Veto 30% of entries
            if vetoed_count > 0 and entry_count > 0:
                vetoed_entry = entry_symbols[:(entry_count - vetoed_count)]
                vetoed_symbols = entry_symbols[(entry_count - vetoed_count):]
        
        return {
            'entry_symbols': vetoed_entry,
            'vetoed_symbols': vetoed_symbols,
            'original_count': entry_count,
            'vetoed_count': len(vetoed_symbols),
            'regime': regime,
            'regime_confidence': regime_confidence,
            'veto_applied': len(vetoed_symbols) > 0
        }
    
    def resolve_position_conflicts(self, entry_signals: Dict[str, Any],
                                 risk_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts between entry signals and risk management.
        
        Args:
            entry_signals: Dictionary with entry signals
            risk_signals: Dictionary with risk management signals
            
        Returns:
            Dictionary with resolved position information
        """
        resolution_mode = self.resolution_modes.get('allocation', 'risk_priority')
        
        # Get entry symbols
        entry_symbols = entry_signals.get('entry_symbols', [])
        
        # Get allocation information if available
        allocations = risk_signals.get('allocations', {})
        position_sizes = risk_signals.get('position_sizes', [])
        
        # Convert position_sizes to dictionary if it's a list of records
        position_dict = {}
        if isinstance(position_sizes, list):
            for pos in position_sizes:
                if isinstance(pos, dict) and 'symbol' in pos:
                    position_dict[pos['symbol']] = pos
        
        # Build resolved positions
        positions = []
        for symbol in entry_symbols:
            position = {
                'symbol': symbol,
                'action': 'buy',
                'timestamp': datetime.now().isoformat()
            }
            
            # Add allocation if available
            if symbol in allocations:
                position['allocation'] = allocations[symbol]
            
            # Add position size details if available
            if symbol in position_dict:
                pos_details = position_dict[symbol]
                for key, value in pos_details.items():
                    if key != 'symbol':  # Avoid duplicate key
                        position[key] = value
            
            positions.append(position)
        
        return {
            'positions': positions,
            'count': len(positions),
            'resolution_mode': resolution_mode
        }


class SignalComposite:
    """
    Combines signals from multiple models using different strategies.
    """
    
    def __init__(self, signal_priority: SignalPriority, conflict_resolution: ConflictResolution):
        """
        Initialize signal composite system.
        
        Args:
            signal_priority: SignalPriority instance
            conflict_resolution: ConflictResolution instance
        """
        self.signal_priority = signal_priority
        self.conflict_resolution = conflict_resolution
    
    def compose_signals(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compose signals from multiple models into a coherent strategy.
        
        Args:
            model_results: Dictionary with results from multiple models
            
        Returns:
            Dictionary with composed strategy information
        """
        # Extract signals from model results
        market_regime = model_results.get('market_regime', {})
        stock_selection = model_results.get('stock_selection', {})
        entry_timing = model_results.get('entry_timing', {})
        peak_detection = model_results.get('peak_detection', {})
        risk_management = model_results.get('risk_management', {})
        
        # Step 1: Resolve entry and exit conflicts
        entry_exit_resolution = self.conflict_resolution.resolve_entry_exit_conflict(
            entry_timing, peak_detection, market_regime
        )
        
        # Step 2: Apply market regime veto
        vetoed_entries = self.conflict_resolution.apply_regime_veto(
            {'entry_symbols': entry_exit_resolution['entry_symbols']},
            market_regime
        )
        
        # Step 3: Resolve position sizing and allocation conflicts
        position_resolution = self.conflict_resolution.resolve_position_conflicts(
            {'entry_symbols': vetoed_entries['entry_symbols']},
            risk_management
        )
        
        # Step 4: Create unified decision structure
        unified_decision = {
            'timestamp': datetime.now().isoformat(),
            'positions': position_resolution['positions'],
            'exits': entry_exit_resolution['exit_symbols'],
            'market_regime': market_regime.get('regime', 'unknown'),
            'regime_confidence': market_regime.get('confidence', 0.5),
            'metrics': {
                'total_entries': len(entry_timing.get('entry_symbols', [])),
                'total_exits': len(peak_detection.get('exit_symbols', [])),
                'total_selections': len(stock_selection.get('selected_symbols', [])),
                'resolved_entries': len(vetoed_entries['entry_symbols']),
                'resolved_exits': len(entry_exit_resolution['exit_symbols']),
                'vetoed_count': vetoed_entries.get('vetoed_count', 0)
            },
            'conflicts': {
                'entry_exit_conflicts': entry_exit_resolution['conflicts'],
                'vetoed_symbols': vetoed_entries.get('vetoed_symbols', [])
            }
        }
        
        # Add portfolio risk if available
        if 'portfolio_risk' in risk_management:
            unified_decision['portfolio_risk'] = risk_management['portfolio_risk']
        
        return unified_decision


class DecisionFramework:
    """
    Reconciles signals from multiple models to produce final trading decisions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the decision framework.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.signal_priority = SignalPriority(self.config.get('signal_priority'))
        self.conflict_resolution = ConflictResolution(self.config.get('conflict_resolution'))
        self.signal_composite = SignalComposite(self.signal_priority, self.conflict_resolution)
        
        # Tracking for decisions
        self.last_decision = None
        self.decision_history = []
        
        logger.info("Decision Framework initialized")
    
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
    
    def reconcile_signals(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile signals from multiple models to produce final trading decisions.
        
        Args:
            model_results: Dictionary with results from multiple models
            
        Returns:
            Dictionary with reconciled trading decisions
        """
        logger.info("Reconciling signals from multiple models")
        
        try:
            # Validate inputs
            if not model_results:
                logger.warning("No model results provided")
                return {'error': 'No model results provided'}
            
            # Check if any models have errors
            errors = {}
            for model_name, result in model_results.items():
                if 'error' in result:
                    errors[model_name] = result['error']
            
            if errors:
                logger.warning(f"Errors in model results: {errors}")
                # Continue with available models
            
            # Compose signals
            decision = self.signal_composite.compose_signals(model_results)
            
            # Record decision
            self.last_decision = decision
            self.decision_history.append(decision)
            
            # Truncate history if too long
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]
            
            logger.info(f"Reconciled signals: {len(decision['positions'])} entries, {len(decision['exits'])} exits")
            
            return decision
        
        except Exception as e:
            logger.error(f"Error reconciling signals: {str(e)}")
            return {'error': f"Error reconciling signals: {str(e)}"}
    
    def validate_entry_signals(self, entry_signals: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate entry signals against market context.
        
        Args:
            entry_signals: Dictionary with entry signals
            market_context: Dictionary with market context
            
        Returns:
            Dictionary with validated entry signals
        """
        logger.info("Validating entry signals against market context")
        
        try:
            # Get market regime
            market_regime = market_context.get('market_regime', {})
            
            # Apply regime veto
            vetoed_entries = self.conflict_resolution.apply_regime_veto(
                entry_signals,
                market_regime
            )
            
            return vetoed_entries
        
        except Exception as e:
            logger.error(f"Error validating entry signals: {str(e)}")
            return {'error': f"Error validating entry signals: {str(e)}"}
    
    def get_last_decision(self) -> Dict[str, Any]:
        """
        Get the last reconciled decision.
        
        Returns:
            Dictionary with last decision
        """
        if self.last_decision is None:
            return {'error': 'No decisions made yet'}
        
        return self.last_decision
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get decision history.
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of historical decisions
        """
        return self.decision_history[-limit:]


# Default decision framework instance
default_decision_framework = None


def get_decision_framework(config_path: Optional[str] = None) -> DecisionFramework:
    """
    Get or create the default decision framework.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        DecisionFramework instance
    """
    global default_decision_framework
    
    if default_decision_framework is None:
        default_decision_framework = DecisionFramework(config_path)
    
    return default_decision_framework


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Decision Framework for Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--results', type=str, help='Path to model results JSON file')
    
    args = parser.parse_args()
    
    # Create decision framework
    decision_framework = DecisionFramework(args.config)
    
    if args.results:
        # Load model results from file
        with open(args.results, 'r') as f:
            model_results = json.load(f)
        
        # Reconcile signals
        decision = decision_framework.reconcile_signals(model_results)
        
        # Print decision
        print(json.dumps(decision, indent=2))
    else:
        print("No model results provided. Please specify --results.")
