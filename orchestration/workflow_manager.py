#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Workflow Manager
---------------
Coordinates the execution of multiple models in the trading system.
Manages the flow of data and signals between models.
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import concurrent.futures
from functools import lru_cache

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from data.processors.data_cache import DataCache
from data.processors.feature_engineer import FeatureEngineer
from orchestration.decision_framework import DecisionFramework
from orchestration.adaptive_thresholds import AdaptiveThresholds
# Setup logging
logger = setup_logger('workflow_manager')


class ModelConfig:
    """
    Configuration for a model in the workflow.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the model configuration.
        
        Args:
            name: Model name
            config: Model configuration dictionary
        """
        self.name = name
        self.enabled = config.get('enabled', True)
        self.model_path = config.get('model_path', f'models/{name}/latest')
        self.onnx_path = config.get('onnx_path', f'models/{name}/latest.onnx')
        self.use_onnx = config.get('use_onnx', True)
        self.batch_size = config.get('batch_size', 64)
        self.feature_sets = config.get('feature_sets', ['price', 'volume', 'technical'])
        self.dependencies = config.get('dependencies', [])
        self.parallel = config.get('parallel', False)
        self.timeout = config.get('timeout', 30)  # seconds
        self.throttle = config.get('throttle', 0)  # seconds between executions
        self.cache_ttl = config.get('cache_ttl', 300)  # seconds to cache results
        self.thresholds = config.get('thresholds', {})
        self.fallback_strategy = config.get('fallback_strategy', 'skip')
        self.metadata = config.get('metadata', {})
        
        # Additional initialization based on model type
        if name == 'entry_timing':
            self.sequence_length = config.get('sequence_length', 30)
            self.input_shape = config.get('input_shape', [None, self.sequence_length, None])


class WorkflowManager:
    """
    Manages the execution workflow for trading models.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the workflow manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model_configs = self._init_model_configs()
        self.model_instances = {}
        self.feature_engineer = FeatureEngineer()
        self.decision_framework = DecisionFramework()
        self.adaptive_thresholds = AdaptiveThresholds()
        self.data_cache = DataCache()
        # Access the monitor manager to get the model metrics exporter
        from monitoring.monitor_manager import get_monitor_manager
        self.monitor_manager = get_monitor_manager()
        self.model_metrics = self.monitor_manager.exporters.get('model')
        if not self.model_metrics:
            logger.warning("Model metrics exporter not available in MonitorManager.")
        
        # Track the last execution time of each model
        self.last_execution = {}
        
        # Initialize models
        self._init_models()
        
        logger.info(f"Workflow Manager initialized with {len(self.model_configs)} models")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config dictionary
        """
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _init_model_configs(self) -> Dict[str, ModelConfig]:
        """
        Initialize model configurations.
        
        Returns:
            Dictionary of model configurations
        """
        model_configs = {}
        
        # Get model configurations
        models_config = self.config.get('models', {})
        
        # Create model configurations
        for model_name, model_config in models_config.items():
            model_configs[model_name] = ModelConfig(model_name, model_config)
        
        return model_configs
    
    def _init_models(self) -> None:
        """
        Initialize model instances.
        """
        for name, config in self.model_configs.items():
            if not config.enabled:
                logger.info(f"Model {name} is disabled, skipping initialization")
                continue
            
            try:
                # Initialize the model based on its type
                if name == 'market_regime':
                    from models.market_regime.model import MarketRegimeModel
                    self.model_instances[name] = MarketRegimeModel(config.model_path, use_onnx=config.use_onnx)
                
                elif name == 'stock_selection':
                    from models.stock_selection.model import StockSelectionModel
                    self.model_instances[name] = StockSelectionModel(config.model_path, use_onnx=config.use_onnx)
                
                elif name == 'entry_timing':
                    from models.entry_timing.model import EntryTimingModel
                    self.model_instances[name] = EntryTimingModel(config.model_path, use_onnx=config.use_onnx)
                
                elif name == 'peak_detection':
                    from models.peak_detection.model import PeakDetectionModel
                    self.model_instances[name] = PeakDetectionModel(config.model_path, use_onnx=config.use_onnx)
                
                elif name == 'risk_management':
                    from models.risk_management.model import RiskModel
                    self.model_instances[name] = RiskModel(config.model_path, use_onnx=config.use_onnx)
                
                else:
                    logger.warning(f"Unknown model type: {name}")
                
                logger.info(f"Initialized model: {name}")
            
            except Exception as e:
                logger.error(f"Error initializing model {name}: {str(e)}")
    
    def _check_dependencies(self, model_name: str, results: Dict[str, Any]) -> bool:
        """
        Check if all dependencies for a model are satisfied.
        
        Args:
            model_name: Name of the model
            results: Dictionary of results from other models
            
        Returns:
            Boolean indicating if dependencies are satisfied
        """
        # Get model configuration
        config = self.model_configs.get(model_name)
        if not config:
            logger.error(f"Model {model_name} not configured")
            return False
        
        # Check dependencies
        for dependency in config.dependencies:
            if dependency not in results:
                logger.warning(f"Dependency {dependency} not satisfied for {model_name}")
                return False
        
        return True
    
    def _get_model_features(self, model_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get features for a specific model.
        
        Args:
            model_name: Name of the model
            data: DataFrame with raw data
            
        Returns:
            DataFrame with features
        """
        # Get model configuration
        config = self.model_configs.get(model_name)
        if not config:
            logger.error(f"Model {model_name} not configured")
            return data
        
        # Generate features
        features = self.feature_engineer.create_features(data, config.feature_sets)
        
        return features
    
    def _prepare_model_input(self, model_name: str, data: pd.DataFrame, 
                          previous_results: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Prepare input for a specific model.
        
        Args:
            model_name: Name of the model
            data: DataFrame with features
            previous_results: Results from other models
            
        Returns:
            Tuple of (model input, metadata)
        """
        # Get model configuration
        config = self.model_configs.get(model_name)
        if not config:
            logger.error(f"Model {model_name} not configured")
            return None, {}
        
        # Get model instance
        model = self.model_instances.get(model_name)
        if not model:
            logger.error(f"Model {model_name} not initialized")
            return None, {}
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'symbols': data['symbol'].unique().tolist(),
            'model_name': model_name
        }
        
        try:
            # Special handling based on model type
            if model_name == 'market_regime':
                # Market regime model typically works with aggregated market data
                market_data = data.copy()
                # Further processing specific to market regime model...
                model_input = model.prepare_input(market_data)
                metadata['regime_features'] = config.feature_sets
            
            elif model_name == 'stock_selection':
                # Stock selection model typically works with latest snapshot data
                latest_data = data.groupby('symbol').last().reset_index()
                # Include market regime in input if available
                if 'market_regime' in previous_results:
                    latest_data['market_regime'] = previous_results['market_regime'].get('regime', 'unknown')
                model_input = model.prepare_input(latest_data)
                metadata['selection_candidates'] = len(latest_data)
            
            elif model_name == 'entry_timing':
                # Entry timing model typically works with sequences of data
                # Filter symbols if stock selection model has run
                if 'stock_selection' in previous_results:
                    selected_symbols = previous_results['stock_selection'].get('selected_symbols', [])
                    if selected_symbols:
                        data = data[data['symbol'].isin(selected_symbols)]
                
                # Prepare sequences
                model_input = model.prepare_input(data, config.sequence_length)
                metadata['sequence_length'] = config.sequence_length
                metadata['symbols_count'] = len(data['symbol'].unique())
            
            elif model_name == 'peak_detection':
                # Peak detection model typically works with sequences of data
                # Only consider symbols with entry signals
                if 'entry_timing' in previous_results:
                    entry_symbols = previous_results['entry_timing'].get('entry_symbols', [])
                    if entry_symbols:
                        data = data[data['symbol'].isin(entry_symbols)]
                
                # Prepare sequences
                model_input = model.prepare_input(data, lookback=config.metadata.get('lookback', 20))
                metadata['lookback'] = config.metadata.get('lookback', 20)
            
            elif model_name == 'risk_management':
                # Risk model typically works with portfolio and entry signals
                # Combine entry signals with portfolio data
                entry_signals = previous_results.get('entry_timing', {}).get('signals', pd.DataFrame())
                
                # TODO: Get portfolio data from portfolio module
                portfolio_data = pd.DataFrame()  # placeholder
                
                model_input = model.prepare_input(entry_signals, portfolio_data)
                metadata['signals_count'] = len(entry_signals) if isinstance(entry_signals, pd.DataFrame) else 0
            
            else:
                # Generic approach for unknown models
                model_input = data
            
            return model_input, metadata
        
        except Exception as e:
            logger.error(f"Error preparing input for {model_name}: {str(e)}")
            return None, metadata
    
    def _execute_model(self, model_name: str, model_input: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific model.
        
        Args:
            model_name: Name of the model
            model_input: Input for the model
            metadata: Metadata for the execution
            
        Returns:
            Dictionary with model results
        """
        # Get model configuration
        config = self.model_configs.get(model_name)
        if not config:
            logger.error(f"Model {model_name} not configured")
            return {'error': f"Model {model_name} not configured"}
        
        # Get model instance
        model = self.model_instances.get(model_name)
        if not model:
            logger.error(f"Model {model_name} not initialized")
            return {'error': f"Model {model_name} not initialized"}
        
        # Check throttling
        current_time = time.time()
        last_time = self.last_execution.get(model_name, 0)
        if current_time - last_time < config.throttle:
            wait_time = config.throttle - (current_time - last_time)
            logger.debug(f"Throttling {model_name} for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Update last execution time
        self.last_execution[model_name] = time.time()
        
        try:
            # Start timing
            start_time = time.time()
            
            # Execute the model with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(model.predict, model_input)
                try:
                    result = future.result(timeout=config.timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout executing {model_name}")
                    return {'error': f"Timeout executing {model_name}"}
            
            # End timing
            execution_time = time.time() - start_time
            
            # Post-process results
            processed_result = self._post_process_results(model_name, result, metadata)
            
            # Store in cache if enabled
            if config.cache_ttl > 0:
                cache_key = f"{model_name}_{metadata['timestamp']}"
                self.data_cache.set(cache_key, processed_result, config.cache_ttl)
            
            # Record metrics if exporter is available
            if self.model_metrics:
                self.model_metrics.record_execution(
                    model_name,
                    execution_time,
                    success=True,
                    metadata=metadata
                )
            
            logger.info(f"Executed {model_name} in {execution_time:.2f} seconds")
            return processed_result
        
        except Exception as e:
            logger.error(f"Error executing {model_name}: {str(e)}")
            
            # Record error metrics if exporter is available
            if self.model_metrics:
                self.model_metrics.record_execution(
                    model_name,
                    0.0,
                    success=False,
                    error=str(e),
                    metadata=metadata
                )
            
            # Return error
            return {'error': str(e)}
    
    def _post_process_results(self, model_name: str, result: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process model results.
        
        Args:
            model_name: Name of the model
            result: Raw model result
            metadata: Metadata for the execution
            
        Returns:
            Dictionary with processed results
        """
        # Get model configuration
        config = self.model_configs.get(model_name)
        if not config:
            return {'error': f"Model {model_name} not configured", 'raw_result': result}
        
        # Basic structure for result
        processed_result = {
            'model_name': model_name,
            'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
            'execution_id': metadata.get('execution_id', str(int(time.time()))),
            'metadata': metadata
        }
        
        try:
            # Special handling based on model type
            if model_name == 'market_regime':
                # Process market regime results
                if isinstance(result, dict) and 'regime' in result:
                    processed_result.update(result)
                else:
                    # Extract regime from model output
                    if hasattr(result, 'iloc'):  # DataFrame-like object
                        regime = result.iloc[0]['regime'] if 'regime' in result.columns else 'unknown'
                    elif isinstance(result, (list, np.ndarray)) and len(result) > 0:
                        # Convert numeric regime to string label
                        regime_map = {0: 'bear', 1: 'neutral', 2: 'bull'}
                        regime = regime_map.get(int(result[0]), 'unknown')
                    else:
                        regime = 'unknown'
                    
                    processed_result['regime'] = regime
                    processed_result['confidence'] = float(result[0][1]) if isinstance(result, list) and len(result) > 0 and len(result[0]) > 1 else 0.5
                
                # Apply adaptive thresholds
                processed_result = self.adaptive_thresholds.adjust_results(processed_result, model_name)
            
            elif model_name == 'stock_selection':
                # Process stock selection results
                if isinstance(result, pd.DataFrame):
                    # Extract selected symbols (assuming result contains 'symbol' and 'selected' columns)
                    if 'symbol' in result.columns and 'selected' in result.columns:
                        selected_df = result[result['selected'] == 1]
                        selected_symbols = selected_df['symbol'].tolist()
                    else:
                        # Assume the DataFrame itself contains selected symbols
                        selected_symbols = result['symbol'].tolist() if 'symbol' in result.columns else []
                    
                    processed_result['selected_symbols'] = selected_symbols
                    processed_result['selected_count'] = len(selected_symbols)
                    processed_result['selection_data'] = result.to_dict('records') if len(result) < 100 else f"{len(result)} records (too large to include)"
                
                elif isinstance(result, list):
                    # Assume the list contains selected symbols
                    processed_result['selected_symbols'] = result
                    processed_result['selected_count'] = len(result)
                
                # Apply adaptive thresholds
                processed_result = self.adaptive_thresholds.adjust_results(processed_result, model_name)
            
            elif model_name == 'entry_timing':
                # Process entry timing results
                if isinstance(result, pd.DataFrame):
                    # Extract entry signals (assuming result contains 'symbol', 'timestamp', and 'entry_signal' columns)
                    entry_df = result[result['entry_signal'] == 1] if 'entry_signal' in result.columns else pd.DataFrame()
                    
                    processed_result['entry_symbols'] = entry_df['symbol'].unique().tolist() if 'symbol' in entry_df.columns else []
                    processed_result['entry_count'] = len(processed_result['entry_symbols'])
                    processed_result['signals'] = entry_df.to_dict('records') if len(entry_df) < 100 else f"{len(entry_df)} signals (too large to include)"
                    
                    # Group signals by symbol for convenience
                    if 'symbol' in entry_df.columns:
                        signals_by_symbol = {}
                        for symbol, group in entry_df.groupby('symbol'):
                            signals_by_symbol[symbol] = group.to_dict('records')
                        processed_result['signals_by_symbol'] = signals_by_symbol
                
                # Apply adaptive thresholds
                processed_result = self.adaptive_thresholds.adjust_results(processed_result, model_name)
            
            elif model_name == 'peak_detection':
                # Process peak detection results
                if isinstance(result, pd.DataFrame):
                    # Extract exit signals
                    exit_df = result[result['exit_signal'] == 1] if 'exit_signal' in result.columns else pd.DataFrame()
                    
                    processed_result['exit_symbols'] = exit_df['symbol'].unique().tolist() if 'symbol' in exit_df.columns else []
                    processed_result['exit_count'] = len(processed_result['exit_symbols'])
                    processed_result['signals'] = exit_df.to_dict('records') if len(exit_df) < 100 else f"{len(exit_df)} signals (too large to include)"
                
                # Apply adaptive thresholds
                processed_result = self.adaptive_thresholds.adjust_results(processed_result, model_name)
            
            elif model_name == 'risk_management':
                # Process risk management results
                if isinstance(result, pd.DataFrame):
                    # Extract position sizing recommendations
                    processed_result['position_sizes'] = result.to_dict('records') if len(result) < 100 else f"{len(result)} positions (too large to include)"
                    
                    # Extract portfolio metrics
                    if 'portfolio_risk' in result.columns and len(result) > 0:
                        processed_result['portfolio_risk'] = float(result['portfolio_risk'].iloc[0])
                    
                    # Extract allocation recommendations
                    if 'allocation' in result.columns:
                        allocations = {}
                        for _, row in result.iterrows():
                            allocations[row['symbol']] = float(row['allocation'])
                        processed_result['allocations'] = allocations
                
                # Apply adaptive thresholds
                processed_result = self.adaptive_thresholds.adjust_results(processed_result, model_name)
            
            else:
                # Generic result processing
                processed_result['result'] = result
            
            return processed_result
        
        except Exception as e:
            logger.error(f"Error post-processing results for {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                'execution_id': metadata.get('execution_id', str(int(time.time()))),
                'error': f"Error post-processing results: {str(e)}",
                'raw_result': str(result)[:1000] if isinstance(result, str) else 'non-string result',
                'metadata': metadata
            }
    
    def execute_workflow(self, data: pd.DataFrame, selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the complete model workflow.
        
        Args:
            data: DataFrame with raw data
            selected_models: Optional list of models to execute (if None, all enabled models are executed)
            
        Returns:
            Dictionary with results from all models
        """
        # Generate a unique execution ID
        execution_id = f"exec_{int(time.time())}"
        logger.info(f"Starting workflow execution: {execution_id}")
        
        # Determine which models to execute
        if selected_models is None:
            # Use all enabled models
            models_to_execute = [name for name, config in self.model_configs.items() if config.enabled]
        else:
            # Use selected models if they are configured and enabled
            models_to_execute = []
            for name in selected_models:
                if name in self.model_configs:
                    if self.model_configs[name].enabled:
                        models_to_execute.append(name)
                    else:
                        logger.warning(f"Skipping disabled model: {name}")
                else:
                    logger.warning(f"Unknown model: {name}")
        
        # Sort models by dependencies to determine execution order
        execution_order = self._determine_execution_order(models_to_execute)
        logger.info(f"Execution order: {execution_order}")
        
        # Results dictionary
        results = {}
        
        # Execute models in order
        for model_name in execution_order:
            logger.info(f"Executing model: {model_name}")
            
            # Check dependencies
            if not self._check_dependencies(model_name, results):
                logger.warning(f"Dependencies not met for {model_name}, skipping")
                continue
            
            # Get model configuration
            config = self.model_configs.get(model_name)
            
            # Get features for the model
            features = self._get_model_features(model_name, data)
            
            # Prepare model input
            model_input, metadata = self._prepare_model_input(model_name, features, results)
            metadata['execution_id'] = execution_id
            
            if model_input is None:
                logger.error(f"Failed to prepare input for {model_name}")
                results[model_name] = {'error': f"Failed to prepare input for {model_name}"}
                continue
            
            # Execute the model
            model_result = self._execute_model(model_name, model_input, metadata)
            
            # Store result
            results[model_name] = model_result
            
            # Update data based on model output if needed
            if model_name == 'stock_selection' and 'selected_symbols' in model_result:
                # Filter data to selected symbols for subsequent models
                selected_symbols = model_result['selected_symbols']
                if selected_symbols:
                    data = data[data['symbol'].isin(selected_symbols)]
        
        # Combine results with decision framework
        combined_result = self.decision_framework.reconcile_signals(results)
        results['combined'] = combined_result
        
        logger.info(f"Completed workflow execution: {execution_id}")
        return results
    
    def _determine_execution_order(self, models: List[str]) -> List[str]:
        """
        Determine the order of model execution based on dependencies.
        
        Args:
            models: List of model names to execute
            
        Returns:
            List of model names in execution order
        """
        # Create a dependency graph
        dependency_graph = {}
        for model in models:
            config = self.model_configs.get(model)
            if config:
                # Only include dependencies that are in the models list
                dependencies = [dep for dep in config.dependencies if dep in models]
                dependency_graph[model] = dependencies
        
        # Topological sort to determine execution order
        execution_order = []
        permanent_marks = set()
        temporary_marks = set()
        
        def visit(node):
            if node in permanent_marks:
                return
            if node in temporary_marks:
                raise ValueError(f"Cyclic dependency detected involving {node}")
            
            temporary_marks.add(node)
            
            for dependency in dependency_graph.get(node, []):
                visit(dependency)
            
            temporary_marks.remove(node)
            permanent_marks.add(node)
            execution_order.append(node)
        
        # Visit all nodes
        for model in models:
            if model not in permanent_marks:
                visit(model)
        
        # Reverse the order (since topological sort gives reverse dependency order)
        execution_order.reverse()
        
        return execution_order
    
    def execute_entry_timing(self, symbols: List[str], data: Optional[pd.DataFrame] = None,
                          market_regime: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute only the entry timing model for specific symbols.
        
        Args:
            symbols: List of symbols to analyze
            data: Optional DataFrame with data for the symbols
            market_regime: Optional market regime override
            
        Returns:
            Dictionary with entry timing results
        """
        logger.info(f"Executing entry timing model for {len(symbols)} symbols")
        
        # Validate the entry timing model is available
        if 'entry_timing' not in self.model_instances:
            logger.error("Entry timing model not initialized")
            return {'error': "Entry timing model not initialized"}
        
        # Get model configuration
        config = self.model_configs.get('entry_timing')
        if not config:
            logger.error("Entry timing model not configured")
            return {'error': "Entry timing model not configured"}
        
        # Generate execution ID
        execution_id = f"entry_{int(time.time())}"
        
        # Fetch data if not provided
        if data is None:
            # Determine date range (e.g., last 45 days for a 30-day sequence)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.sequence_length * 1.5)
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data using feature engineer
            logger.info(f"Fetching data for {len(symbols)} symbols from {start_str} to {end_str}")
            try:
                data = self.feature_engineer.create_features_for_symbols(
                    symbols, start_str, end_str, resolution='1d', feature_sets=config.feature_sets
                )
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                return {'error': f"Error fetching data: {str(e)}"}
        
        # Get features for the model
        features = self._get_model_features('entry_timing', data)
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'execution_id': execution_id,
            'symbols': symbols,
            'model_name': 'entry_timing',
            'sequence_length': config.sequence_length,
            'symbols_count': len(symbols),
            'market_regime': market_regime
        }
        
        # Prepare model input
        model_input, _ = self._prepare_model_input('entry_timing', features, 
                                                {'market_regime': {'regime': market_regime}} if market_regime else {})
        
        if model_input is None:
            logger.error("Failed to prepare input for entry timing model")
            return {'error': "Failed to prepare input for entry timing model"}
        
        # Execute the model
        result = self._execute_model('entry_timing', model_input, metadata)
        
        return result
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all configured models.
        
        Returns:
            Dictionary with model status information
        """
        status = {}
        
        for name, config in self.model_configs.items():
            model_status = {
                'enabled': config.enabled,
                'initialized': name in self.model_instances,
                'last_execution': datetime.fromtimestamp(self.last_execution.get(name, 0)).isoformat() if name in self.last_execution else None,
                'configuration': {
                    'model_path': config.model_path,
                    'use_onnx': config.use_onnx,
                    'dependencies': config.dependencies,
                    'feature_sets': config.feature_sets
                }
            }
            
            # Add model-specific information
            if name == 'entry_timing':
                model_status['configuration']['sequence_length'] = config.sequence_length
            
            # Get metrics if exporter is available
            if self.model_metrics:
                try:
                    model_status['metrics'] = self.model_metrics.get_model_metrics(name)
                except Exception as e:
                    logger.warning(f"Error getting metrics for {name}: {str(e)}")
            
            status[name] = model_status
        
        return status
    
    def reload_model(self, model_name: str) -> bool:
        """
        Reload a specific model.
        
        Args:
            model_name: Name of the model to reload
            
        Returns:
            Boolean indicating success
        """
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        
        try:
            # Unload current model if exists
            if model_name in self.model_instances:
                del self.model_instances[model_name]
            
            # Initialize the model based on its type
            if model_name == 'market_regime':
                from models.market_regime.model import MarketRegimeModel
                self.model_instances[model_name] = MarketRegimeModel(config.model_path, use_onnx=config.use_onnx)
            
            elif model_name == 'stock_selection':
                from models.stock_selection.model import StockSelectionModel
                self.model_instances[model_name] = StockSelectionModel(config.model_path, use_onnx=config.use_onnx)
            
            elif model_name == 'entry_timing':
                from models.entry_timing.model import EntryTimingModel
                self.model_instances[model_name] = EntryTimingModel(config.model_path, use_onnx=config.use_onnx)
            
            elif model_name == 'peak_detection':
                from models.peak_detection.model import PeakDetectionModel
                self.model_instances[model_name] = PeakDetectionModel(config.model_path, use_onnx=config.use_onnx)
            
            elif model_name == 'risk_management':
                from models.risk_management.model import RiskModel
                self.model_instances[model_name] = RiskModel(config.model_path, use_onnx=config.use_onnx)
            
            else:
                logger.warning(f"Unknown model type: {model_name}")
                return False
            
            logger.info(f"Reloaded model: {model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error reloading model {model_name}: {str(e)}")
            return False
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific model.
        
        Args:
            model_name: Name of the model
            updates: Dictionary with configuration updates
            
        Returns:
            Boolean indicating success
        """
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        try:
            # Get current configuration
            config = self.model_configs[model_name]
            
            # Update configuration
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
            
            logger.info(f"Updated configuration for model: {model_name}")
            
            # Reload the model if necessary
            if updates.get('model_path') or updates.get('use_onnx'):
                logger.info(f"Reloading model {model_name} due to configuration changes")
                return self.reload_model(model_name)
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating configuration for {model_name}: {str(e)}")
            return False


# Default workflow manager instance
default_workflow_manager = None


def get_workflow_manager(config_path: str = 'config/workflow.json') -> WorkflowManager:
    """
    Get or create the default workflow manager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        WorkflowManager instance
    """
    global default_workflow_manager
    
    if default_workflow_manager is None:
        default_workflow_manager = WorkflowManager(config_path)
    
    return default_workflow_manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Workflow Manager for Trading System')
    parser.add_argument('--config', type=str, default='config/workflow.json', help='Path to configuration file')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to process')
    parser.add_argument('--model', type=str, help='Specific model to execute')
    parser.add_argument('--status', action='store_true', help='Get status of all models')
    
    args = parser.parse_args()
    
    # Create workflow manager
    workflow_manager = WorkflowManager(args.config)
    
    if args.status:
        # Get and print model status
        status = workflow_manager.get_model_status()
        import json
        print(json.dumps(status, indent=2))
    
    elif args.model and args.symbols:
        # Execute specific model for symbols
        if args.model == 'entry_timing':
            result = workflow_manager.execute_entry_timing(args.symbols)
            import json
            print(json.dumps(result, indent=2))
        else:
            print(f"Direct execution not supported for model: {args.model}")
    
    else:
        print("Please specify --status or both --model and --symbols")
