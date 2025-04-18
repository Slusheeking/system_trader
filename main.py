#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
System Trader Main Application
------------------------------
Entry point for the autonomous day trading system.
Initializes all components and starts the trading system.
"""

import os
import sys
import argparse
import logging
import signal
import time
from datetime import datetime
import threading

# Import system components
from utils.logging import setup_logger
from utils.config_loader import load_config
from monitoring.monitor_manager import get_monitor_manager
from monitoring import register_component, record_heartbeat
from data.collectors.factory import CollectorFactory
from data.database.timeseries_db import get_timescale_client, TimeseriesDBClient as TimeSeriesDB
from trading.execution import get_circuit_breaker, get_execution_monitor, OrderRouter
from trading import TradingManager
from orchestration.workflow_manager import get_workflow_manager
from orchestration import AdaptiveThresholds, ErrorHandler
from portfolio import PositionTracker, PerformanceMonitor, RiskCalculator as RiskManager
from portfolio import DiversificationEngine
from models.optimization.onnx_converter import ONNXConverter
from mlflow.tracking import MLflowTracker
# Import our custom ModelRegistry implementation
from mlflow.registry import ModelRegistry
from monitoring.prometheus.exporters.model_metrics_exporter import ModelMetricsExporter
from monitoring.prometheus.exporters.system_metrics_exporter import SystemMetricsExporter
from monitoring.prometheus.exporters.trading_metrics_exporter import TradingMetricsExporter

# Setup root logger
logger = setup_logger('system_trader')


class SystemTrader:
    """
    Main application class for the autonomous day trading system.
    """
    
    def __init__(self, config_path: str = 'config/system_config.yaml'):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to system configuration file
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()
        
        # Initialize monitoring first
        self._init_monitoring()
        
        # Register with monitoring
        register_component('system_trader', self)
        
        # Initialize components
        self._init_components()
        
        logger.info("System Trader initialized")
    
    def _init_monitoring(self):
        """
        Initialize monitoring system.
        """
        try:
            # Get monitor manager (initializes monitoring)
            monitor_manager = get_monitor_manager(self.config_path)
            self.components['monitor_manager'] = monitor_manager
            
            logger.info("Monitoring system initialized")
        
        except Exception as e:
            logger.error(f"Error initializing monitoring: {str(e)}")
            raise
    
    def _init_components(self):
        """
        Initialize system components.
        """
        try:
            # Initialize database
            db_config = self.config.get('database', {})
            # Create a config dictionary for TimeseriesDB
            timeseries_config = {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 5432),
                'dbname': db_config.get('name', 'trading_db'),
                'user': db_config.get('user', 'trading_user'),
                'password': os.environ.get(db_config.get('password_env', 'DB_PASSWORD'), '')
            }
            timeseries_db = get_timescale_client(timeseries_config)
            self.components['timeseries_db'] = timeseries_db
            register_component('timeseries_db', timeseries_db)
            
            # Initialize data collectors
            from config.collector_config import CollectorConfig, get_collector_config
            collectors = {}

            # Initialize primary data collector based on system_config.yaml
            data_client_config = self.config.get('data_client_config', {})
            primary_collector_name = data_client_config.get('type')

            if primary_collector_name:
                try:
                    # Use get_collector_config to correctly load credentials and other settings
                    config = get_collector_config(primary_collector_name)
                    collector = CollectorFactory.create(primary_collector_name, config)
                    collectors[primary_collector_name] = collector
                    register_component(f'collector_{primary_collector_name}', collector)
                    logger.info(f"Initialized primary data collector: {primary_collector_name}")
                except Exception as e:
                    logger.error(f"Error initializing primary data collector {primary_collector_name}: {str(e)}")

            # Initialize other specific collectors as needed (e.g., Reddit for sentiment)
            # Based on initial logs, Reddit collector was attempted and seemed to initialize
            try:
                reddit_config = CollectorConfig.load('reddit')
                reddit_collector = CollectorFactory.create('reddit', reddit_config)
                collectors['reddit'] = reddit_collector
                register_component('collector_reddit', reddit_collector)
                logger.info("Initialized Reddit collector")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit collector: {str(e)}")

            self.components['collectors'] = collectors

            # Initialize circuit breaker
            circuit_breaker = get_circuit_breaker('config/circuit_breaker.json') # Pass JSON path
            self.components['circuit_breaker'] = circuit_breaker
            
            # Initialize execution monitor
            execution_monitor = get_execution_monitor()
            self.components['execution_monitor'] = execution_monitor
            register_component('execution_monitor', execution_monitor)
            
            # Initialize order router
            order_router = OrderRouter('config/order_router.json') # Pass JSON path
            self.components['order_router'] = order_router
            register_component('order_router', order_router)
            
            # Initialize position tracker
            position_tracker = PositionTracker()
            self.components['position_tracker'] = position_tracker
            register_component('position_tracker', position_tracker)
            
            # Initialize performance monitor
            performance_monitor = PerformanceMonitor()
            self.components['performance_monitor'] = performance_monitor
            register_component('performance_monitor', performance_monitor)
            
            # Initialize risk management components
            risk_manager = RiskManager(config_path='config/risk_management.json') # Pass JSON path
            self.components['risk_manager'] = risk_manager
            register_component('risk_manager', risk_manager)
            
            # Initialize diversification engine
            diversification_engine = DiversificationEngine(risk_manager, self.config_path)
            self.components['diversification_engine'] = diversification_engine
            register_component('diversification_engine', diversification_engine)
            
            # Initialize MLflow components
            mlflow_config = self.config.get('mlflow', {})
            mlflow_tracker = MLflowTracker(
                tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
                registry_uri=mlflow_config.get('registry_uri', 'http://localhost:5000'),
                experiment_prefix=mlflow_config.get('experiment_prefix', 'trading_')
            )
            self.components['mlflow_tracker'] = mlflow_tracker
            register_component('mlflow_tracker', mlflow_tracker)
            
            # Initialize model registry with tracking URI
            model_registry = ModelRegistry(
                tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000')
            )
            self.components['model_registry'] = model_registry
            register_component('model_registry', model_registry)
            
            # Initialize ONNX converter
            onnx_config = self.config.get('optimization', {})
            onnx_converter = ONNXConverter(
                target_hardware=onnx_config.get('target_hardware', 'gh200')
            )
            self.components['onnx_converter'] = onnx_converter
            register_component('onnx_converter', onnx_converter)
            
            # Initialize orchestration components
            workflow_manager = get_workflow_manager('config/workflow.json') # Pass JSON path
            self.components['workflow_manager'] = workflow_manager
            register_component('workflow_manager', workflow_manager)
            
            # Initialize adaptive thresholds
            adaptive_thresholds = AdaptiveThresholds()
            self.components['adaptive_thresholds'] = adaptive_thresholds
            register_component('adaptive_thresholds', adaptive_thresholds)
            
            # Initialize error handler
            error_handler = ErrorHandler()
            self.components['error_handler'] = error_handler
            register_component('error_handler', error_handler)
            
            # Initialize data cache
            from data.processors.data_cache import DataCache
            # Extract cache settings from the main config (using redis ttl as default)
            cache_namespace = self.config.get('cache', {}).get('namespace', 'data_cache')
            default_ttl = self.config.get('redis', {}).get('ttl', {}).get('market_data', 3600) # Default 1 hour
            data_cache = DataCache(namespace=cache_namespace, ttl=default_ttl)
            self.components['data_cache'] = data_cache
            register_component('data_cache', data_cache)

            # Get model metrics exporter from monitor manager
            model_metrics_exporter = self.components['monitor_manager'].exporters.get('model')
            if not model_metrics_exporter:
                 logger.warning("Model metrics exporter not available in MonitorManager.")

            # Initialize trading manager
            trading_manager = TradingManager(
                circuit_breaker=circuit_breaker,
                execution_monitor=execution_monitor,
                order_router=order_router,
                position_tracker=position_tracker,
                performance_monitor=performance_monitor,
                risk_manager=risk_manager,
                diversification_engine=diversification_engine,
                error_handler=error_handler,
                workflow_manager=workflow_manager,
                data_cache=data_cache,
                model_metrics=model_metrics_exporter,
                config_path=self.config_path # Pass config_path to TradingManager
            )
            self.components['trading_manager'] = trading_manager
            register_component('trading_manager', trading_manager)
            
            logger.info("All components initialized")
        
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def start(self):
        """
        Start the trading system.
        """
        if self.running:
            logger.warning("Trading system already running")
            return
        
        try:
            logger.info("Starting trading system")
            self.running = True
            
            # Start heartbeat thread
            self._start_heartbeat_thread()
            
            # Start components
            for name, component in self.components.items():
                if hasattr(component, 'start') and callable(component.start):
                    component.start()
                    logger.info(f"Started component: {name}")
            
            logger.info("Trading system started")
            
            # Wait for shutdown signal
            while not self.shutdown_event.is_set():
                time.sleep(1)
            
            logger.info("Shutdown signal received")
            
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            self.shutdown()
    
    def shutdown(self):
        """
        Shutdown the trading system.
        """
        if not self.running:
            return
        
        logger.info("Shutting down trading system")
        self.running = False
        self.shutdown_event.set()
        
        # Shutdown components in reverse order
        component_names = list(self.components.keys())
        component_names.reverse()
        
        for name in component_names:
            component = self.components.get(name)
            if component and hasattr(component, 'shutdown') and callable(component.shutdown):
                try:
                    component.shutdown()
                    logger.info(f"Shutdown component: {name}")
                except Exception as e:
                    logger.error(f"Error shutting down component {name}: {str(e)}")
        
        logger.info("Trading system shutdown complete")
    
    def _start_heartbeat_thread(self):
        """
        Start heartbeat thread.
        """
        def heartbeat_loop():
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Send heartbeat
                    record_heartbeat('system_trader')
                except Exception as e:
                    logger.error(f"Error in heartbeat: {str(e)}")
                
                # Sleep for 30 seconds
                time.sleep(30)
        
        # Start thread
        heartbeat_thread = threading.Thread(target=heartbeat_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        logger.info("Heartbeat thread started")


def signal_handler(sig, frame):
    """
    Handle signals.
    """
    logger.info(f"Received signal {sig}")
    if trader:
        trader.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autonomous Day Trading System')
    parser.add_argument('--config', type=str, default='config/system_config.yaml', help='Path to configuration file')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start trader
    trader = SystemTrader(args.config)
    
    if args.test:
        logger.info("Running in test mode")
        # Set test mode for components
        if 'circuit_breaker' in trader.components:
            trader.components['circuit_breaker'].set_test_mode(True)
    
    # Start the trader
    trader.start()


