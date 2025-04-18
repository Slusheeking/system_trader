#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monitor Manager
--------------
Centralized monitoring initialization and management for the trading system.
Handles setup of all monitoring components and provides a unified interface.
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
import threading

# Import monitoring components
from monitoring.prometheus import setup_monitoring, get_exporter
from monitoring.prometheus.exporters.base_exporter import BaseExporter
from monitoring.prometheus.exporters.system_metrics_exporter import SystemMetricsExporter
from monitoring.prometheus.exporters.trading_metrics_exporter import TradingMetricsExporter
from monitoring.prometheus.exporters.model_metrics_exporter import ModelMetricsExporter

# Import utility modules
from utils.logging import setup_logger
from utils.config_loader import load_config

# Setup logging
logger = setup_logger('monitor_manager')


class MonitorManager:
    """
    Centralized monitoring manager for the trading system.
    Handles initialization and management of all monitoring components.
    """
    
    def __init__(self, config_path: str = 'config/system_config.yaml'):
        """
        Initialize the monitoring manager.
        
        Args:
            config_path: Path to system configuration file
        """
        self.config = load_config(config_path)
        self.monitoring_config = self.config.get('monitoring', {})
        self.exporters: Dict[str, BaseExporter] = {}
        self.components_registered = set()
        self.health_check_interval = self.monitoring_config.get('health_check_interval', 60)
        self.component_health_status = {}
        
        # Health check thread
        self._stop_health_check = threading.Event()
        self._health_check_thread = None
        
        # Initialize monitoring
        self._init_monitoring()
        
        logger.info("Monitor Manager initialized")
    
    def _init_monitoring(self):
        """
        Initialize all monitoring components.
        """
        try:
            # Check if monitoring is enabled
            if not self.monitoring_config.get('enabled', True):
                logger.info("Monitoring is disabled in configuration")
                return
            
            # Initialize Prometheus exporters
            prometheus_config = self.monitoring_config.get('prometheus', {})
            if prometheus_config.get('enabled', True):
                self._init_prometheus_exporters(prometheus_config)
            
            # Start health check thread
            self._start_health_check_thread()
            
            logger.info("Monitoring initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing monitoring: {str(e)}")
    
    def _init_prometheus_exporters(self, prometheus_config: Dict[str, Any]):
        """
        Initialize Prometheus exporters.
        
        Args:
            prometheus_config: Prometheus configuration dictionary
        """
        # Try to load monitoring config from dedicated file first
        try:
            from utils.config_loader import load_config
            monitoring_config = load_config('config/monitoring.json')
            if monitoring_config:
                prometheus_config = monitoring_config.get('prometheus', prometheus_config)
                logger.info("Loaded monitoring configuration from config/monitoring.json")
        except Exception as e:
            logger.warning(f"Could not load monitoring.json config, using system_config.yaml: {str(e)}")
            # Continue with prometheus_config from system_config.yaml
        
        # Get exporter configurations
        exporter_configs = prometheus_config.get('exporters', {})
        
        # Initialize system metrics exporter if enabled
        system_config = exporter_configs.get('system', {})
        if system_config.get('enabled', True):
            try:
                # Get port from config or use default
                port = system_config.get('port', 9001)  # Use 9001 instead of 8001
                
                # Clear existing metrics with similar names to prevent duplication
                import prometheus_client
                from prometheus_client.core import REGISTRY
                
                # Get list of collector names to remove by prefix
                collectors_to_remove = []
                for collector in list(REGISTRY._collector_to_names.keys()):
                    for name in REGISTRY._collector_to_names[collector]:
                        if name.startswith('trading_system_system_'):
                            collectors_to_remove.append(collector)
                            break
                
                # Remove collectors to prevent duplication
                for collector in collectors_to_remove:
                    try:
                        REGISTRY.unregister(collector)
                    except:
                        pass
                
                system_exporter = SystemMetricsExporter(
                    port=port,
                    interval=system_config.get('interval', 15)
                )
                system_exporter.start()
                self.exporters['system'] = system_exporter
                logger.info("System metrics exporter initialized")
            except Exception as e:
                logger.error(f"Error initializing system metrics exporter: {str(e)}")
        
        # Initialize trading metrics exporter if enabled
        trading_config = exporter_configs.get('trading', {})
        if trading_config.get('enabled', True):
            try:
                # Get port from config or use default
                port = trading_config.get('port', 9002)  # Use 9002 instead of 8002
                
                # Clear existing metrics with similar names to prevent duplication
                import prometheus_client
                from prometheus_client.core import REGISTRY
                
                # Get list of collector names to remove by prefix
                collectors_to_remove = []
                for collector in list(REGISTRY._collector_to_names.keys()):
                    for name in REGISTRY._collector_to_names[collector]:
                        if name.startswith('trading_system_trading_'):
                            collectors_to_remove.append(collector)
                            break
                
                # Remove collectors to prevent duplication
                for collector in collectors_to_remove:
                    try:
                        REGISTRY.unregister(collector)
                    except:
                        pass
                
                trading_exporter = TradingMetricsExporter(
                    port=port,
                    interval=trading_config.get('interval', 5)
                )
                trading_exporter.start()
                self.exporters['trading'] = trading_exporter
                logger.info("Trading metrics exporter initialized")
            except Exception as e:
                logger.error(f"Error initializing trading metrics exporter: {str(e)}")
        
        # Initialize model metrics exporter if enabled
        model_config = exporter_configs.get('model', {})
        if model_config.get('enabled', True):
            try:
                # Get port from config or use default
                port = model_config.get('port', 9003)  # Use 9003 instead of 8003
                
                # Clear existing metrics with similar names to prevent duplication
                import prometheus_client
                from prometheus_client.core import REGISTRY
                
                # Get list of collector names to remove by prefix
                collectors_to_remove = []
                for collector in list(REGISTRY._collector_to_names.keys()):
                    for name in REGISTRY._collector_to_names[collector]:
                        if name.startswith('trading_system_model_'):
                            collectors_to_remove.append(collector)
                            break
                
                # Remove collectors to prevent duplication
                for collector in collectors_to_remove:
                    try:
                        REGISTRY.unregister(collector)
                    except:
                        pass
                
                # Now initialize the model exporter
                model_exporter = ModelMetricsExporter(
                    port=port,
                    interval=model_config.get('interval', 30)
                )
                model_exporter.start()
                self.exporters['model'] = model_exporter
                logger.info("Model metrics exporter initialized")
            except Exception as e:
                logger.error(f"Error initializing model metrics exporter: {str(e)}")
    
    def register_component(self, component_name: str, component: Any):
        """
        Register a component with the monitoring system.
        
        Args:
            component_name: Name of the component
            component: Component instance
        """
        try:
            # Add to registered components
            self.components_registered.add(component_name)
            
            # Initialize health status
            self.component_health_status[component_name] = {
                'status': 'healthy',
                'last_heartbeat': time.time(),
                'error_count': 0,
                'last_error': None
            }
            
            # Log registration
            logger.info(f"Component registered with monitoring: {component_name}")
            
            # Update component health metric if trading exporter is available
            if 'trading' in self.exporters:
                trading_exporter = self.exporters['trading']
                component_health_metric = trading_exporter.get_metric("component_health")
                if component_health_metric:
                    component_health_metric.labels(component=component_name).set(2)  # 2 = healthy
            
        except Exception as e:
            logger.error(f"Error registering component {component_name}: {str(e)}")
    
    def update_component_health(self, component_name: str, status: str, error: Optional[str] = None):
        """
        Update health status for a component.
        
        Args:
            component_name: Name of the component
            status: Health status ('healthy', 'degraded', or 'down')
            error: Optional error message
        """
        try:
            # Check if component is registered
            if component_name not in self.components_registered:
                self.register_component(component_name, None)
            
            # Update health status
            health_status = self.component_health_status.get(component_name, {})
            health_status['status'] = status
            health_status['last_heartbeat'] = time.time()
            
            if error:
                health_status['error_count'] += 1
                health_status['last_error'] = error
            
            self.component_health_status[component_name] = health_status
            
            # Update component health metric if trading exporter is available
            if 'trading' in self.exporters:
                trading_exporter = self.exporters['trading']
                component_health_metric = trading_exporter.get_metric("component_health")
                if component_health_metric:
                    # Convert status to numeric value
                    status_value = {
                        'healthy': 2,
                        'degraded': 1,
                        'down': 0
                    }.get(status, 0)
                    
                    component_health_metric.labels(component=component_name).set(status_value)
            
            # Log status change if not healthy
            if status != 'healthy':
                logger.warning(f"Component {component_name} health status: {status}")
                if error:
                    logger.warning(f"Component {component_name} error: {error}")
            
        except Exception as e:
            logger.error(f"Error updating component health for {component_name}: {str(e)}")
    
    def record_heartbeat(self, component_name: str):
        """
        Record a heartbeat for a component.
        
        Args:
            component_name: Name of the component
        """
        try:
            # Check if component is registered
            if component_name not in self.components_registered:
                self.register_component(component_name, None)
            
            # Update last heartbeat
            health_status = self.component_health_status.get(component_name, {})
            health_status['last_heartbeat'] = time.time()
            self.component_health_status[component_name] = health_status
            
            # Update component heartbeat metric if trading exporter is available
            if 'trading' in self.exporters:
                trading_exporter = self.exporters['trading']
                heartbeat_metric = trading_exporter.get_metric("component_last_heartbeat_seconds")
                if heartbeat_metric:
                    heartbeat_metric.labels(component=component_name).set(0)  # 0 seconds since last heartbeat
            
        except Exception as e:
            logger.error(f"Error recording heartbeat for {component_name}: {str(e)}")
    
    def record_error(self, component_name: str, error_type: str, severity: str, error_message: str):
        """
        Record an error for a component.
        
        Args:
            component_name: Name of the component
            error_type: Type of error
            severity: Error severity ('critical', 'error', or 'warning')
            error_message: Error message
        """
        try:
            # Check if component is registered
            if component_name not in self.components_registered:
                self.register_component(component_name, None)
            
            # Update health status based on severity
            if severity == 'critical':
                self.update_component_health(component_name, 'down', error_message)
            elif severity == 'error':
                self.update_component_health(component_name, 'degraded', error_message)
            
            # Increment error counter if trading exporter is available
            if 'trading' in self.exporters:
                trading_exporter = self.exporters['trading']
                error_metric = trading_exporter.get_metric("errors_total")
                if error_metric:
                    error_metric.labels(
                        component=component_name,
                        severity=severity,
                        error_type=error_type
                    ).inc()
            
            # Log error
            if severity == 'critical':
                logger.critical(f"{component_name} - {error_type}: {error_message}")
            elif severity == 'error':
                logger.error(f"{component_name} - {error_type}: {error_message}")
            else:
                logger.warning(f"{component_name} - {error_type}: {error_message}")
            
        except Exception as e:
            logger.error(f"Error recording error for {component_name}: {str(e)}")
    
    def record_metric(self, metric_type: str, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a metric value.
        
        Args:
            metric_type: Type of metric ('system', 'trading', or 'model')
            metric_name: Name of the metric
            value: Metric value
            labels: Optional metric labels
        """
        try:
            # Get appropriate exporter
            exporter = self.exporters.get(metric_type)
            if not exporter:
                logger.warning(f"No exporter found for metric type: {metric_type}")
                return
            
            # Get metric
            metric = exporter.get_metric(metric_name)
            if not metric:
                logger.warning(f"Metric not found: {metric_name}")
                return
            
            # Update metric value
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {str(e)}")
    
    def increment_counter(self, metric_type: str, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            metric_type: Type of metric ('system', 'trading', or 'model')
            metric_name: Name of the metric
            labels: Optional metric labels
        """
        try:
            # Get appropriate exporter
            exporter = self.exporters.get(metric_type)
            if not exporter:
                logger.warning(f"No exporter found for metric type: {metric_type}")
                return
            
            # Get metric
            metric = exporter.get_metric(metric_name)
            if not metric:
                logger.warning(f"Metric not found: {metric_name}")
                return
            
            # Increment counter
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
            
        except Exception as e:
            logger.error(f"Error incrementing counter {metric_name}: {str(e)}")
    
    def observe_histogram(self, metric_type: str, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Observe a value for a histogram metric.
        
        Args:
            metric_type: Type of metric ('system', 'trading', or 'model')
            metric_name: Name of the metric
            value: Value to observe
            labels: Optional metric labels
        """
        try:
            # Get appropriate exporter
            exporter = self.exporters.get(metric_type)
            if not exporter:
                logger.warning(f"No exporter found for metric type: {metric_type}")
                return
            
            # Get metric
            metric = exporter.get_metric(metric_name)
            if not metric:
                logger.warning(f"Metric not found: {metric_name}")
                return
            
            # Observe value
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
            
        except Exception as e:
            logger.error(f"Error observing histogram {metric_name}: {str(e)}")
    
    def get_component_health(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health status for a component or all components.
        
        Args:
            component_name: Optional name of the component
            
        Returns:
            Dictionary with health status
        """
        if component_name:
            return self.component_health_status.get(component_name, {})
        else:
            return self.component_health_status
    
    def _check_component_health(self):
        """
        Check health of all registered components.
        """
        current_time = time.time()
        
        for component_name, health_status in self.component_health_status.items():
            try:
                # Check last heartbeat
                last_heartbeat = health_status.get('last_heartbeat', 0)
                seconds_since_heartbeat = current_time - last_heartbeat
                
                # Update heartbeat metric if trading exporter is available
                if 'trading' in self.exporters:
                    trading_exporter = self.exporters['trading']
                    heartbeat_metric = trading_exporter.get_metric("component_last_heartbeat_seconds")
                    if heartbeat_metric:
                        heartbeat_metric.labels(component=component_name).set(seconds_since_heartbeat)
                
                # Check if component is down based on heartbeat
                if seconds_since_heartbeat > 300:  # 5 minutes
                    # Update component health status
                    health_status['status'] = 'down'
                    self.component_health_status[component_name] = health_status
                    
                    # Update component health metric if trading exporter is available
                    if 'trading' in self.exporters:
                        trading_exporter = self.exporters['trading']
                        component_health_metric = trading_exporter.get_metric("component_health")
                        if component_health_metric:
                            component_health_metric.labels(component=component_name).set(0)  # 0 = down
                    
                    logger.warning(f"Component {component_name} marked as down due to missing heartbeat")
                
            except Exception as e:
                logger.error(f"Error checking health for component {component_name}: {str(e)}")
    
    def _start_health_check_thread(self):
        """
        Start the health check thread.
        """
        if self._health_check_thread is not None and self._health_check_thread.is_alive():
            return
        
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(target=self._health_check_loop)
        self._health_check_thread.daemon = True
        self._health_check_thread.start()
        logger.info("Health check thread started")
    
    def _stop_health_check_thread(self):
        """
        Stop the health check thread.
        """
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            return
        
        self._stop_health_check.set()
        self._health_check_thread.join(timeout=1.0)
        logger.info("Health check thread stopped")
    
    def _health_check_loop(self):
        """
        Health check loop.
        """
        while not self._stop_health_check.is_set():
            try:
                # Check component health
                self._check_component_health()
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
            
            # Sleep for the interval
            self._stop_health_check.wait(self.health_check_interval)
    
    def shutdown(self):
        """
        Shutdown monitoring.
        """
        try:
            # Stop health check thread
            self._stop_health_check_thread()
            
            # Stop all exporters
            for name, exporter in self.exporters.items():
                try:
                    exporter.stop()
                    logger.info(f"Stopped {name} exporter")
                except Exception as e:
                    logger.error(f"Error stopping {name} exporter: {str(e)}")
            
            logger.info("Monitoring shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down monitoring: {str(e)}")
    
    def __del__(self):
        """
        Cleanup on deletion.
        """
        self.shutdown()
    
    def start(self):
        """
        Start the monitoring system.
        This method is called by the run.py script.
        """
        logger.info("Starting monitoring system")
        # The monitoring system is already started in __init__
        # This method is just a placeholder for the run.py script
        pass


# Singleton instance
_monitor_manager = None


def get_monitor_manager(config_path: str = 'config/system_config.yaml') -> MonitorManager:
    """
    Get or create the monitor manager instance.
    
    Args:
        config_path: Path to system configuration file
        
    Returns:
        MonitorManager instance
    """
    global _monitor_manager
    
    if _monitor_manager is None:
        _monitor_manager = MonitorManager(config_path)
    
    return _monitor_manager


# Convenience functions for monitoring

def register_component(component_name: str, component: Any):
    """
    Register a component with the monitoring system.
    
    Args:
        component_name: Name of the component
        component: Component instance
    """
    monitor = get_monitor_manager()
    monitor.register_component(component_name, component)


def update_health(component_name: str, status: str, error: Optional[str] = None):
    """
    Update health status for a component.
    
    Args:
        component_name: Name of the component
        status: Health status ('healthy', 'degraded', or 'down')
        error: Optional error message
    """
    monitor = get_monitor_manager()
    monitor.update_component_health(component_name, status, error)


def record_heartbeat(component_name: str):
    """
    Record a heartbeat for a component.
    
    Args:
        component_name: Name of the component
    """
    monitor = get_monitor_manager()
    monitor.record_heartbeat(component_name)


def record_error(component_name: str, error_type: str, severity: str, error_message: str):
    """
    Record an error for a component.
    
    Args:
        component_name: Name of the component
        error_type: Type of error
        severity: Error severity ('critical', 'error', or 'warning')
        error_message: Error message
    """
    monitor = get_monitor_manager()
    monitor.record_error(component_name, error_type, severity, error_message)


def record_metric(metric_type: str, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """
    Record a metric value.
    
    Args:
        metric_type: Type of metric ('system', 'trading', or 'model')
        metric_name: Name of the metric
        value: Metric value
        labels: Optional metric labels
    """
    monitor = get_monitor_manager()
    monitor.record_metric(metric_type, metric_name, value, labels)


def increment_counter(metric_type: str, metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Increment a counter metric.
    
    Args:
        metric_type: Type of metric ('system', 'trading', or 'model')
        metric_name: Name of the metric
        labels: Optional metric labels
    """
    monitor = get_monitor_manager()
    monitor.increment_counter(metric_type, metric_name, labels)


def observe_histogram(metric_type: str, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """
    Observe a value for a histogram metric.
    
    Args:
        metric_type: Type of metric ('system', 'trading', or 'model')
        metric_name: Name of the metric
        value: Value to observe
        labels: Optional metric labels
    """
    monitor = get_monitor_manager()
    monitor.observe_histogram(metric_type, metric_name, value, labels)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Manager for Trading System')
    parser.add_argument('--config', type=str, default='config/system_config.yaml', help='Path to configuration file')
    parser.add_argument('--status', action='store_true', help='Get component health status')
    parser.add_argument('--component', type=str, help='Specific component to check')
    
    args = parser.parse_args()
    
    # Create monitor manager
    monitor_manager = MonitorManager(args.config)
    
    if args.status:
        # Get and print component health status
        if args.component:
            status = monitor_manager.get_component_health(args.component)
            print(f"Health status for {args.component}: {status}")
        else:
            status = monitor_manager.get_component_health()
            import json
            print(json.dumps(status, indent=2))
    else:
        print("Monitor manager initialized. Use --status to check component health.")
