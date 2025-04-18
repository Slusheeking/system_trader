"""
Prometheus Integration Package for Day Trading System

This package provides Prometheus integration for monitoring all aspects of the
day trading system including system metrics, trading performance,
model behavior, and data quality.
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Import exporters
from monitoring.prometheus.exporters.base_exporter import BaseExporter, CustomCollector
from monitoring.prometheus.exporters.system_metrics_exporter import SystemMetricsExporter
from monitoring.prometheus.exporters.trading_metrics_exporter import TradingMetricsExporter
from monitoring.prometheus.exporters.model_metrics_exporter import ModelMetricsExporter

# Export symbols
__all__ = [
    'BaseExporter',
    'CustomCollector',
    'SystemMetricsExporter',
    'TradingMetricsExporter',
    'ModelMetricsExporter',
    'setup_monitoring',
    'get_exporter'
]

# Global registry of exporters
_exporters: Dict[str, BaseExporter] = {}

def setup_monitoring(
    config: Dict[str, Any],
    portfolio_manager=None,
    model_interfaces=None,
    data_collectors=None,
    trade_manager=None,
    mlflow_uri=None
) -> Dict[str, BaseExporter]:
    """
    Set up monitoring exporters based on configuration.
    
    Args:
        config: Monitoring configuration
        portfolio_manager: Portfolio management component
        model_interfaces: Model interfaces
        data_collectors: Data collection components
        trade_manager: Trade execution component
        mlflow_uri: URI for MLflow tracking server
        
    Returns:
        Dictionary of initialized exporters
    """
    global _exporters
    
    try:
        # Clear any existing exporters
        stop_all_exporters()
        _exporters = {}
        
        # Get exporter configurations
        system_config = config.get("system", {})
        trading_config = config.get("trading", {})
        model_config = config.get("model", {})
        
        # Create system metrics exporter if enabled
        if system_config.get("enabled", True):
            system_exporter = SystemMetricsExporter(
                port=system_config.get("port", 8001),
                interval=system_config.get("interval", 15)
            )
            system_exporter.start()
            _exporters["system"] = system_exporter
            logger.info("Started system metrics exporter")
        
        # Create trading metrics exporter if enabled
        if trading_config.get("enabled", True):
            trading_exporter = TradingMetricsExporter(
                portfolio_manager=portfolio_manager,
                data_collectors=data_collectors,
                model_manager=model_interfaces,
                trade_manager=trade_manager,
                port=trading_config.get("port", 8002),
                interval=trading_config.get("interval", 5)
            )
            trading_exporter.start()
            _exporters["trading"] = trading_exporter
            logger.info("Started trading metrics exporter")
        
        # Create model metrics exporter if enabled
        if model_config.get("enabled", True):
            model_exporter = ModelMetricsExporter(
                model_interfaces=model_interfaces,
                mlflow_uri=mlflow_uri,
                port=model_config.get("port", 8003),
                interval=model_config.get("interval", 30)
            )
            model_exporter.start()
            _exporters["model"] = model_exporter
            logger.info("Started model metrics exporter")
        
        return _exporters
        
    except Exception as e:
        logger.error(f"Error setting up monitoring: {str(e)}")
        # Try to stop any exporters that were started
        stop_all_exporters()
        raise

def get_exporter(name: str) -> Optional[BaseExporter]:
    """
    Get an exporter by name.
    
    Args:
        name: Name of the exporter
        
    Returns:
        The exporter or None if not found
    """
    return _exporters.get(name)

def stop_all_exporters() -> None:
    """Stop all exporters."""
    global _exporters
    
    for name, exporter in _exporters.items():
        try:
            exporter.stop()
            logger.info(f"Stopped {name} exporter")
        except Exception as e:
            logger.error(f"Error stopping {name} exporter: {str(e)}")
