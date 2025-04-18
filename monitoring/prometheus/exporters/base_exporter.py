"""
Base Exporter Module for Prometheus Metrics

This module provides the base class for all Prometheus exporters in the
day trading system. It handles core functionality for exposing metrics
to the Prometheus server.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable

from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily, SummaryMetricFamily

logger = logging.getLogger(__name__)

class BaseExporter(ABC):
    """
    Abstract base class for all Prometheus exporters.
    Defines the interface and common functionality for metric collection.
    """
    
    def __init__(self, name: str, port: int = 8000, interval: int = 15, 
                 prefix: str = "trading_system"):
        """
        Initialize the base exporter.
        
        Args:
            name: Name of the exporter
            port: Port to expose metrics on
            interval: Collection interval in seconds
            prefix: Prefix for all metrics
        """
        self.name = name
        self.port = port
        self.interval = interval
        self.prefix = prefix
        
        # Collection thread
        self._collector_thread = None
        self._is_running = False
        
        # Metrics registry
        self._metrics: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.__class__.__name__} exporter on port {port}")
    
    def start(self) -> None:
        """Start the exporter server and collection thread."""
        try:
            # Check if the server is already running on this port
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('localhost', self.port))
                s.close()
                # Port is available, start the server
                start_http_server(self.port)
                logger.info(f"Started Prometheus HTTP server on port {self.port}")
            except socket.error:
                logger.warning(f"Port {self.port} is already in use, exporter metrics may already be available")
                # Don't raise an exception - the exporter might already be running
                # Continue with the collector thread
            
            # Start collection thread
            self._is_running = True
            self._collector_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True,
                name=f"{self.name}_collector"
            )
            self._collector_thread.start()
            
            logger.info(f"Started metrics collection thread for {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to start exporter: {str(e)}")
            # Log error but don't raise - allow system to continue running
    
    def stop(self) -> None:
        """Stop the collection thread."""
        self._is_running = False
        if self._collector_thread and self._collector_thread.is_alive():
            self._collector_thread.join(timeout=5.0)
            logger.info(f"Stopped metrics collection for {self.name}")
    
    def _collection_loop(self) -> None:
        """Main collection loop that runs in a separate thread."""
        while self._is_running:
            try:
                self.collect()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
                time.sleep(self.interval)  # Still sleep to avoid tight loop on error
    
    @abstractmethod
    def collect(self) -> None:
        """
        Collect metrics. To be implemented by subclasses.
        """
        pass
    
    def _create_metric_name(self, name: str) -> str:
        """
        Create a prefixed metric name.
        
        Args:
            name: Base metric name
            
        Returns:
            Prefixed metric name
        """
        return f"{self.prefix}_{self.name}_{name}"
    
    def create_counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Counter:
        """
        Create and register a Prometheus counter.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional list of label names
            
        Returns:
            The created Counter
        """
        metric_name = self._create_metric_name(name)
        counter = Counter(
            metric_name,
            description,
            labelnames=labels or []
        )
        self._metrics[name] = counter
        return counter
    
    def create_gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Gauge:
        """
        Create and register a Prometheus gauge.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional list of label names
            
        Returns:
            The created Gauge
        """
        metric_name = self._create_metric_name(name)
        gauge = Gauge(
            metric_name,
            description,
            labelnames=labels or []
        )
        self._metrics[name] = gauge
        return gauge
    
    def create_histogram(self, name: str, description: str, labels: Optional[List[str]] = None,
                        buckets: Optional[List[float]] = None) -> Histogram:
        """
        Create and register a Prometheus histogram.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional list of label names
            buckets: Optional bucket specification
            
        Returns:
            The created Histogram
        """
        metric_name = self._create_metric_name(name)
        histogram = Histogram(
            metric_name,
            description,
            labelnames=labels or [],
            buckets=buckets
        )
        self._metrics[name] = histogram
        return histogram
    
    def create_summary(self, name: str, description: str, labels: Optional[List[str]] = None) -> Summary:
        """
        Create and register a Prometheus summary.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Optional list of label names
            
        Returns:
            The created Summary
        """
        metric_name = self._create_metric_name(name)
        summary = Summary(
            metric_name,
            description,
            labelnames=labels or []
        )
        self._metrics[name] = summary
        return summary
    
    def get_metric(self, name: str) -> Any:
        """
        Get a registered metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            The requested metric or None if not found
        """
        return self._metrics.get(name)


class CustomCollector(ABC):
    """
    Base class for custom Prometheus collectors.
    """
    
    def __init__(self, name: str, prefix: str = "trading_system"):
        """
        Initialize the collector.
        
        Args:
            name: Name of the collector
            prefix: Prefix for all metrics
        """
        self.name = name
        self.prefix = prefix
        REGISTRY.register(self)
        logger.info(f"Registered custom collector: {name}")
    
    @abstractmethod
    def collect(self):
        """
        Collect metrics to be exposed.
        To be implemented by subclasses.
        """
        pass
    
    def _create_metric_name(self, name: str) -> str:
        """
        Create a prefixed metric name.
        
        Args:
            name: Base metric name
            
        Returns:
            Prefixed metric name
        """
        return f"{self.prefix}_{self.name}_{name}"
