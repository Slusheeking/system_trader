#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Execution Monitor
---------------
Monitors trade execution quality and performance metrics.
Tracks fills, slippage, latency, and other execution quality metrics.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import threading
import uuid
import statistics
from collections import deque

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from utils.metrics import calculate_metrics

# Setup logging
logger = setup_logger('execution_monitor')


class ExecutionMetrics:
    """
    Tracks and calculates execution quality metrics.
    """
    
    def __init__(self):
        """
        Initialize execution metrics.
        """
        # Timing metrics
        self.submission_times = {}
        self.acknowledgment_times = {}
        self.fill_times = {}
        
        # Price metrics
        self.expected_prices = {}
        self.executed_prices = {}
        
        # Fill metrics
        self.filled_quantities = {}
        self.requested_quantities = {}
        
        # Performance metrics
        self.latency_ms = deque(maxlen=100)  # Last 100 latency measurements
        self.slippage_bps = deque(maxlen=100)  # Last 100 slippage measurements in basis points
        self.fill_ratios = deque(maxlen=100)  # Last 100 fill ratio measurements
        
        # Rejection metrics
        self.rejection_reasons = {}
        self.rejection_count = 0
        self.submission_count = 0
        
        # Broker metrics
        self.broker_latencies = {}
        self.broker_slippages = {}
        self.broker_fill_ratios = {}
        self.broker_rejections = {}
        
        # Market data timestamps
        self.last_market_data = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def record_submission(self, order_id: str, timestamp: Optional[datetime] = None, 
                        expected_price: Optional[float] = None, quantity: Optional[float] = None,
                        broker: Optional[str] = None):
        """
        Record order submission.
        
        Args:
            order_id: Order ID
            timestamp: Submission timestamp
            expected_price: Expected execution price
            quantity: Requested quantity
            broker: Broker name
        """
        with self.lock:
            # Use current time if timestamp not provided
            timestamp = timestamp or datetime.now()
            
            # Record metrics
            self.submission_times[order_id] = timestamp
            
            if expected_price is not None:
                self.expected_prices[order_id] = expected_price
            
            if quantity is not None:
                self.requested_quantities[order_id] = quantity
            
            # Update submission count
            self.submission_count += 1
            
            # Update broker metrics
            if broker:
                if broker not in self.broker_rejections:
                    self.broker_rejections[broker] = 0
                if broker not in self.broker_latencies:
                    self.broker_latencies[broker] = deque(maxlen=100)
                if broker not in self.broker_slippages:
                    self.broker_slippages[broker] = deque(maxlen=100)
                if broker not in self.broker_fill_ratios:
                    self.broker_fill_ratios[broker] = deque(maxlen=100)
    
    def record_acknowledgment(self, order_id: str, timestamp: Optional[datetime] = None):
        """
        Record order acknowledgment.
        
        Args:
            order_id: Order ID
            timestamp: Acknowledgment timestamp
        """
        with self.lock:
            # Use current time if timestamp not provided
            timestamp = timestamp or datetime.now()
            
            # Record acknowledgment time
            self.acknowledgment_times[order_id] = timestamp
            
            # Calculate latency if we have a submission time
            if order_id in self.submission_times:
                latency_seconds = (timestamp - self.submission_times[order_id]).total_seconds()
                latency_ms = latency_seconds * 1000
                self.latency_ms.append(latency_ms)
                
                # Update broker latency metrics
                broker = self._get_order_broker(order_id)
                if broker and broker in self.broker_latencies:
                    self.broker_latencies[broker].append(latency_ms)
    
    def record_fill(self, order_id: str, timestamp: Optional[datetime] = None, 
                  executed_price: Optional[float] = None, filled_quantity: Optional[float] = None):
        """
        Record order fill.
        
        Args:
            order_id: Order ID
            timestamp: Fill timestamp
            executed_price: Executed price
            filled_quantity: Filled quantity
        """
        with self.lock:
            # Use current time if timestamp not provided
            timestamp = timestamp or datetime.now()
            
            # Record fill time
            self.fill_times[order_id] = timestamp
            
            # Record executed price
            if executed_price is not None:
                self.executed_prices[order_id] = executed_price
            
            # Record filled quantity
            if filled_quantity is not None:
                self.filled_quantities[order_id] = filled_quantity
            
            # Calculate slippage if we have expected and executed prices
            if order_id in self.expected_prices and executed_price is not None:
                expected_price = self.expected_prices[order_id]
                slippage = (executed_price - expected_price) / expected_price
                slippage_bps = slippage * 10000  # Convert to basis points
                self.slippage_bps.append(slippage_bps)
                
                # Update broker slippage metrics
                broker = self._get_order_broker(order_id)
                if broker and broker in self.broker_slippages:
                    self.broker_slippages[broker].append(slippage_bps)
            
            # Calculate fill ratio if we have requested and filled quantities
            if order_id in self.requested_quantities and filled_quantity is not None:
                requested_qty = self.requested_quantities[order_id]
                fill_ratio = filled_quantity / requested_qty if requested_qty > 0 else 0
                self.fill_ratios.append(fill_ratio)
                
                # Update broker fill ratio metrics
                broker = self._get_order_broker(order_id)
                if broker and broker in self.broker_fill_ratios:
                    self.broker_fill_ratios[broker].append(fill_ratio)
    
    def record_rejection(self, order_id: str, reason: Optional[str] = None, broker: Optional[str] = None):
        """
        Record order rejection.
        
        Args:
            order_id: Order ID
            reason: Rejection reason
            broker: Broker name
        """
        with self.lock:
            # Record rejection reason
            if reason:
                self.rejection_reasons[order_id] = reason
                
                # Update rejection count for this reason
                if reason not in self.rejection_reasons:
                    self.rejection_reasons[reason] = 0
                self.rejection_reasons[reason] += 1
            
            # Update rejection count
            self.rejection_count += 1
            
            # Update broker rejection metrics
            if broker and broker in self.broker_rejections:
                self.broker_rejections[broker] += 1
    
    def record_market_data(self, symbol: str, timestamp: datetime, data: Dict[str, Any]):
        """
        Record latest market data for a symbol.
        
        Args:
            symbol: Symbol
            timestamp: Market data timestamp
            data: Market data dictionary
        """
        with self.lock:
            # Record market data timestamp and data
            self.last_market_data[symbol] = {
                'timestamp': timestamp,
                'data': data
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.
        
        Returns:
            Dictionary with execution metrics
        """
        with self.lock:
            # Calculate average metrics
            avg_latency_ms = statistics.mean(self.latency_ms) if self.latency_ms else 0
            avg_slippage_bps = statistics.mean(self.slippage_bps) if self.slippage_bps else 0
            avg_fill_ratio = statistics.mean(self.fill_ratios) if self.fill_ratios else 0
            rejection_rate = self.rejection_count / self.submission_count if self.submission_count > 0 else 0
            
            # Calculate broker metrics
            broker_metrics = {}
            for broker in self.broker_latencies:
                broker_metrics[broker] = {
                    'avg_latency_ms': statistics.mean(self.broker_latencies[broker]) if self.broker_latencies[broker] else 0,
                    'avg_slippage_bps': statistics.mean(self.broker_slippages[broker]) if self.broker_slippages[broker] else 0,
                    'avg_fill_ratio': statistics.mean(self.broker_fill_ratios[broker]) if self.broker_fill_ratios[broker] else 0,
                    'rejection_count': self.broker_rejections.get(broker, 0),
                    'rejection_rate': self.broker_rejections.get(broker, 0) / self.submission_count if self.submission_count > 0 else 0
                }
            
            # Build metrics dictionary
            metrics = {
                'avg_latency_ms': avg_latency_ms,
                'avg_slippage_bps': avg_slippage_bps,
                'avg_fill_ratio': avg_fill_ratio,
                'rejection_rate': rejection_rate,
                'submission_count': self.submission_count,
                'rejection_count': self.rejection_count,
                'broker_metrics': broker_metrics,
                'rejection_reasons': self.rejection_reasons,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
    
    def get_order_metrics(self, order_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dictionary with order metrics
        """
        with self.lock:
            metrics = {
                'order_id': order_id,
                'submission_time': self.submission_times.get(order_id),
                'acknowledgment_time': self.acknowledgment_times.get(order_id),
                'fill_time': self.fill_times.get(order_id),
                'expected_price': self.expected_prices.get(order_id),
                'executed_price': self.executed_prices.get(order_id),
                'requested_quantity': self.requested_quantities.get(order_id),
                'filled_quantity': self.filled_quantities.get(order_id),
                'rejection_reason': self.rejection_reasons.get(order_id)
            }
            
            # Calculate latency
            if order_id in self.submission_times and order_id in self.acknowledgment_times:
                latency_seconds = (self.acknowledgment_times[order_id] - self.submission_times[order_id]).total_seconds()
                metrics['latency_ms'] = latency_seconds * 1000
            
            # Calculate slippage
            if order_id in self.expected_prices and order_id in self.executed_prices:
                expected_price = self.expected_prices[order_id]
                executed_price = self.executed_prices[order_id]
                slippage = (executed_price - expected_price) / expected_price
                metrics['slippage_bps'] = slippage * 10000  # Convert to basis points
            
            # Calculate fill ratio
            if order_id in self.requested_quantities and order_id in self.filled_quantities:
                requested_qty = self.requested_quantities[order_id]
                filled_qty = self.filled_quantities[order_id]
                metrics['fill_ratio'] = filled_qty / requested_qty if requested_qty > 0 else 0
            
            # Format timestamps
            for key in ['submission_time', 'acknowledgment_time', 'fill_time']:
                if metrics[key] is not None:
                    metrics[key] = metrics[key].isoformat()
            
            return metrics
    
    def reset(self):
        """
        Reset execution metrics.
        """
        with self.lock:
            # Clear dictionaries
            self.submission_times.clear()
            self.acknowledgment_times.clear()
            self.fill_times.clear()
            self.expected_prices.clear()
            self.executed_prices.clear()
            self.filled_quantities.clear()
            self.requested_quantities.clear()
            self.rejection_reasons.clear()
            self.last_market_data.clear()
            
            # Clear deques
            self.latency_ms.clear()
            self.slippage_bps.clear()
            self.fill_ratios.clear()
            
            # Reset counters
            self.rejection_count = 0
            self.submission_count = 0
            
            # Reset broker metrics
            self.broker_latencies.clear()
            self.broker_slippages.clear()
            self.broker_fill_ratios.clear()
            self.broker_rejections.clear()
    
    def _get_order_broker(self, order_id: str) -> Optional[str]:
        """
        Get broker for an order. For Alpaca-specific implementation,
        we can assume all orders are with Alpaca.
        
        Args:
            order_id: Order ID
            
        Returns:
            Broker name or None if not found
        """
        # For Alpaca-specific implementation, just return "alpaca"
        return "alpaca"


class ExecutionAnalyzer:
    """
    Analyzes execution quality and performance.
    """
    
    def __init__(self, metrics: ExecutionMetrics):
        """
        Initialize execution analyzer.
        
        Args:
            metrics: ExecutionMetrics instance
        """
        self.metrics = metrics
    
    def analyze_latency(self) -> Dict[str, Any]:
        """
        Analyze execution latency.
        
        Returns:
            Dictionary with latency analysis
        """
        # Get metrics data
        metrics_data = self.metrics.get_metrics()
        latency_ms = self.metrics.latency_ms
        
        # Calculate statistics
        if latency_ms:
            percentiles = {
                'p50': np.percentile(latency_ms, 50),
                'p95': np.percentile(latency_ms, 95),
                'p99': np.percentile(latency_ms, 99)
            }
            std_dev = statistics.stdev(latency_ms) if len(latency_ms) > 1 else 0
        else:
            percentiles = {'p50': 0, 'p95': 0, 'p99': 0}
            std_dev = 0
        
        # Build analysis
        analysis = {
            'avg_latency_ms': metrics_data['avg_latency_ms'],
            'percentiles': percentiles,
            'std_dev': std_dev,
            'sample_size': len(latency_ms),
            'by_broker': {
                broker: {
                    'avg_latency_ms': metrics['avg_latency_ms'],
                    'sample_size': len(self.metrics.broker_latencies.get(broker, []))
                }
                for broker, metrics in metrics_data['broker_metrics'].items()
            }
        }
        
        return analysis
    
    def analyze_slippage(self) -> Dict[str, Any]:
        """
        Analyze price slippage.
        
        Returns:
            Dictionary with slippage analysis
        """
        # Get metrics data
        metrics_data = self.metrics.get_metrics()
        slippage_bps = self.metrics.slippage_bps
        
        # Calculate statistics
        if slippage_bps:
            percentiles = {
                'p50': np.percentile(slippage_bps, 50),
                'p5': np.percentile(slippage_bps, 5),
                'p95': np.percentile(slippage_bps, 95)
            }
            std_dev = statistics.stdev(slippage_bps) if len(slippage_bps) > 1 else 0
        else:
            percentiles = {'p50': 0, 'p5': 0, 'p95': 0}
            std_dev = 0
        
        # Build analysis
        analysis = {
            'avg_slippage_bps': metrics_data['avg_slippage_bps'],
            'percentiles': percentiles,
            'std_dev': std_dev,
            'sample_size': len(slippage_bps),
            'by_broker': {
                broker: {
                    'avg_slippage_bps': metrics['avg_slippage_bps'],
                    'sample_size': len(self.metrics.broker_slippages.get(broker, []))
                }
                for broker, metrics in metrics_data['broker_metrics'].items()
            }
        }
        
        return analysis
    
    def analyze_fills(self) -> Dict[str, Any]:
        """
        Analyze fill rates and quality.
        
        Returns:
            Dictionary with fill analysis
        """
        # Get metrics data
        metrics_data = self.metrics.get_metrics()
        fill_ratios = self.metrics.fill_ratios
        
        # Calculate statistics
        if fill_ratios:
            percentiles = {
                'p50': np.percentile(fill_ratios, 50),
                'p5': np.percentile(fill_ratios, 5),
                'p95': np.percentile(fill_ratios, 95)
            }
            std_dev = statistics.stdev(fill_ratios) if len(fill_ratios) > 1 else 0
        else:
            percentiles = {'p50': 0, 'p5': 0, 'p95': 0}
            std_dev = 0
        
        # Build analysis
        analysis = {
            'avg_fill_ratio': metrics_data['avg_fill_ratio'],
            'percentiles': percentiles,
            'std_dev': std_dev,
            'sample_size': len(fill_ratios),
            'by_broker': {
                broker: {
                    'avg_fill_ratio': metrics['avg_fill_ratio'],
                    'sample_size': len(self.metrics.broker_fill_ratios.get(broker, []))
                }
                for broker, metrics in metrics_data['broker_metrics'].items()
            }
        }
        
        return analysis
    
    def analyze_rejections(self) -> Dict[str, Any]:
        """
        Analyze order rejections.
        
        Returns:
            Dictionary with rejection analysis
        """
        # Get metrics data
        metrics_data = self.metrics.get_metrics()
        
        # Build analysis
        analysis = {
            'rejection_rate': metrics_data['rejection_rate'],
            'rejection_count': metrics_data['rejection_count'],
            'submission_count': metrics_data['submission_count'],
            'reasons': metrics_data['rejection_reasons'],
            'by_broker': {
                broker: {
                    'rejection_rate': metrics['rejection_rate'],
                    'rejection_count': metrics['rejection_count']
                }
                for broker, metrics in metrics_data['broker_metrics'].items()
            }
        }
        
        return analysis
    
    def analyze_market_impact(self) -> Dict[str, Any]:
        """
        Analyze market impact of orders.
        
        Returns:
            Dictionary with market impact analysis
        """
        # This requires more detailed market data than we're tracking here
        # Placeholder implementation for Alpaca-specific needs
        analysis = {
            'avg_price_impact_bps': 0,
            'by_symbol': {},
            'note': 'Market impact analysis requires detailed order book data from Alpaca'
        }
        
        return analysis
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive execution analysis.
        
        Returns:
            Dictionary with comprehensive analysis
        """
        analysis = {
            'latency': self.analyze_latency(),
            'slippage': self.analyze_slippage(),
            'fills': self.analyze_fills(),
            'rejections': self.analyze_rejections(),
            'market_impact': self.analyze_market_impact(),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis


class ExecutionMonitor:
    """
    Monitors trade execution quality and performance metrics.
    Specific implementation for Alpaca.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the execution monitor.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.metrics = ExecutionMetrics()
        self.analyzer = ExecutionAnalyzer(self.metrics)
        
        # Order tracking
        self.orders = {}
        self.last_order_update = {}
        
        # Alert thresholds
        self.latency_threshold_ms = self.config.get('latency_threshold_ms', 500)
        self.slippage_threshold_bps = self.config.get('slippage_threshold_bps', 20)
        self.rejection_rate_threshold = self.config.get('rejection_rate_threshold', 0.1)
        self.fill_ratio_threshold = self.config.get('fill_ratio_threshold', 0.9)
        
        # Alert counters
        self.latency_alerts = 0
        self.slippage_alerts = 0
        self.rejection_alerts = 0
        self.fill_ratio_alerts = 0
        
        # Alpaca-specific settings
        self.alpaca_rate_limit_alerts = 0
        self.alpaca_api_errors = 0
        
        # Performance tracking
        self.last_analysis_time = None
        self.analysis_interval = self.config.get('analysis_interval', 300)  # 5 minutes
        self.analysis_history = deque(maxlen=24)  # Last 24 analyses
        
        # Auto analysis thread
        self.auto_analysis_enabled = self.config.get('auto_analysis', {}).get('enabled', True)
        self.auto_analysis_interval = self.config.get('auto_analysis', {}).get('interval', 3600)  # 1 hour
        self._stop_auto_analysis = threading.Event()
        self._auto_analysis_thread = None
        
        # Start auto-analysis thread if enabled
        if self.auto_analysis_enabled:
            self._start_auto_analysis_thread()
        
        logger.info("Execution Monitor initialized for Alpaca")
    
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
    
    def track_order(self, order):
        """
        Track an order in the execution monitor.
        
        Args:
            order: Order object
        """
        order_id = order.request.client_order_id
        
        # Store order for monitoring
        self.orders[order_id] = order
        self.last_order_update[order_id] = datetime.now()
        
        # Record submission metrics
        try:
            # Get expected price (limit price for limit orders, market price for market orders)
            expected_price = None
            if order.request.order_type.value == 'limit':
                expected_price = order.request.price
            
            # Record submission
            self.metrics.record_submission(
                order_id=order_id,
                timestamp=order.request.timestamp,
                expected_price=expected_price,
                quantity=order.request.quantity,
                broker="alpaca"  # Hardcoded for Alpaca-specific implementation
            )
            
            logger.debug(f"Tracking order: {order_id}")
        
        except Exception as e:
            logger.error(f"Error tracking order {order_id}: {str(e)}")
    
    def update_order_status(self, order):
        """
        Update status for a tracked order.
        
        Args:
            order: Order object with updated status
        """
        order_id = order.request.client_order_id
        
        # Check if order is being tracked
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found in tracking")
            self.track_order(order)
            return
        
        # Check if this is a duplicate update
        previous_order = self.orders[order_id]
        if (previous_order.status == order.status and 
            previous_order.filled_quantity == order.filled_quantity):
            logger.debug(f"Duplicate order update for {order_id}, skipping")
            return
        
        # Update order in tracking
        self.orders[order_id] = order
        self.last_order_update[order_id] = datetime.now()
        
        try:
            # Record metrics based on order status
            if order.status.value == 'submitted':
                # Record acknowledgment
                self.metrics.record_acknowledgment(
                    order_id=order_id,
                    timestamp=order.last_updated
                )
            
            elif order.status.value in ['filled', 'partially_filled']:
                # Record fill
                self.metrics.record_fill(
                    order_id=order_id,
                    timestamp=order.last_updated,
                    executed_price=order.average_price,
                    filled_quantity=order.filled_quantity
                )
            
            elif order.status.value == 'rejected':
                # Record rejection
                self.metrics.record_rejection(
                    order_id=order_id,
                    reason=order.rejected_reason,
                    broker="alpaca"  # Hardcoded for Alpaca-specific implementation
                )
                
                # Check for specific Alpaca error patterns
                if order.rejected_reason:
                    if "rate limit" in order.rejected_reason.lower():
                        self.alpaca_rate_limit_alerts += 1
                        logger.warning(f"Alpaca rate limit exceeded, alert #{self.alpaca_rate_limit_alerts}")
                    elif "api" in order.rejected_reason.lower() and "error" in order.rejected_reason.lower():
                        self.alpaca_api_errors += 1
                        logger.warning(f"Alpaca API error: {order.rejected_reason}")
            
            # Check for alerts
            self._check_alerts(order_id)
            
            logger.debug(f"Updated order status: {order_id} => {order.status.value}")
        
        except Exception as e:
            logger.error(f"Error updating order status for {order_id}: {str(e)}")
    
    def record_market_data(self, symbol: str, data: Dict[str, Any]):
        """
        Record market data for a symbol.
        
        Args:
            symbol: Symbol
            data: Market data dictionary
        """
        try:
            self.metrics.record_market_data(symbol, datetime.now(), data)
        except Exception as e:
            logger.error(f"Error recording market data for {symbol}: {str(e)}")
    
    def _check_alerts(self, order_id: str):
        """
        Check for execution quality alerts for an order.
        
        Args:
            order_id: Order ID
        """
        order_metrics = self.metrics.get_order_metrics(order_id)
        
        # Check latency alert
        if 'latency_ms' in order_metrics and order_metrics['latency_ms'] > self.latency_threshold_ms:
            self.latency_alerts += 1
            logger.warning(f"High latency alert for order {order_id}: {order_metrics['latency_ms']:.2f} ms")
        
        # Check slippage alert
        if 'slippage_bps' in order_metrics:
            abs_slippage = abs(order_metrics['slippage_bps'])
            if abs_slippage > self.slippage_threshold_bps:
                self.slippage_alerts += 1
                logger.warning(f"High slippage alert for order {order_id}: {order_metrics['slippage_bps']:.2f} bps")
        
        # Check fill ratio alert
        if 'fill_ratio' in order_metrics and order_metrics['fill_ratio'] < self.fill_ratio_threshold:
            self.fill_ratio_alerts += 1
            logger.warning(f"Low fill ratio alert for order {order_id}: {order_metrics['fill_ratio']:.2f}")
        
        # Check overall metrics for rejection rate alert
        metrics = self.metrics.get_metrics()
        if metrics['rejection_rate'] > self.rejection_rate_threshold:
            self.rejection_alerts += 1
            logger.warning(f"High rejection rate alert: {metrics['rejection_rate']:.2f}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run a comprehensive execution analysis.
        
        Returns:
            Dictionary with analysis results
        """
        try:
            analysis = self.analyzer.get_full_analysis()
            
            # Add Alpaca-specific analysis
            analysis['alpaca'] = {
                'rate_limit_alerts': self.alpaca_rate_limit_alerts,
                'api_errors': self.alpaca_api_errors
            }
            
            # Save analysis to history
            self.analysis_history.append(analysis)
            self.last_analysis_time = datetime.now()
            
            logger.info("Completed execution analysis")
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error running execution analysis: {str(e)}")
            return {'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current execution metrics.
        
        Returns:
            Dictionary with execution metrics
        """
        try:
            metrics = self.metrics.get_metrics()
            
            # Add alert counts
            metrics['alerts'] = {
                'latency_alerts': self.latency_alerts,
                'slippage_alerts': self.slippage_alerts,
                'rejection_alerts': self.rejection_alerts,
                'fill_ratio_alerts': self.fill_ratio_alerts,
                'alpaca_rate_limit_alerts': self.alpaca_rate_limit_alerts,
                'alpaca_api_errors': self.alpaca_api_errors
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting execution metrics: {str(e)}")
            return {'error': str(e)}
    
    def get_order_metrics(self, order_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dictionary with order metrics
        """
        try:
            return self.metrics.get_order_metrics(order_id)
        except Exception as e:
            logger.error(f"Error getting metrics for order {order_id}: {str(e)}")
            return {'error': str(e)}
    
    def record(self, signal, report):
        """
        Record execution details and compute metrics.
        
        Args:
            signal: Trading signal that generated the order
            report: Execution report containing fill details
        """
        try:
            # Compute fill latency
            fill_latency = (report.fill_time - report.send_time).total_seconds() * 1000  # in milliseconds
            
            # Log execution details
            logger.info(f"Order executed: {report.order_id} for {report.symbol} - "
                       f"Quantity: {report.filled_quantity}/{report.requested_quantity}, "
                       f"Price: {report.executed_price}, Latency: {fill_latency:.2f}ms")
            
            # Calculate fill ratio
            fill_ratio = report.filled_quantity / report.requested_quantity if report.requested_quantity > 0 else 0
            
            # Emit metrics
            metrics_data = {
                'fill_latency_ms': fill_latency,
                'fill_ratio': fill_ratio,
                'slippage_bps': (report.executed_price - report.expected_price) / report.expected_price * 10000 
                                if report.expected_price else 0
            }
            
            # Track in execution metrics
            self.metrics.record_fill(
                order_id=report.order_id,
                timestamp=report.fill_time,
                executed_price=report.executed_price,
                filled_quantity=report.filled_quantity
            )
            
            # Check for alerts based on this execution
            self._check_alerts(report.order_id)
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error recording execution: {str(e)}")
            return {'error': str(e)}
    
    def check_order_timeliness(self):
        """
        Check for orders that haven't been updated recently.
        Specific to Alpaca to detect stalled orders.
        
        Returns:
            Dictionary with stalled orders information
        """
        now = datetime.now()
        stalled_orders = {}
        
        for order_id, order in self.orders.items():
            if order.status.value not in ['filled', 'canceled', 'rejected', 'expired']:
                last_update = self.last_order_update.get(order_id)
                if last_update and (now - last_update).total_seconds() > 300:  # 5 minutes
                    stalled_orders[order_id] = {
                        'symbol': order.request.symbol,
                        'status': order.status.value,
                        'time_since_update': (now - last_update).total_seconds(),
                        'submitted_at': order.request.timestamp.isoformat() if hasattr(order.request, 'timestamp') else None
                    }
        
        if stalled_orders:
            logger.warning(f"Found {len(stalled_orders)} stalled orders")
        
        return stalled_orders
    
    def reset_metrics(self):
        """
        Reset execution metrics.
        """
        try:
            self.metrics.reset()
            
            # Reset alert counters
            self.latency_alerts = 0
            self.slippage_alerts = 0
            self.rejection_alerts = 0
            self.fill_ratio_alerts = 0
            self.alpaca_rate_limit_alerts = 0
            self.alpaca_api_errors = 0
            
            logger.info("Reset execution metrics")
        
        except Exception as e:
            logger.error(f"Error resetting execution metrics: {str(e)}")
    
    def _start_auto_analysis_thread(self):
        """
        Start the auto-analysis thread.
        """
        if self._auto_analysis_thread is not None and self._auto_analysis_thread.is_alive():
            return
        
        self._stop_auto_analysis.clear()
        self._auto_analysis_thread = threading.Thread(target=self._auto_analysis_loop)
        self._auto_analysis_thread.daemon = True
        self._auto_analysis_thread.start()
        logger.info("Auto-analysis thread started")
    
    def _stop_auto_analysis_thread(self):
        """
        Stop the auto-analysis thread.
        """
        if self._auto_analysis_thread is None or not self._auto_analysis_thread.is_alive():
            return
        
        self._stop_auto_analysis.set()
        self._auto_analysis_thread.join(timeout=1.0)
        logger.info("Auto-analysis thread stopped")
    
    def _auto_analysis_loop(self):
        """
        Auto-analysis loop.
        """
        while not self._stop_auto_analysis.is_set():
            try:
                # Run analysis
                self.run_analysis()
                
                # Also check for stalled orders
                stalled_orders = self.check_order_timeliness()
                if stalled_orders:
                    # Log stalled orders for investigation
                    logger.warning(f"Stalled orders detected: {json.dumps(stalled_orders)}")
            except Exception as e:
                logger.error(f"Error in auto-analysis loop: {str(e)}")
            
            # Sleep for the interval
            self._stop_auto_analysis.wait(self.auto_analysis_interval)
    
    def __del__(self):
        """
        Cleanup on deletion.
        """
        self._stop_auto_analysis_thread()


# Default execution monitor instance
default_execution_monitor = None


def get_execution_monitor(config_path: Optional[str] = None) -> ExecutionMonitor:
    """
    Get or create the default execution monitor.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ExecutionMonitor instance
    """
    global default_execution_monitor
    
    if default_execution_monitor is None:
        default_execution_monitor = ExecutionMonitor(config_path)
    
    return default_execution_monitor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Execution Monitor for Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--metrics', action='store_true', help='Get execution metrics')
    parser.add_argument('--analyze', action='store_true', help='Run execution analysis')
    parser.add_argument('--reset', action='store_true', help='Reset execution metrics')
    parser.add_argument('--order', type=str, help='Get metrics for a specific order')
    parser.add_argument('--check-stalled', action='store_true', help='Check for stalled orders')
    
    args = parser.parse_args()
    
    # Create execution monitor
    execution_monitor = ExecutionMonitor(args.config)
    
    if args.metrics:
        # Get metrics
        metrics = execution_monitor.get_metrics()
        import json
        print(json.dumps(metrics, indent=2))
    
    elif args.analyze:
        # Run analysis
        analysis = execution_monitor.run_analysis()
        import json
        print(json.dumps(analysis, indent=2))
    
    elif args.reset:
        # Reset metrics
        execution_monitor.reset_metrics()
        print("Execution metrics reset")
    
    elif args.order:
        # Get order metrics
        order_metrics = execution_monitor.get_order_metrics(args.order)
        import json
        print(json.dumps(order_metrics, indent=2))
    
    elif args.check_stalled:
        # Check for stalled orders
        stalled_orders = execution_monitor.check_order_timeliness()
        if stalled_orders:
            print(f"Found {len(stalled_orders)} stalled orders:")
            import json
            print(json.dumps(stalled_orders, indent=2))
        else:
            print("No stalled orders found")
    
    else:
        parser.print_help()
