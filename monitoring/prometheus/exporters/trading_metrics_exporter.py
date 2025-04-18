"""
Trading Metrics Exporter for Prometheus

This module collects and exports trading-specific metrics including portfolio
performance, trade statistics, model performance, and system status.
"""

import logging
import datetime
import time
from typing import Dict, Any, List, Optional, Union, Set

import pandas as pd
import numpy as np

from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)

class TradingMetricsExporter(BaseExporter):
    """
    Exporter for trading metrics.
    
    Collects and exports metrics related to:
    - Portfolio performance
    - Trading activity
    - Model predictions
    - Data quality
    - System health
    """
    
    def __init__(self, portfolio_manager=None, data_collectors=None, 
                model_manager=None, trade_manager=None, port: int = 8002, 
                interval: int = 5):
        """
        Initialize the trading metrics exporter.
        
        Args:
            portfolio_manager: Portfolio management component
            data_collectors: Data collection components
            model_manager: Model management component
            trade_manager: Trade execution component
            port: Port to expose metrics on
            interval: Collection interval in seconds
        """
        super().__init__(name="trading", port=port, interval=interval)
        
        # Store references to system components
        self.portfolio_manager = portfolio_manager
        self.data_collectors = data_collectors or {}
        self.model_manager = model_manager
        self.trade_manager = trade_manager
        
        # Initialize metrics
        self._init_portfolio_metrics()
        self._init_trading_metrics()
        self._init_model_metrics()
        self._init_data_metrics()
        self._init_system_metrics()
        
        # Last update timestamps
        self._last_update_times = {}
        
        logger.info("Initialized trading metrics exporter")
    
    def _init_portfolio_metrics(self) -> None:
        """Initialize portfolio performance metrics."""
        # Portfolio value metrics
        self.create_gauge(
            name="portfolio_total_value",
            description="Total portfolio value in USD"
        )
        
        self.create_gauge(
            name="portfolio_cash_value",
            description="Cash component of portfolio in USD"
        )
        
        self.create_gauge(
            name="portfolio_asset_value",
            description="Asset component of portfolio in USD"
        )
        
        self.create_gauge(
            name="portfolio_daily_return",
            description="Daily portfolio return as a percentage"
        )
        
        self.create_gauge(
            name="portfolio_intraday_return",
            description="Intraday portfolio return as a percentage"
        )
        
        # Individual position metrics
        self.create_gauge(
            name="position_value",
            description="Value of position in USD",
            labels=["symbol"]
        )
        
        self.create_gauge(
            name="position_unrealized_pnl",
            description="Unrealized profit/loss of position in USD",
            labels=["symbol"]
        )
        
        self.create_gauge(
            name="position_weight",
            description="Position weight as percentage of portfolio",
            labels=["symbol"]
        )
        
        # Risk metrics
        self.create_gauge(
            name="portfolio_volatility",
            description="Portfolio volatility (30-day)"
        )
        
        self.create_gauge(
            name="portfolio_sharpe_ratio",
            description="Portfolio Sharpe ratio (30-day)"
        )
        
        self.create_gauge(
            name="portfolio_drawdown",
            description="Current portfolio drawdown as percentage"
        )
        
        self.create_gauge(
            name="portfolio_max_drawdown",
            description="Maximum portfolio drawdown (30-day) as percentage"
        )
        
        self.create_gauge(
            name="portfolio_beta",
            description="Portfolio beta to market"
        )
    
    def _init_trading_metrics(self) -> None:
        """Initialize trading activity metrics."""
        # Order metrics
        self.create_counter(
            name="orders_submitted_total",
            description="Total number of orders submitted",
            labels=["side", "type"]
        )
        
        self.create_counter(
            name="orders_filled_total",
            description="Total number of orders filled",
            labels=["side", "type"]
        )
        
        self.create_counter(
            name="orders_cancelled_total",
            description="Total number of orders cancelled",
            labels=["side", "type"]
        )
        
        self.create_counter(
            name="orders_rejected_total",
            description="Total number of orders rejected",
            labels=["side", "type", "reason"]
        )
        
        self.create_gauge(
            name="orders_open_count",
            description="Number of currently open orders",
            labels=["side", "type"]
        )
        
        # Trade metrics
        self.create_counter(
            name="trades_total",
            description="Total number of trades executed",
            labels=["side"]
        )
        
        self.create_counter(
            name="trade_volume_total",
            description="Total trading volume in USD",
            labels=["side"]
        )
        
        self.create_gauge(
            name="trade_win_rate",
            description="Win rate of closed trades (percentage)"
        )
        
        self.create_gauge(
            name="trade_avg_profit",
            description="Average profit of winning trades in USD"
        )
        
        self.create_gauge(
            name="trade_avg_loss",
            description="Average loss of losing trades in USD"
        )
        
        self.create_histogram(
            name="trade_duration_seconds",
            description="Distribution of trade durations in seconds",
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400]
        )
        
        # Execution quality metrics
        self.create_gauge(
            name="execution_slippage_bps",
            description="Average execution slippage in basis points",
            labels=["side"]
        )
        
        self.create_histogram(
            name="execution_latency_ms",
            description="Distribution of execution latencies in milliseconds",
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
    

        # Circuit breaker metrics
        self.create_counter(
            name="circuit_breaker_triggered",
            description="Total number of times a circuit breaker was triggered",
            labels=["type", "trigger"]
        )
        
        self.create_gauge(
            name="trading_halted",
            description="Indicates if trading is currently halted (1 = halted, 0 = active)",
            labels=["scope"]
        )

    def _init_model_metrics(self) -> None:
        """Initialize model performance metrics."""
        # Model prediction metrics
        model_types = [
            "stock_selection", "entry_timing", "peak_detection", 
            "risk_sizing", "market_regime"
        ]
        
        for model_type in model_types:
            self.create_counter(
                name=f"{model_type}_predictions_total",
                description=f"Total number of {model_type} model predictions"
            )
            
            self.create_gauge(
                name=f"{model_type}_prediction_latency_ms",
                description=f"Latency of {model_type} model predictions in milliseconds"
            )
            
            self.create_gauge(
                name=f"{model_type}_prediction_accuracy",
                description=f"Accuracy of {model_type} model predictions"
            )
        
        # Strategy metrics
        self.create_counter(
            name="strategy_signals_total",
            description="Total number of strategy signals generated",
            labels=["signal_type", "direction"]
        )
        
        self.create_gauge(
            name="strategy_signal_quality",
            description="Quality score of strategy signals (0-1)",
            labels=["signal_type"]
        )
    
    def _init_data_metrics(self) -> None:
        """Initialize data quality and collection metrics."""
        # Data collection metrics
        self.create_counter(
            name="data_collection_attempts_total",
            description="Total number of data collection attempts",
            labels=["collector", "data_type"]
        )
        
        self.create_counter(
            name="data_collection_successes_total",
            description="Total number of successful data collections",
            labels=["collector", "data_type"]
        )
        
        self.create_counter(
            name="data_collection_failures_total",
            description="Total number of failed data collections",
            labels=["collector", "data_type", "reason"]
        )
        
        self.create_gauge(
            name="data_collection_latency_ms",
            description="Latency of data collection in milliseconds",
            labels=["collector", "data_type"]
        )
        
        # Data quality metrics
        self.create_gauge(
            name="data_quality_completeness",
            description="Completeness of data as percentage",
            labels=["data_type"]
        )
        
        self.create_gauge(
            name="data_quality_timeliness_ms",
            description="Timeliness of data in milliseconds (lower is better)",
            labels=["data_type"]
        )
        
        self.create_counter(
            name="data_anomalies_total",
            description="Total number of detected data anomalies",
            labels=["data_type", "anomaly_type"]
        )
    
    def _init_system_metrics(self) -> None:
        """Initialize system health metrics."""
        # Component health metrics
        self.create_gauge(
            name="component_health",
            description="Health status of system components (0=down, 1=degraded, 2=healthy)",
            labels=["component"]
        )
        
        self.create_gauge(
            name="component_last_heartbeat_seconds",
            description="Seconds since last component heartbeat",
            labels=["component"]
        )
        
        # Error tracking
        self.create_counter(
            name="errors_total",
            description="Total number of system errors",
            labels=["component", "severity", "error_type"]
        )
        
        # Performance metrics
        self.create_gauge(
            name="processing_time_ms",
            description="Processing time in milliseconds",
            labels=["operation"]
        )
        
        self.create_gauge(
            name="queue_size",
            description="Number of items in processing queues",
            labels=["queue"]
        )
    
    def collect(self) -> None:
        """Collect and update all trading metrics."""
        try:
            # Only collect metrics if components are available
            if self.portfolio_manager:
                self._collect_portfolio_metrics()
            
            if self.trade_manager:
                self._collect_trading_metrics()
            
            if self.model_manager:
                self._collect_model_metrics()
            
            if self.data_collectors:
                self._collect_data_metrics()
            
            # Always collect system metrics
            self._collect_system_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {str(e)}")
    
    def _collect_portfolio_metrics(self) -> None:
        """Collect portfolio performance metrics."""
        try:
            # Get current portfolio data
            portfolio = self.portfolio_manager.get_portfolio_snapshot()
            
            # Update portfolio value metrics
            self.get_metric("portfolio_total_value").set(portfolio.get("total_value", 0))
            self.get_metric("portfolio_cash_value").set(portfolio.get("cash", 0))
            self.get_metric("portfolio_asset_value").set(portfolio.get("assets_value", 0))
            
            # Update portfolio return metrics
            self.get_metric("portfolio_daily_return").set(portfolio.get("daily_return_pct", 0))
            self.get_metric("portfolio_intraday_return").set(portfolio.get("intraday_return_pct", 0))
            
            # Update position metrics
            position_metric = self.get_metric("position_value")
            pnl_metric = self.get_metric("position_unrealized_pnl")
            weight_metric = self.get_metric("position_weight")
            
            for position in portfolio.get("positions", []):
                symbol = position.get("symbol", "UNKNOWN")
                position_metric.labels(symbol=symbol).set(position.get("market_value", 0))
                pnl_metric.labels(symbol=symbol).set(position.get("unrealized_pnl", 0))
                weight_metric.labels(symbol=symbol).set(position.get("weight_pct", 0))
            
            # Update risk metrics
            self.get_metric("portfolio_volatility").set(portfolio.get("volatility_30d", 0))
            self.get_metric("portfolio_sharpe_ratio").set(portfolio.get("sharpe_ratio_30d", 0))
            self.get_metric("portfolio_drawdown").set(portfolio.get("current_drawdown_pct", 0))
            self.get_metric("portfolio_max_drawdown").set(portfolio.get("max_drawdown_30d_pct", 0))
            self.get_metric("portfolio_beta").set(portfolio.get("beta", 0))
            
        except Exception as e:
            logger.error(f"Error collecting portfolio metrics: {str(e)}")
    
    def _collect_trading_metrics(self) -> None:
        """Collect trading activity metrics."""
        try:
            # Get trading stats
            stats = self.trade_manager.get_trading_stats()
            
            # Update order metrics
            open_orders = stats.get("open_orders", {})
            for side in ["buy", "sell"]:
                for order_type in ["market", "limit", "stop", "stop_limit"]:
                    count = open_orders.get(f"{side}_{order_type}", 0)
                    self.get_metric("orders_open_count").labels(
                        side=side, type=order_type
                    ).set(count)
            
            # Update trade performance metrics
            self.get_metric("trade_win_rate").set(stats.get("win_rate_pct", 0))
            self.get_metric("trade_avg_profit").set(stats.get("avg_win_usd", 0))
            self.get_metric("trade_avg_loss").set(stats.get("avg_loss_usd", 0))
            
            # Update execution quality metrics
            for side in ["buy", "sell"]:
                slippage = stats.get(f"avg_slippage_{side}_bps", 0)
                self.get_metric("execution_slippage_bps").labels(side=side).set(slippage)
            
            # Get recent execution latencies
            latencies = stats.get("recent_execution_latencies_ms", [])
            latency_hist = self.get_metric("execution_latency_ms")
            for latency in latencies:
                latency_hist.observe(latency)
            
            # Get recent trade durations
            durations = stats.get("recent_trade_durations_seconds", [])
            duration_hist = self.get_metric("trade_duration_seconds")
            for duration in durations:
                duration_hist.observe(duration)
                
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {str(e)}")
    
    def _collect_model_metrics(self) -> None:
        """Collect model performance metrics."""
        try:
            # Get model performance stats
            model_stats = self.model_manager.get_model_performance_stats()
            
            # Update model metrics
            model_types = [
                "stock_selection", "entry_timing", "peak_detection", 
                "risk_sizing", "market_regime"
            ]
            
            for model_type in model_types:
                stats = model_stats.get(model_type, {})
                
                # Update accuracy
                accuracy = stats.get("accuracy", 0)
                self.get_metric(f"{model_type}_prediction_accuracy").set(accuracy)
                
                # Update latency
                latency = stats.get("avg_prediction_latency_ms", 0)
                self.get_metric(f"{model_type}_prediction_latency_ms").set(latency)
            
            # Update strategy signal metrics
            strategy_stats = model_stats.get("strategy", {})
            for signal_type, quality in strategy_stats.get("signal_quality", {}).items():
                self.get_metric("strategy_signal_quality").labels(
                    signal_type=signal_type
                ).set(quality)
                
        except Exception as e:
            logger.error(f"Error collecting model metrics: {str(e)}")
    
    def _collect_data_metrics(self) -> None:
        """Collect data quality and collection metrics."""
        try:
            # For each data collector
            for collector_name, collector in self.data_collectors.items():
                # Skip if collector doesn't have stats method
                if not hasattr(collector, "get_statistics") or not callable(collector.get_statistics):
                    continue
                
                # Get collector stats
                stats = collector.get_statistics()
                
                # Update data quality metrics
                for data_type, quality in stats.get("data_quality", {}).items():
                    completeness = quality.get("completeness_pct", 0)
                    timeliness = quality.get("timeliness_ms", 0)
                    
                    self.get_metric("data_quality_completeness").labels(
                        data_type=data_type
                    ).set(completeness)
                    
                    self.get_metric("data_quality_timeliness_ms").labels(
                        data_type=data_type
                    ).set(timeliness)
                
                # Update collection latency metrics
                for data_type, latency in stats.get("collection_latency", {}).items():
                    self.get_metric("data_collection_latency_ms").labels(
                        collector=collector_name,
                        data_type=data_type
                    ).set(latency)
                
                # Log any data anomalies detected since last check
                anomalies = stats.get("new_anomalies", {})
                anomaly_counter = self.get_metric("data_anomalies_total")
                
                for data_type, anomaly_types in anomalies.items():
                    for anomaly_type, count in anomaly_types.items():
                        if count > 0:
                            anomaly_counter.labels(
                                data_type=data_type,
                                anomaly_type=anomaly_type
                            ).inc(count)
                
        except Exception as e:
            logger.error(f"Error collecting data metrics: {str(e)}")
    
    def _collect_system_metrics(self) -> None:
        """Collect system health metrics."""
        try:
            # Get component health statuses
            components = {
                "portfolio_manager": self.portfolio_manager,
                "trade_manager": self.trade_manager,
                "model_manager": self.model_manager
            }
            
            # Add data collectors
            for name, collector in self.data_collectors.items():
                components[f"collector_{name}"] = collector
            
            # Update health metrics
            health_metric = self.get_metric("component_health")
            heartbeat_metric = self.get_metric("component_last_heartbeat_seconds")
            
            current_time = time.time()
            
            for name, component in components.items():
                if component is None:
                    continue
                
                # Skip if component doesn't have health check method
                if not hasattr(component, "get_health_status") or not callable(component.get_health_status):
                    continue
                
                try:
                    # Get health status
                    health = component.get_health_status()
                    
                    # Update health status
                    status_value = {
                        "down": 0,
                        "degraded": 1,
                        "healthy": 2
                    }.get(health.get("status", "down"), 0)
                    
                    health_metric.labels(component=name).set(status_value)
                    
                    # Update heartbeat
                    last_heartbeat = health.get("last_heartbeat_timestamp")
                    if last_heartbeat:
                        seconds_since = current_time - last_heartbeat
                        heartbeat_metric.labels(component=name).set(seconds_since)
                    
                except Exception as e:
                    logger.error(f"Error checking health of {name}: {str(e)}")
                    # Mark as down if check fails
                    health_metric.labels(component=name).set(0)
            
            # Update queue sizes
            if hasattr(self, "scheduler") and self.scheduler:
                queues = self.scheduler.get_queue_sizes()
                queue_metric = self.get_metric("queue_size")
                
                for queue_name, size in queues.items():
                    queue_metric.labels(queue=queue_name).set(size)
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    # Methods to increment counter metrics from outside the exporter
    
    def increment_order_submitted(self, side: str, order_type: str) -> None:
        """
        Increment the orders submitted counter.
        
        Args:
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
        """
        self.get_metric("orders_submitted_total").labels(
            side=side, type=order_type
        ).inc()
    
    def increment_order_filled(self, side: str, order_type: str) -> None:
        """
        Increment the orders filled counter.
        
        Args:
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
        """
        self.get_metric("orders_filled_total").labels(
            side=side, type=order_type
        ).inc()
    
    def increment_order_cancelled(self, side: str, order_type: str) -> None:
        """
        Increment the orders cancelled counter.
        
        Args:
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
        """
        self.get_metric("orders_cancelled_total").labels(
            side=side, type=order_type
        ).inc()
    
    def increment_order_rejected(self, side: str, order_type: str, reason: str) -> None:
        """
        Increment the orders rejected counter.
        
        Args:
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            reason: Rejection reason
        """
        self.get_metric("orders_rejected_total").labels(
            side=side, type=order_type, reason=reason
        ).inc()
    
    def increment_trade_executed(self, side: str, volume: float) -> None:
        """
        Increment trade counters.
        
        Args:
            side: Trade side ('buy' or 'sell')
            volume: Trade volume in USD
        """
        self.get_metric("trades_total").labels(side=side).inc()
        self.get_metric("trade_volume_total").labels(side=side).inc(volume)
    
    def record_trade_duration(self, duration_seconds: float) -> None:
        """
        Record a trade duration.
        
        Args:
            duration_seconds: Duration of the trade in seconds
        """
        self.get_metric("trade_duration_seconds").observe(duration_seconds)
    
    def record_execution_latency(self, latency_ms: float) -> None:
        """
        Record an execution latency.
        
        Args:
            latency_ms: Execution latency in milliseconds
        """
        self.get_metric("execution_latency_ms").observe(latency_ms)
    
    def increment_strategy_signal(self, signal_type: str, direction: str) -> None:
        """
        Increment the strategy signals counter.
        
        Args:
            signal_type: Type of signal
            direction: Signal direction ('long', 'short', 'neutral')
        """
        self.get_metric("strategy_signals_total").labels(
            signal_type=signal_type, direction=direction
        ).inc()
    
    def increment_data_collection_attempt(self, collector: str, data_type: str) -> None:
        """
        Increment the data collection attempts counter.
        
        Args:
            collector: Name of the data collector
            data_type: Type of data being collected
        """
        self.get_metric("data_collection_attempts_total").labels(
            collector=collector, data_type=data_type
        ).inc()
    
    def increment_data_collection_success(self, collector: str, data_type: str) -> None:
        """
        Increment the data collection successes counter.
        
        Args:
            collector: Name of the data collector
            data_type: Type of data being collected
        """
        self.get_metric("data_collection_successes_total").labels(
            collector=collector, data_type=data_type
        ).inc()
    
    def increment_data_collection_failure(self, collector: str, data_type: str, reason: str) -> None:
        """
        Increment the data collection failures counter.
        
        Args:
            collector: Name of the data collector
            data_type: Type of data being collected
            reason: Failure reason
        """
        self.get_metric("data_collection_failures_total").labels(
            collector=collector, data_type=data_type, reason=reason
        ).inc()
    
    def increment_data_anomaly(self, data_type: str, anomaly_type: str) -> None:
        """
        Increment the data anomalies counter.
        
        Args:
            data_type: Type of data
            anomaly_type: Type of anomaly detected
        """
        self.get_metric("data_anomalies_total").labels(
            data_type=data_type, anomaly_type=anomaly_type
        ).inc()
    
    def increment_error(self, component: str, severity: str, error_type: str) -> None:
        """
        Increment the errors counter.
        
        Args:
            component: Component where the error occurred
            severity: Error severity ('critical', 'error', 'warning')
            error_type: Type of error
        """
        self.get_metric("errors_total").labels(
            component=component, severity=severity, error_type=error_type
        ).inc()
    
    def record_processing_time(self, operation: str, time_ms: float) -> None:
        """
        Record a processing time.
        
        Args:
            operation: Name of the operation
            time_ms: Processing time in milliseconds
        """
        self.get_metric("processing_time_ms").labels(
            operation=operation
        ).set(time_ms)
