#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Manager
-------------
Coordinates and manages the entire trading system.
Handles position management, risk controls, and trading session lifecycle.
"""

import logging
import time
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
import schedule
import pytz
from functools import wraps

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from orchestration.workflow_manager import WorkflowManager
from orchestration.decision_framework import DecisionFramework
from trading.strategy import StrategyComposer, TradingStrategy
from trading.execution.order_router import OrderRouter, OrderRequest, OrderSide, OrderType, TimeInForce
from trading.execution.circuit_breaker import CircuitBreaker
from trading.execution.execution_monitor import ExecutionMonitor
from portfolio.position_tracker import PositionTracker
from portfolio.risk_calculator import RiskCalculator
from data.processors.data_cache import DataCache
from monitoring.prometheus.exporters.model_metrics_exporter import ModelMetricsExporter
from data.collectors.factory import CollectorFactory
from config.collector_config import CollectorConfig
from utils.config_loader import ConfigLoader

# Setup logging
logger = setup_logger('trading_manager')


# Trading session state
class TradingSessionState:
    """
    States for the trading session.
    """
    STOPPED = 'stopped'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    PAUSED = 'paused'
    ERROR = 'error'


# Function to synchronize method access
def synchronized(lock_name):
    """
    Decorator to synchronize method access.
    
    Args:
        lock_name: Name of the lock attribute
    """
    def decorator(method):
        @wraps(method)
        def synced_method(self, *args, **kwargs):
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kwargs)
        return synced_method
    return decorator


class TradingManager:
    """
    Manages the trading system.
    """
    
    def __init__(self,
                 circuit_breaker: CircuitBreaker,
                 execution_monitor: ExecutionMonitor,
                 order_router: OrderRouter,
                 position_tracker: PositionTracker,
                 performance_monitor: Any, # Use Any for now as PerformanceMonitor is not fully defined here
                 risk_manager: RiskCalculator,
                 diversification_engine: Any, # Use Any for now as DiversificationEngine is not fully defined here
                 error_handler: Any, # Use Any for now as ErrorHandler is not fully defined here
                 workflow_manager: WorkflowManager,
                 data_cache: DataCache,
                 model_metrics: Any, # Use Any for now as ModelMetricsExporter is not fully defined here
                 config_path: Optional[str] = None):
        """
        Initialize trading manager with pre-initialized components.
        
        Args:
            circuit_breaker: Initialized CircuitBreaker instance
            execution_monitor: Initialized ExecutionMonitor instance
            order_router: Initialized OrderRouter instance
            position_tracker: Initialized PositionTracker instance
            performance_monitor: Initialized PerformanceMonitor instance
            risk_manager: Initialized RiskCalculator instance
            diversification_engine: Initialized DiversificationEngine instance
            error_handler: Initialized ErrorHandler instance
            workflow_manager: Initialized WorkflowManager instance
            data_cache: Initialized DataCache instance
            model_metrics: Initialized ModelMetricsExporter instance
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Assign pre-initialized components
        self.circuit_breaker = circuit_breaker
        self.execution_monitor = execution_monitor
        self.order_router = order_router
        self.position_tracker = position_tracker
        self.performance_monitor = performance_monitor
        self.risk_calculator = risk_manager # Renamed from risk_manager to risk_calculator for consistency
        self.diversification_engine = diversification_engine
        self.error_handler = error_handler
        self.workflow_manager = workflow_manager
        self.data_cache = data_cache
        self.model_metrics = model_metrics
        
        # Initialize data feed client
        # Get the data client configuration from system_config.yaml
        data_client_config_system = self.config.get('data_client_config', {})
        collector_type = data_client_config_system.get('type', 'alpaca') # Default to alpaca
        
        # Load all collector configurations
        config_loader = ConfigLoader()
        collector_configs_all = config_loader.load_yaml('config/collector_config.yaml')

        # Get the specific configuration for the selected collector type
        specific_collector_config_dict = collector_configs_all.get(collector_type, {})
        
        # Create a CollectorConfig object from the specific configuration
        try:
            collector_config = CollectorConfig(specific_collector_config_dict) # Pass the dictionary directly
        except Exception as e:
            logger.error(f"Error creating CollectorConfig for {collector_type}: {str(e)}")
            raise # Re-raise the exception as we cannot proceed without a valid config
        
        # Create the data client using the factory
        self.data_client = CollectorFactory.create(collector_type, collector_config)
        
        # Initialize strategy composer and strategy (still needs config)
        self.strategy_composer = StrategyComposer(self.config.get('strategy_config'))
        self.strategy = TradingStrategy("default", self.config.get('strategy_config', {}))
        
        # Initialize broker client (using order_router as the broker interface)
        self.broker = self.order_router
        
        # Trading session parameters
        self.trading_days = self.config.get('trading_days', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        self.market_open_time = self.config.get('market_open_time', '09:30')
        self.market_close_time = self.config.get('market_close_time', '16:00')
        self.timezone = pytz.timezone(self.config.get('timezone', 'America/New_York'))
        
        # Scheduler
        self.scheduler = schedule.Scheduler()
        self._setup_scheduler()
        self.scheduler_thread = None
        self._stop_scheduler = threading.Event()
        
        # Processing thread
        self.processing_thread = None
        self._stop_processing = threading.Event()
        self.processing_interval = self.config.get('processing_interval', 60)  # 60 seconds
        
        # Locks for thread safety
        self.state_lock = threading.Lock()
        self.position_lock = threading.Lock()
        
        # Statistics
        self.session_stats = {
            'start_time': None,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'execution_errors': 0,
            'signals_processed': 0,
            'entry_signals': 0,
            'exit_signals': 0,
            'model_executions': 0
        }
        
        logger.info("Trading Manager initialized")
    
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
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _setup_scheduler(self):
        """
        Setup scheduled tasks.
        """
        # Schedule market open/close handling
        self.scheduler.every().day.at(self.market_open_time).do(self._handle_market_open)
        self.scheduler.every().day.at(self.market_close_time).do(self._handle_market_close)
        
        # Schedule regular tasks
        self.scheduler.every(10).minutes.do(self._run_risk_calculations)
        self.scheduler.every(30).minutes.do(self._update_market_data)
        self.scheduler.every().day.at("17:00").do(self._end_of_day_processing)
        
        logger.info("Scheduler setup complete")
    
    def _start_scheduler_thread(self):
        """
        Start the scheduler thread.
        """
        if self.scheduler_thread is not None and self.scheduler_thread.is_alive():
            return
        
        self._stop_scheduler.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Scheduler thread started")
    
    def _stop_scheduler_thread(self):
        """
        Stop the scheduler thread.
        """
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            return
        
        self._stop_scheduler.set()
        self.scheduler_thread.join(timeout=1.0)
        logger.info("Scheduler thread stopped")
    
    def _scheduler_loop(self):
        """
        Scheduler loop.
        """
        while not self._stop_scheduler.is_set():
            try:
                self.scheduler.run_pending()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
            
            # Sleep a short time
            time.sleep(1)
    
    def _start_processing_thread(self):
        """
        Start the processing thread.
        """
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        
        self._stop_processing.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Processing thread started")
    
    def _stop_processing_thread(self):
        """
        Stop the processing thread.
        """
        if self.processing_thread is None or not self.processing_thread.is_alive():
            return
        
        self._stop_processing.set()
        self.processing_thread.join(timeout=5.0)
        logger.info("Processing thread stopped")
    
    def _processing_loop(self):
        """
        Main processing loop.
        """
        while not self._stop_processing.is_set():
            try:
                if self.state == TradingSessionState.RUNNING:
                    self._execute_trading_cycle()
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                self._set_state(TradingSessionState.ERROR)
                self.last_error = str(e)
            
            # Sleep for the processing interval
            self._stop_processing.wait(self.processing_interval)
    
    @synchronized('state_lock')
    def _set_state(self, new_state: str):
        """
        Set the trading session state.
        
        Args:
            new_state: New state
        """
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.last_state_change = datetime.now()
            logger.info(f"Trading session state changed: {old_state} -> {new_state}")
    
    def _handle_market_open(self):
        """
        Handle market open event.
        """
        # Check if today is a trading day
        today = datetime.now(self.timezone).strftime('%A')
        if today not in self.trading_days:
            logger.info(f"Today ({today}) is not a trading day, skipping market open")
            return
        
        logger.info("Market open event")
        self.start_trading()
    
    def _handle_market_close(self):
        """
        Handle market close event.
        """
        logger.info("Market close event")
        self.stop_trading()
    
    def _run_risk_calculations(self):
        """
        Run risk calculations.
        """
        try:
            # Get current positions
            positions = self.position_tracker.get_all_positions()
            
            # Calculate portfolio risk
            risk_metrics = self.risk_calculator.calculate_portfolio_risk(positions)
            
            # Check if risk exceeds threshold
            max_risk = self.config.get('max_portfolio_risk', 0.05)  # 5% default
            
            if risk_metrics.get('portfolio_var', 0) > max_risk:
                logger.warning(f"Portfolio risk ({risk_metrics['portfolio_var']:.2%}) exceeds threshold ({max_risk:.2%})")
                
                # Reduce position sizes if needed
                if self.config.get('auto_risk_adjustment', True):
                    self._adjust_position_sizes(risk_metrics)
            
            logger.info(f"Risk calculations: VaR = {risk_metrics.get('portfolio_var', 0):.2%}, ES = {risk_metrics.get('expected_shortfall', 0):.2%}")
        
        except Exception as e:
            logger.error(f"Error running risk calculations: {str(e)}")
    
    def _adjust_position_sizes(self, risk_metrics: Dict[str, Any]):
        """
        Adjust position sizes based on risk metrics.
        
        Args:
            risk_metrics: Risk metrics dictionary
        """
        try:
            # Get position adjustments from risk calculator
            adjustments = self.risk_calculator.calculate_position_adjustments(risk_metrics)
            
            # Apply adjustments
            for symbol, adjustment in adjustments.items():
                if adjustment['action'] == 'reduce':
                    # Create order to reduce position
                    current_position = self.position_tracker.get_position(symbol)
                    if current_position is None:
                        continue
                    
                    # Calculate reduction quantity
                    reduction_qty = current_position['quantity'] * adjustment['reduction_pct']
                    
                    # Create order request
                    order_request = OrderRequest(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=reduction_qty,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"risk_reduce_{symbol}_{int(time.time())}"
                    )
                    
                    # Submit order
                    self.order_router.submit_order(order_request)
                    logger.info(f"Reduced position for {symbol} by {adjustment['reduction_pct']:.2%} due to risk limits")
            
            logger.info(f"Adjusted {len(adjustments)} positions for risk management")
        
        except Exception as e:
            logger.error(f"Error adjusting position sizes: {str(e)}")
    
    def _update_market_data(self):
        """
        Update market data.
        """
        try:
            # Get symbols to update
            positions = self.position_tracker.get_all_positions()
            watchlist = self.config.get('watchlist', [])
            
            symbols = list(set([p['symbol'] for p in positions] + watchlist))
            
            if not symbols:
                return
            
            logger.info(f"Updating market data for {len(symbols)} symbols")
            
            # Fetch market data
            # This would typically call your data collection module
            # For now, it's a placeholder
            
            # Update model state with new market data
            # This would update any model state that depends on latest market data
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def _end_of_day_processing(self):
        """
        End of day processing.
        """
        try:
            logger.info("Running end of day processing")
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Generate end of day reports
            reports = self._generate_reports()
            
            # Reset session statistics for next day
            if self.session_stats['start_time'] is not None:
                session_duration = datetime.now() - self.session_stats['start_time']
                logger.info(f"Trading session ended. Duration: {session_duration}, Trades: {self.session_stats['total_trades']}")
            
            self._reset_session_stats()
            
        except Exception as e:
            logger.error(f"Error in end of day processing: {str(e)}")
    
    def _update_performance_metrics(self):
        """
        Update performance metrics.
        """
        try:
            # Get all completed trades
            trades = self.position_tracker.get_closed_positions()
            
            # Calculate performance metrics
            win_count = sum(1 for t in trades if t.get('profit_pct', 0) > 0)
            loss_count = sum(1 for t in trades if t.get('profit_pct', 0) <= 0)
            
            win_rate = win_count / len(trades) if len(trades) > 0 else 0
            
            # Calculate average win and loss
            avg_win = np.mean([t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) > 0]) if win_count > 0 else 0
            avg_loss = np.mean([abs(t.get('profit_pct', 0)) for t in trades if t.get('profit_pct', 0) <= 0]) if loss_count > 0 else 0
            
            # Calculate profit factor
            profit_factor = sum(t.get('profit', 0) for t in trades if t.get('profit_pct', 0) > 0) / abs(sum(t.get('profit', 0) for t in trades if t.get('profit_pct', 0) <= 0)) if loss_count > 0 else float('inf')
            
            # Log metrics
            logger.info(f"Performance metrics - Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}, Avg win: {avg_win:.2%}, Avg loss: {avg_loss:.2%}")
            
            # Update strategy performance
            for name, strategy in self.strategy_composer.strategies.items():
                strategy_trades = [t for t in trades if t.get('strategy') == name]
                for trade in strategy_trades:
                    strategy.update_performance(trade)
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _generate_reports(self) -> Dict[str, Any]:
        """
        Generate end of day reports.
        
        Returns:
            Dictionary with report data
        """
        reports = {
            'timestamp': datetime.now().isoformat(),
            'performance': {},
            'positions': {},
            'risks': {},
            'statistics': {}
        }
        
        try:
            # Performance report
            reports['performance'] = {
                'daily_pnl': self.position_tracker.get_daily_pnl(),
                'open_positions': len(self.position_tracker.get_all_positions()),
                'closed_positions': len(self.position_tracker.get_closed_positions()),
                'session_stats': self.session_stats.copy()
            }
            
            # Positions report
            positions = self.position_tracker.get_all_positions()
            reports['positions'] = {
                'total_value': sum(p.get('current_value', 0) for p in positions),
                'position_count': len(positions),
                'positions': positions
            }
            
            # Risk report
            risk_metrics = self.risk_calculator.calculate_portfolio_risk(positions)
            reports['risks'] = risk_metrics
            
            # Statistics report
            reports['statistics'] = {
                'model_metrics': self.model_metrics.get_all_metrics(),
                'execution_metrics': self.execution_monitor.get_metrics()
            }
            
            logger.info("Generated end of day reports")
            
            # Save reports to file
            report_path = self.config.get('report_path', 'reports')
            os.makedirs(report_path, exist_ok=True)
            
            report_file = os.path.join(report_path, f"report_{datetime.now().strftime('%Y%m%d')}.json")
            with open(report_file, 'w') as f:
                json.dump(reports, f, indent=2)
            
            logger.info(f"Saved reports to {report_file}")
        
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
        
        return reports
    
    def _reset_session_stats(self):
        """
        Reset session statistics.
        """
        self.session_stats = {
            'start_time': None,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'execution_errors': 0,
            'signals_processed': 0,
            'entry_signals': 0,
            'exit_signals': 0,
            'model_executions': 0
        }
    
    def _fetch_latest_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch the latest market data for the given symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            
        Returns:
            DataFrame with market data
        """
        try:
            # Calculate time window - default to last trading day
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)  # Default to last day
            
            # Use data cache if available
            if hasattr(self.data_cache, 'get_latest'):
                cached_data = self.data_cache.get_latest(symbols)
                if cached_data is not None and not cached_data.empty:
                    logger.debug(f"Using cached data for {len(symbols)} symbols")
                    return cached_data
            
            # Collect data using the data client
            logger.debug(f"Collecting latest data for {len(symbols)} symbols")
            records = self.data_client.collect(start_time, end_time)
            
            # Filter for requested symbols
            symbol_set = set(symbols)
            filtered_records = [r for r in records if r.symbol in symbol_set]
            
            # Convert to DataFrame
            if not filtered_records:
                logger.warning(f"No data found for symbols: {symbols}")
                return pd.DataFrame()
            
            # Create DataFrame from records
            df = pd.DataFrame([r.__dict__ for r in filtered_records])
            
            # Cache the data if cache is available
            if hasattr(self.data_cache, 'update'):
                self.data_cache.update(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")
            return pd.DataFrame()
    
    def _execute_trading_cycle(self):
        """
        Execute a single trading cycle.
        """
        try:
            # Skip if circuit breaker is active
            if not self.circuit_breaker.allow_trading():
                logger.warning("Circuit breaker is active, skipping trading cycle")
                return
            
            # Get universe of symbols
            positions = self.position_tracker.get_all_positions()
            watchlist = self.config.get('watchlist', [])
            
            symbols = list(set([p['symbol'] for p in positions] + watchlist))
            
            if not symbols:
                logger.debug("No symbols to process")
                return
            
            # Execute workflow for symbols
            logger.debug(f"Executing trading cycle for {len(symbols)} symbols")
            
            # Fetch latest market data
            raw_data = self._fetch_latest_data(symbols)
            
            # Generate signals using strategy
            signals = self.strategy.generate_signals(raw_data)
            self.session_stats['signals_processed'] += len(signals)
            
            # Check circuit breaker with current metrics
            metrics = self.model_metrics.get_all_metrics()
            self.circuit_breaker.check(metrics)
            
            # Process each signal
            for sig in signals.itertuples():
                # Route order through order router
                order_result = self.order_router.route(sig._asdict())
                
                # Get execution report
                report = self.broker.get_order_status(order_result.get('order', {}).get('client_order_id'))
                
                # Record execution in monitor
                if report:
                    self.execution_monitor.record(sig._asdict(), report)
            
            # Also process signals through strategy composer for backward compatibility
            trading_decisions = self.strategy_composer.process_signals({
                'positions': signals.to_dict('records'),
                'market_regime': self.model_metrics.get_metric('market_regime')
            })
            
            # Record signal counts
            self.session_stats['entry_signals'] += len(trading_decisions.get('entries', []))
            self.session_stats['exit_signals'] += len(trading_decisions.get('exits', []))
            
            # Execute trading decisions
            execution_results = self.strategy_composer.execute_decisions(trading_decisions)
            
            # Update position tracker with execution results
            self._update_positions(execution_results)
            
            # Update statistics
            self.session_stats['total_trades'] += (
                len(execution_results.get('entries', {}).get('orders', [])) +
                len(execution_results.get('exits', {}).get('orders', []))
            )
            
            # Log summary
            logger.info(f"Trading cycle completed: {len(signals)} signals processed, {len(trading_decisions.get('entries', []))} entries, {len(trading_decisions.get('exits', []))} exits")
            
            # Sleep for the configured interval (handled by the processing thread)
        
        except Exception as e:
            logger.error(f"Error executing trading cycle: {str(e)}")
            self.session_stats['execution_errors'] += 1
    
    @synchronized('position_lock')
    def _update_positions(self, execution_results: Dict[str, Any]):
        """
        Update positions based on execution results.
        
        Args:
            execution_results: Execution results dictionary
        """
        try:
            # Process entry orders
            entry_orders = execution_results.get('entries', {}).get('orders', {})
            for order_id, order in entry_orders.items():
                if order.status.value in ['filled', 'partially_filled']:
                    # Add position
                    position_data = {
                        'symbol': order.request.symbol,
                        'quantity': order.filled_quantity,
                        'entry_price': order.average_price,
                        'entry_time': order.last_updated,
                        'strategy': order.request.client_order_id.split('_')[0] if '_' in order.request.client_order_id else 'unknown'
                    }
                    
                    self.position_tracker.add_position(position_data)
                    logger.info(f"Added position for {order.request.symbol}: {order.filled_quantity} shares at {order.average_price}")
            
            # Process exit orders
            exit_orders = execution_results.get('exits', {}).get('orders', {})
            for order_id, order in exit_orders.items():
                if order.status.value in ['filled', 'partially_filled']:
                    # Update position
                    self.position_tracker.update_position(
                        symbol=order.request.symbol,
                        quantity=order.filled_quantity,
                        exit_price=order.average_price,
                        exit_time=order.last_updated
                    )
                    logger.info(f"Updated position for {order.request.symbol}: closed {order.filled_quantity} shares at {order.average_price}")
        
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def start_trading(self) -> bool:
        """
        Start the trading session.
        
        Returns:
            Boolean indicating success
        """
        if self.state in [TradingSessionState.RUNNING, TradingSessionState.STARTING]:
            logger.warning("Trading session already running or starting")
            return False
        
        try:
            logger.info("Starting trading session")
            self._set_state(TradingSessionState.STARTING)
            
            # Reset session statistics
            self._reset_session_stats()
            self.session_stats['start_time'] = datetime.now()
            
            # Initialize components
            self._initialize_components()
            
            # Start threads
            self._start_scheduler_thread()
            self._start_processing_thread()
            
            self._set_state(TradingSessionState.RUNNING)
            logger.info("Trading session started")
            return True
        
        except Exception as e:
            logger.error(f"Error starting trading session: {str(e)}")
            self._set_state(TradingSessionState.ERROR)
            self.last_error = str(e)
            return False
    
    def _initialize_components(self):
        """
        Initialize trading components.
        """
        # Update account information
        account_info = self.order_router.get_account_balances().get('alpaca', {})
        
        # Initialize position tracker with current positions
        self._sync_positions()
        
        # Clear any stale data in cache
        self.data_cache.clear()
        
        # Update market data
        self._update_market_data()
        
        # Run initial risk calculations
        self._run_risk_calculations()
    
    def _sync_positions(self):
        """
        Synchronize positions with broker.
        """
        try:
            # Get positions from broker
            account_info = self.order_router.get_account_balances().get('alpaca', {})
            broker_positions = account_info.get('positions', {})
            
            # Synchronize with position tracker
            self.position_tracker.sync_positions(broker_positions)
            
            logger.info(f"Synchronized {len(broker_positions)} positions with broker")
        
        except Exception as e:
            logger.error(f"Error synchronizing positions: {str(e)}")
    
    def stop_trading(self) -> bool:
        """
        Stop the trading session.
        
        Returns:
            Boolean indicating success
        """
        if self.state not in [TradingSessionState.RUNNING, TradingSessionState.PAUSED]:
            logger.warning("Trading session not running or paused")
            return False
        
        try:
            logger.info("Stopping trading session")
            self._set_state(TradingSessionState.STOPPING)
            
            # Stop threads
            self._stop_processing_thread()
            
            # Run end of day processing
            self._end_of_day_processing()
            
            self._set_state(TradingSessionState.STOPPED)
            logger.info("Trading session stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping trading session: {str(e)}")
            self._set_state(TradingSessionState.ERROR)
            self.last_error = str(e)
            return False
    
    def pause_trading(self) -> bool:
        """
        Pause the trading session.
        
        Returns:
            Boolean indicating success
        """
        if self.state != TradingSessionState.RUNNING:
            logger.warning("Trading session not running")
            return False
        
        try:
            logger.info("Pausing trading session")
            self._set_state(TradingSessionState.PAUSED)
            
            # Stop processing thread but keep scheduler running
            self._stop_processing_thread()
            
            logger.info("Trading session paused")
            return True
        
        except Exception as e:
            logger.error(f"Error pausing trading session: {str(e)}")
            self._set_state(TradingSessionState.ERROR)
            self.last_error = str(e)
            return False
    
    def resume_trading(self) -> bool:
        """
        Resume the trading session.
        
        Returns:
            Boolean indicating success
        """
        if self.state != TradingSessionState.PAUSED:
            logger.warning("Trading session not paused")
            return False
        
        try:
            logger.info("Resuming trading session")
            
            # Restart processing thread
            self._start_processing_thread()
            
            self._set_state(TradingSessionState.RUNNING)
            logger.info("Trading session resumed")
            return True
        
        except Exception as e:
            logger.error(f"Error resuming trading session: {str(e)}")
            self._set_state(TradingSessionState.ERROR)
            self.last_error = str(e)
            return False
    
    def run_trading_session(self):
        """
        Run the trading session manually (blocking).
        
        This implements the pseudocode from the requirements:
        while True:
            raw_data = data_client.fetch_latest()
            signals = strategy.generate_signals(raw_data)
            for sig in signals.itertuples():
                circuit_breaker.check(metrics)
                order_router.route(sig)
                report = broker.get_report()
                execution_monitor.record(sig, report)
            sleep(config.loop_interval)
        """
        if self.state != TradingSessionState.RUNNING:
            logger.warning("Trading session not running")
            return False
            
        try:
            logger.info("Starting manual trading session run")
            
            while self.state == TradingSessionState.RUNNING:
                # Execute a single trading cycle
                self._execute_trading_cycle()
                
                # Sleep for the configured interval
                time.sleep(self.processing_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading session interrupted by user")
            self.stop_trading()
            
        except Exception as e:
            logger.error(f"Error in trading session run: {str(e)}")
            self._set_state(TradingSessionState.ERROR)
            self.last_error = str(e)
            return False
            
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get trading session status.
        
        Returns:
            Dictionary with session status
        """
        status = {
            'state': self.state,
            'last_state_change': self.last_state_change.isoformat(),
            'last_error': self.last_error,
            'session_stats': self.session_stats,
            'positions': {
                'count': len(self.position_tracker.get_all_positions()),
                'value': sum(p.get('current_value', 0) for p in self.position_tracker.get_all_positions())
            },
            'components': {
                'workflow_manager': self.workflow_manager.get_model_status(),
                'circuit_breaker': self.circuit_breaker.get_status(),
                'execution_monitor': self.execution_monitor.get_metrics()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return status


# Default trading manager instance
default_trading_manager = None


def get_trading_manager(config_path: Optional[str] = None) -> TradingManager:
    """
    Get or create the default trading manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        TradingManager instance
    """
    global default_trading_manager
    
    if default_trading_manager is None:
        default_trading_manager = TradingManager(config_path)
    
    return default_trading_manager


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Trading Manager')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--start', action='store_true', help='Start trading session')
    parser.add_argument('--stop', action='store_true', help='Stop trading session')
    parser.add_argument('--pause', action='store_true', help='Pause trading session')
    parser.add_argument('--resume', action='store_true', help='Resume trading session')
    parser.add_argument('--status', action='store_true', help='Get trading session status')
    
    args = parser.parse_args()
    
    # Create trading manager
    trading_manager = TradingManager(args.config)
    
    if args.start:
        # Start trading session
        success = trading_manager.start_trading()
        print(f"Started trading session: {success}")
    
    elif args.stop:
        # Stop trading session
        success = trading_manager.stop_trading()
        print(f"Stopped trading session: {success}")
    
    elif args.pause:
        # Pause trading session
        success = trading_manager.pause_trading()
        print(f"Paused trading session: {success}")
    
    elif args.resume:
        # Resume trading session
        success = trading_manager.resume_trading()
        print(f"Resumed trading session: {success}")
    
    elif args.status:
        # Get trading session status
        status = trading_manager.get_status()
        print(json.dumps(status, indent=2))
    
    else:
        parser.print_help()
