#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Manager
-----------
Manages risk limits, executes risk controls, and coordinates risk workflows.
Acts as central risk management hub for the trading system.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
import uuid
from enum import Enum
from collections import deque

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
# Avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portfolio.risk_calculator import RiskCalculator as RiskCalculatorType
    from portfolio.diversification_engine import DiversificationEngine
else:
    # Import non-circular dependencies
    pass
from portfolio.position_tracker import PositionTracker
from portfolio.performance_monitor import PerformanceMonitor

# Setup logging
logger = setup_logger('risk_manager')


class RiskStatus(Enum):
    """Risk status levels"""
    NORMAL = 'normal'
    ELEVATED = 'elevated'
    HIGH = 'high'
    EXTREME = 'extreme'
    SUSPENDED = 'suspended'


class RiskControl(Enum):
    """Risk control actions"""
    REDUCE_POSITION = 'reduce_position'
    HEDGE_PORTFOLIO = 'hedge_portfolio'
    ADD_STOP_LOSS = 'add_stop_loss'
    SUSPEND_TRADING = 'suspend_trading'
    INCREASE_CASH = 'increase_cash'
    REBALANCE = 'rebalance'
    CLOSE_POSITIONS = 'close_positions'


class RiskAlert:
    """Risk alert representation"""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
               data: Optional[Dict[str, Any]] = None, timestamp: Optional[datetime] = None):
        """
        Initialize a risk alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            data: Optional alert data
            timestamp: Timestamp (defaults to now)
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
        self.id = str(uuid.uuid4())
        self.acknowledged = False
        self.resolved = False
        self.resolution_time = None
        self.resolution_note = None
    
    def acknowledge(self):
        """Acknowledge the alert."""
        self.acknowledged = True
    
    def resolve(self, note: Optional[str] = None):
        """
        Resolve the alert.
        
        Args:
            note: Optional resolution note
        """
        self.resolved = True
        self.resolution_time = datetime.now()
        self.resolution_note = note
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert alert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None,
            'resolution_note': self.resolution_note
        }


class RiskLimits:
    """Risk limits configuration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk limits.
        
        Args:
            config: Optional configuration dictionary
        """
        config = config or {}
        
        # Portfolio-level limits
        self.max_portfolio_var = config.get('max_portfolio_var', 0.05)  # 5% VaR
        self.max_portfolio_volatility = config.get('max_portfolio_volatility', 0.25)  # 25% annualized vol
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% max drawdown
        self.min_sharpe_ratio = config.get('min_sharpe_ratio', 0.5)  # 0.5 min Sharpe
        
        # Position-level limits
        self.max_position_size = config.get('max_position_size', 0.10)  # 10% max position
        self.max_position_var = config.get('max_position_var', 0.01)  # 1% VaR per position
        
        # Exposure limits
        self.max_sector_exposure = config.get('max_sector_exposure', 0.30)  # 30% sector exposure
        self.max_single_name_exposure = config.get('max_single_name_exposure', 0.10)  # 10% to single name
        self.min_cash_allocation = config.get('min_cash_allocation', 0.05)  # 5% min cash
        
        # Correlation limits
        self.max_avg_correlation = config.get('max_avg_correlation', 0.50)  # Maximum average correlation
        self.max_pair_correlation = config.get('max_pair_correlation', 0.80)  # Maximum pair correlation
        
        # Trading limits
        self.max_daily_trading_volume = config.get('max_daily_trading_volume', 0.20)  # 20% of account per day
        self.max_order_count_per_day = config.get('max_order_count_per_day', 100)  # Max 100 orders per day
        
        # Custom limits
        self.custom_limits = config.get('custom_limits', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert limits to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'portfolio_limits': {
                'max_portfolio_var': self.max_portfolio_var,
                'max_portfolio_volatility': self.max_portfolio_volatility,
                'max_drawdown': self.max_drawdown,
                'min_sharpe_ratio': self.min_sharpe_ratio
            },
            'position_limits': {
                'max_position_size': self.max_position_size,
                'max_position_var': self.max_position_var
            },
            'exposure_limits': {
                'max_sector_exposure': self.max_sector_exposure,
                'max_single_name_exposure': self.max_single_name_exposure,
                'min_cash_allocation': self.min_cash_allocation
            },
            'correlation_limits': {
                'max_avg_correlation': self.max_avg_correlation,
                'max_pair_correlation': self.max_pair_correlation
            },
            'trading_limits': {
                'max_daily_trading_volume': self.max_daily_trading_volume,
                'max_order_count_per_day': self.max_order_count_per_day
            },
            'custom_limits': self.custom_limits
        }


class RiskCalculator:
    """
    Calculates risk metrics for portfolios and positions.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize risk calculator.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        logger.info("Risk calculator initialized")
    
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
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def calculate_portfolio_metrics(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate basic portfolio metrics.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with portfolio metrics
        """
        total_value = sum(p.get('market_value', 0) for p in positions)
        total_cost = sum(p.get('cost_basis', 0) for p in positions)
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'unrealized_pnl': total_value - total_cost,
            'unrealized_pnl_pct': (total_value / total_cost - 1) if total_cost > 0 else 0,
            'position_count': len(positions)
        }
    
    def calculate_portfolio_var(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with risk metrics
        """
        # Simple implementation - in a real system this would use historical data
        # and proper VaR calculation methods (historical simulation, parametric, Monte Carlo)
        total_value = sum(p.get('market_value', 0) for p in positions)
        
        # Assume average volatility of 2% daily for simplicity
        avg_volatility = 0.02
        
        # Calculate portfolio VaR (95% confidence, 1-day)
        # Using simplified parametric approach: VaR = Z * σ * √T * P
        # where Z is Z-score for confidence level, σ is volatility, T is time horizon, P is portfolio value
        z_score = 1.645  # 95% confidence
        portfolio_var = z_score * avg_volatility * total_value
        
        return {
            'portfolio_var': portfolio_var,
            'portfolio_var_pct': portfolio_var / total_value if total_value > 0 else 0,
            'portfolio_volatility': avg_volatility,
            'expected_shortfall': portfolio_var * 1.2,  # Simplified ES calculation
            'confidence_level': 0.95,
            'time_horizon': 1  # 1 day
        }
    
    def calculate_stress_test(self, positions: List[Dict[str, Any]], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate impact of stress test scenario.
        
        Args:
            positions: List of position dictionaries
            scenario: Stress test scenario parameters
            
        Returns:
            Dictionary with stress test results
        """
        total_value = sum(p.get('market_value', 0) for p in positions)
        
        # Apply market move
        market_move = scenario.get('market_move', 0)
        market_impact = total_value * market_move
        
        # Apply sector-specific moves if available
        sector_moves = scenario.get('sector_moves', {})
        sector_impact = 0
        
        for position in positions:
            sector = position.get('sector', 'Unknown')
            if sector in sector_moves:
                sector_impact += position.get('market_value', 0) * sector_moves[sector]
        
        # Calculate total impact
        total_impact = market_impact + sector_impact
        total_impact_pct = total_impact / total_value if total_value > 0 else 0
        
        return {
            'scenario': scenario.get('name', 'Unnamed Scenario'),
            'total_impact': total_impact,
            'total_impact_pct': total_impact_pct,
            'market_impact': market_impact,
            'sector_impact': sector_impact,
            'post_stress_value': total_value + total_impact
        }


class RiskManager:
    """
    Manages trading risk and enforces risk controls.
    """
    def __init__(self, config_path: Optional[str] = None,
               position_tracker: Optional[PositionTracker] = None,
               risk_calculator: Optional["RiskCalculatorType"] = None,
               diversification_engine: Optional["DiversificationEngine"] = None,
               performance_monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize risk manager.
        
        Args:
            config_path: Optional path to configuration file
            position_tracker: Optional position tracker instance
            risk_calculator: Optional risk calculator instance
            diversification_engine: Optional diversification engine instance
            performance_monitor: Optional performance monitor instance
        """
        self.config = self._load_config(config_path)
        
        # Initialize risk limits
        self.risk_limits = RiskLimits(self.config.get('risk_limits', {}))
        
        # Initialize portfolio components or use provided ones
        self.position_tracker = position_tracker or PositionTracker(config_path)
        self.risk_calculator = risk_calculator or RiskCalculator(config_path)
        # Import DiversificationEngine here to avoid circular imports
        if diversification_engine is None:
            from portfolio.diversification_engine import DiversificationEngine
            self.diversification_engine = DiversificationEngine(self.risk_calculator, config_path)
        else:
            self.diversification_engine = diversification_engine
        self.performance_monitor = performance_monitor or PerformanceMonitor(self.position_tracker, config_path)
        
        # Risk status tracking
        self.risk_status = RiskStatus.NORMAL
        self.last_risk_status_change = datetime.now()
        self.status_history = []
        
        # Risk monitoring
        self.active_alerts = []
        self.resolved_alerts = []
        self.current_risk_metrics = {}
        self.risk_metrics_history = {}
        
        # Risk controls
        self.active_risk_controls = set()
        self.control_history = []
        
        # Trading volume tracking
        self.daily_trading_volume = 0.0
        self.daily_order_count = 0
        self.last_volume_reset = datetime.now().date()
        
        # Risk analysis state
        self.last_full_analysis = None
        self.analysis_frequency = self.config.get('analysis_frequency_seconds', 300)
        self.continuous_monitoring = self.config.get('continuous_monitoring', True)
        
        # Risk model settings
        self.var_confidence_level = self.config.get('var_confidence_level', 0.95)
        self.var_timeframe = self.config.get('var_timeframe', 1)  # 1-day VaR
        self.stress_test_scenarios = self.config.get('stress_test_scenarios', [
            {
                'name': 'Market Crash',
                'market_move': -0.15,  # 15% market drop
                'volatility_increase': 2.0,  # Volatility doubles
                'correlation_increase': 0.3  # Correlations increase by 0.3
            },
            {
                'name': 'Sector Rotation',
                'market_move': -0.05,  # 5% market drop
                'volatility_increase': 1.5,  # Volatility increases 50%
                'correlation_increase': 0.15,  # Correlations increase by 0.15
                'sector_moves': {
                    'Technology': -0.15,
                    'Energy': 0.10,
                    'Financials': -0.08,
                    'Healthcare': 0.05
                }
            },
            {
                'name': 'Volatility Spike',
                'market_move': -0.08,  # 8% market drop
                'volatility_increase': 3.0,  # Volatility triples
                'correlation_increase': 0.25  # Correlations increase by 0.25
            }
        ])
        
        # Load sector and industry mappings
        self._load_sector_mappings()
        
        # Thread for continuous monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # Start continuous monitoring if enabled
        if self.continuous_monitoring:
            self._start_monitoring()
        
        logger.info("Risk manager initialized")
    
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
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _load_sector_mappings(self):
        """Load sector and industry mappings."""
        try:
            # Load sector mappings
            sector_map_path = self.config.get('sector_mappings_path')
            if sector_map_path:
                with open(sector_map_path, 'r') as f:
                    sector_mappings = json.load(f)
                self.diversification_engine.load_sector_mappings(sector_mappings)
                logger.info(f"Loaded sector mappings for {len(sector_mappings)} symbols")
            
            # Load industry mappings
            industry_map_path = self.config.get('industry_mappings_path')
            if industry_map_path:
                with open(industry_map_path, 'r') as f:
                    industry_mappings = json.load(f)
                self.diversification_engine.load_industry_mappings(industry_mappings)
                logger.info(f"Loaded industry mappings for {len(industry_mappings)} symbols")
            
            # Load factor exposures
            factor_map_path = self.config.get('factor_exposures_path')
            if factor_map_path:
                with open(factor_map_path, 'r') as f:
                    factor_exposures = json.load(f)
                self.diversification_engine.load_factor_exposures(factor_exposures)
                logger.info(f"Loaded factor exposures for {len(factor_exposures)} symbols")
        
        except Exception as e:
            logger.error(f"Error loading mappings: {str(e)}")
    
    def _start_monitoring(self):
        """Start continuous risk monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.info("Monitoring thread already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info("Started continuous risk monitoring")
    
    def _stop_monitoring(self):
        """Stop continuous risk monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            logger.info("Monitoring thread not running")
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        logger.info("Stopped continuous risk monitoring")
    
    def _monitoring_loop(self):
        """Continuous risk monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Reset daily trading volume at the start of a new day
                today = datetime.now().date()
                if today != self.last_volume_reset:
                    self.daily_trading_volume = 0.0
                    self.daily_order_count = 0
                    self.last_volume_reset = today
                
                # Run risk analysis
                self.analyze_risk()
                
                # Check risk limits
                self.check_risk_limits()
                
                # Sleep for the analysis interval
                self._stop_monitoring.wait(self.analysis_frequency)
            
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {str(e)}")
                # Sleep a bit before trying again
                time.sleep(10)
    
    def analyze_risk(self) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Returns:
            Dictionary with risk analysis results
        """
        with self._lock:
            # Get current positions
            positions = self.position_tracker.get_all_positions()
            
            if not positions:
                logger.info("No positions to analyze")
                self.last_full_analysis = datetime.now()
                
                # Update risk status
                self._update_risk_status(RiskStatus.NORMAL)
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'positions': [],
                    'portfolio_metrics': {},
                    'risk_metrics': {},
                    'diversification_metrics': {},
                    'performance_metrics': {},
                    'alerts': [],
                    'status': self.risk_status.value
                }
            
            # Calculate portfolio metrics
            portfolio_metrics = self.risk_calculator.calculate_portfolio_metrics(positions)
            
            # Calculate portfolio risk metrics
            risk_metrics = self.risk_calculator.calculate_portfolio_var(positions)
            
            # Calculate diversification metrics
            diversification_metrics = self.diversification_engine.analyze_portfolio_diversity(positions)
            
            # Run correlation analysis
            correlation_metrics = self.diversification_engine.calculate_correlation_matrix(positions)
            
            # Calculate performance metrics
            performance_metrics = self.performance_monitor.get_performance_metrics()
            
            # Run stress tests
            stress_test_results = {}
            for scenario in self.stress_test_scenarios:
                scenario_name = scenario.get('name', 'Unnamed Scenario')
                stress_test_results[scenario_name] = self.risk_calculator.calculate_stress_test(
                    positions, scenario
                )
            
            # Combine all metrics
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'positions': [p for p in positions],
                'portfolio_metrics': portfolio_metrics,
                'risk_metrics': risk_metrics,
                'diversification_metrics': diversification_metrics,
                'correlation_metrics': correlation_metrics,
                'performance_metrics': performance_metrics,
                'stress_test_results': stress_test_results,
                'alerts': [alert.to_dict() for alert in self.active_alerts],
                'status': self.risk_status.value
            }
            
            # Store current metrics
            self.current_risk_metrics = {
                'portfolio_var': risk_metrics.get('portfolio_var_pct', 0.0),
                'portfolio_volatility': risk_metrics.get('portfolio_volatility', 0.0),
                'expected_shortfall': risk_metrics.get('expected_shortfall', 0.0),
                'herfindahl_index': diversification_metrics.get('herfindahl_index', 1.0),
                'effective_position_count': diversification_metrics.get('effective_position_count', 1.0),
                'avg_correlation': correlation_metrics.get('avg_correlation', 0.0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'current_drawdown': performance_metrics.get('current_drawdown', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in history (daily)
            today = datetime.now().date().isoformat()
            if today not in self.risk_metrics_history:
                self.risk_metrics_history[today] = []
            
            self.risk_metrics_history[today].append(self.current_risk_metrics)
            
            # Update last analysis time
            self.last_full_analysis = datetime.now()
            
            # Log analysis
            logger.info(f"Completed risk analysis: VAR={risk_metrics.get('portfolio_var_pct', 0.0):.2%}, " +
                      f"Volatility={risk_metrics.get('portfolio_volatility', 0.0):.2%}, " +
                      f"Effective Positions={diversification_metrics.get('effective_position_count', 1.0):.1f}")
            
            return analysis_results
    
    def check_risk_limits(self) -> List[RiskAlert]:
        """
        Check current metrics against risk limits.
        
        Returns:
            List of new risk alerts
        """
        with self._lock:
            # Get current positions
            positions = self.position_tracker.get_all_positions()
            
            if not positions:
                return []
            
            new_alerts = []
            
            # Check portfolio VaR limit
            portfolio_var = self.current_risk_metrics.get('portfolio_var', 0.0)
            if portfolio_var > self.risk_limits.max_portfolio_var:
                alert = RiskAlert(
                    alert_type='portfolio_var',
                    severity='high' if portfolio_var > self.risk_limits.max_portfolio_var * 1.5 else 'medium',
                    message=f"Portfolio VaR {portfolio_var:.2%} exceeds limit of {self.risk_limits.max_portfolio_var:.2%}",
                    data={'current_var': portfolio_var, 'limit': self.risk_limits.max_portfolio_var}
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check portfolio volatility limit
            portfolio_volatility = self.current_risk_metrics.get('portfolio_volatility', 0.0)
            if portfolio_volatility > self.risk_limits.max_portfolio_volatility:
                alert = RiskAlert(
                    alert_type='portfolio_volatility',
                    severity='high' if portfolio_volatility > self.risk_limits.max_portfolio_volatility * 1.5 else 'medium',
                    message=f"Portfolio volatility {portfolio_volatility:.2%} exceeds limit of {self.risk_limits.max_portfolio_volatility:.2%}",
                    data={'current_volatility': portfolio_volatility, 'limit': self.risk_limits.max_portfolio_volatility}
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check max drawdown limit
            current_drawdown = self.current_risk_metrics.get('current_drawdown', 0.0)
            if current_drawdown > self.risk_limits.max_drawdown:
                alert = RiskAlert(
                    alert_type='drawdown',
                    severity='high' if current_drawdown > self.risk_limits.max_drawdown * 1.5 else 'medium',
                    message=f"Current drawdown {current_drawdown:.2%} exceeds limit of {self.risk_limits.max_drawdown:.2%}",
                    data={'current_drawdown': current_drawdown, 'limit': self.risk_limits.max_drawdown}
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check Sharpe ratio limit
            sharpe_ratio = self.current_risk_metrics.get('sharpe_ratio', 0.0)
            if sharpe_ratio < self.risk_limits.min_sharpe_ratio:
                alert = RiskAlert(
                    alert_type='sharpe_ratio',
                    severity='medium',
                    message=f"Sharpe ratio {sharpe_ratio:.2f} below minimum of {self.risk_limits.min_sharpe_ratio:.2f}",
                    data={'current_sharpe': sharpe_ratio, 'limit': self.risk_limits.min_sharpe_ratio}
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check position size limits
            portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
            
            if portfolio_value > 0:
                for position in positions:
                    position_size = position.get('current_value', 0.0) / portfolio_value
                    
                    if position_size > self.risk_limits.max_position_size:
                        alert = RiskAlert(
                            alert_type='position_size',
                            severity='high' if position_size > self.risk_limits.max_position_size * 1.5 else 'medium',
                            message=f"Position {position.get('symbol')} size {position_size:.2%} exceeds limit of {self.risk_limits.max_position_size:.2%}",
                            data={
                                'symbol': position.get('symbol'),
                                'current_size': position_size,
                                'limit': self.risk_limits.max_position_size
                            }
                        )
                        self.active_alerts.append(alert)
                        new_alerts.append(alert)
                        logger.warning(f"Risk Alert: {alert.message}")
            
            # Check sector exposure limits
            sector_exposures = self.diversification_engine.current_sector_allocations
            
            for sector, exposure in sector_exposures.items():
                if exposure > self.risk_limits.max_sector_exposure:
                    alert = RiskAlert(
                        alert_type='sector_exposure',
                        severity='high' if exposure > self.risk_limits.max_sector_exposure * 1.5 else 'medium',
                        message=f"Sector {sector} exposure {exposure:.2%} exceeds limit of {self.risk_limits.max_sector_exposure:.2%}",
                        data={
                            'sector': sector,
                            'current_exposure': exposure,
                            'limit': self.risk_limits.max_sector_exposure
                        }
                    )
                    self.active_alerts.append(alert)
                    new_alerts.append(alert)
                    logger.warning(f"Risk Alert: {alert.message}")
            
            # Check correlation limits
            avg_correlation = self.current_risk_metrics.get('avg_correlation', 0.0)
            if avg_correlation > self.risk_limits.max_avg_correlation:
                alert = RiskAlert(
                    alert_type='avg_correlation',
                    severity='medium',
                    message=f"Average correlation {avg_correlation:.2f} exceeds limit of {self.risk_limits.max_avg_correlation:.2f}",
                    data={
                        'avg_correlation': avg_correlation,
                        'limit': self.risk_limits.max_avg_correlation
                    }
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check daily trading volume limit
            account_equity = portfolio_value + self.position_tracker.get_account_cash()
            
            if self.daily_trading_volume > account_equity * self.risk_limits.max_daily_trading_volume:
                alert = RiskAlert(
                    alert_type='trading_volume',
                    severity='medium',
                    message=f"Daily trading volume ${self.daily_trading_volume:.2f} exceeds {self.risk_limits.max_daily_trading_volume:.0%} of account",
                    data={
                        'daily_volume': self.daily_trading_volume,
                        'account_equity': account_equity,
                        'limit_pct': self.risk_limits.max_daily_trading_volume
                    }
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check daily order count limit
            if self.daily_order_count > self.risk_limits.max_order_count_per_day:
                alert = RiskAlert(
                    alert_type='order_count',
                    severity='medium',
                    message=f"Daily order count {self.daily_order_count} exceeds limit of {self.risk_limits.max_order_count_per_day}",
                    data={
                        'order_count': self.daily_order_count,
                        'limit': self.risk_limits.max_order_count_per_day
                    }
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Check cash allocation
            cash_allocation = 1.0 - sum(position.get('current_value', 0.0) / account_equity for position in positions)
            
            if cash_allocation < self.risk_limits.min_cash_allocation:
                alert = RiskAlert(
                    alert_type='cash_allocation',
                    severity='medium',
                    message=f"Cash allocation {cash_allocation:.2%} below minimum of {self.risk_limits.min_cash_allocation:.2%}",
                    data={
                        'cash_allocation': cash_allocation,
                        'limit': self.risk_limits.min_cash_allocation
                    }
                )
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                logger.warning(f"Risk Alert: {alert.message}")
            
            # Update risk status based on alerts
            self._update_risk_status_from_alerts()
            
            # If critical alerts, apply risk controls
            if self.risk_status in [RiskStatus.HIGH, RiskStatus.EXTREME]:
                self._apply_risk_controls()
            
            return new_alerts
    
    def _update_risk_status_from_alerts(self):
        """Update risk status based on current alerts."""
        # Count alerts by severity
        high_count = sum(1 for alert in self.active_alerts if alert.severity == 'high' and not alert.resolved)
        medium_count = sum(1 for alert in self.active_alerts if alert.severity == 'medium' and not alert.resolved)
        low_count = sum(1 for alert in self.active_alerts if alert.severity == 'low' and not alert.resolved)
        
        # Determine new status
        new_status = RiskStatus.NORMAL
        
        if high_count >= 3 or (high_count >= 1 and medium_count >= 3):
            new_status = RiskStatus.EXTREME
        elif high_count >= 1 or medium_count >= 3:
            new_status = RiskStatus.HIGH
        elif medium_count >= 1 or low_count >= 3:
            new_status = RiskStatus.ELEVATED
        
        # Manual suspension takes precedence
        if self.risk_status == RiskStatus.SUSPENDED:
            return
        
        # Update status if changed
        if new_status != self.risk_status:
            self._update_risk_status(new_status)
    
    def _update_risk_status(self, new_status: RiskStatus):
        """
        Update the risk status.
        
        Args:
            new_status: New risk status
        """
        if new_status != self.risk_status:
            old_status = self.risk_status
            self.risk_status = new_status
            self.last_risk_status_change = datetime.now()
            
            # Record in history
            self.status_history.append({
                'timestamp': self.last_risk_status_change.isoformat(),
                'old_status': old_status.value,
                'new_status': new_status.value
            })
            
            logger.warning(f"Risk status changed from {old_status.value} to {new_status.value}")
    
    def _execute_reduce_position(self):
        """
        Execute risk control to reduce position sizes.
        Reduces positions that exceed thresholds or have high risk metrics.
        """
        logger.info("Applying risk control: REDUCE_POSITION")
        
        # Get all positions
        positions = self.position_tracker.get_all_positions()
        if not positions:
            logger.warning("No positions to reduce")
            return
        
        # Get portfolio value
        portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
        if portfolio_value <= 0:
            logger.warning("Invalid portfolio value")
            return
        
        # Identify positions to reduce
        positions_to_reduce = []
        for position in positions:
            position_size = position.get('current_value', 0.0) / portfolio_value
            
            # Check if position exceeds maximum size
            if position_size > self.risk_limits.max_position_size:
                positions_to_reduce.append({
                    'symbol': position.get('symbol'),
                    'current_size': position_size,
                    'target_size': self.risk_limits.max_position_size,
                    'reduction_pct': (position_size - self.risk_limits.max_position_size) / position_size
                })
        
        # Execute position reductions
        for reduction in positions_to_reduce:
            symbol = reduction['symbol']
            reduction_pct = reduction['reduction_pct']
            
            # Calculate shares to sell
            position = next((p for p in positions if p.get('symbol') == symbol), None)
            if position:
                shares = position.get('quantity', 0)
                shares_to_sell = int(shares * reduction_pct)
                
                if shares_to_sell > 0:
                    logger.info(f"Reducing position {symbol} by {reduction_pct:.2%} ({shares_to_sell} shares)")
                    # In a real implementation, this would call the trading system to execute the order
                    # self.trading_system.create_order(symbol, 'sell', shares_to_sell, order_type='market')
        
        logger.info(f"Position reduction completed for {len(positions_to_reduce)} positions")

    def _execute_add_stop_loss(self):
        """
        Execute risk control to add or tighten stop-loss orders.
        Adds stop-loss orders to positions that don't have them or tightens existing ones.
        """
        logger.info("Applying risk control: ADD_STOP_LOSS")
        
        # Get all positions
        positions = self.position_tracker.get_all_positions()
        if not positions:
            logger.warning("No positions to add stop-loss orders to")
            return
        
        # Get current risk metrics
        volatility = self.current_risk_metrics.get('portfolio_volatility', 0.0)
        
        # Calculate stop-loss percentage based on volatility
        # Higher volatility = tighter stops
        base_stop_pct = 0.05  # 5% base stop-loss
        volatility_factor = min(3.0, max(1.0, volatility / 0.15))  # Scale based on volatility
        stop_pct = base_stop_pct / volatility_factor
        
        # Apply stop-loss orders to all positions
        for position in positions:
            symbol = position.get('symbol')
            current_price = position.get('current_price', 0.0)
            
            if current_price <= 0:
                continue
                
            # Calculate stop price
            stop_price = current_price * (1 - stop_pct)
            
            logger.info(f"Setting {stop_pct:.2%} stop-loss for {symbol} at {stop_price:.2f}")
            # In a real implementation, this would call the trading system to place the stop order
            # self.trading_system.create_order(symbol, 'sell', position.get('quantity', 0),
            #                                order_type='stop', stop_price=stop_price)
        
        logger.info(f"Stop-loss orders added/updated for {len(positions)} positions")

    def _execute_increase_cash(self):
        """
        Execute risk control to increase cash allocation.
        Sells positions to increase cash reserves to the minimum required level.
        """
        logger.info("Applying risk control: INCREASE_CASH")
        
        # Get account information
        positions = self.position_tracker.get_all_positions()
        account_cash = self.position_tracker.get_account_cash()
        portfolio_value = sum(position.get('current_value', 0.0) for position in positions) + account_cash
        
        # Calculate current cash allocation
        current_cash_pct = account_cash / portfolio_value if portfolio_value > 0 else 0
        
        # Check if cash allocation is below minimum
        if current_cash_pct >= self.risk_limits.min_cash_allocation:
            logger.info(f"Cash allocation ({current_cash_pct:.2%}) already meets minimum requirement ({self.risk_limits.min_cash_allocation:.2%})")
            return
        
        # Calculate how much to sell
        target_cash = portfolio_value * self.risk_limits.min_cash_allocation
        cash_to_raise = target_cash - account_cash
        
        logger.info(f"Need to raise ${cash_to_raise:.2f} to meet minimum cash allocation of {self.risk_limits.min_cash_allocation:.2%}")
        
        # Sort positions by risk (could use various metrics)
        # For simplicity, we'll use position size as a proxy for risk
        positions_by_risk = sorted(positions, key=lambda p: p.get('current_value', 0.0), reverse=True)
        
        # Sell positions to raise cash
        cash_raised = 0.0
        positions_sold = []
        
        for position in positions_by_risk:
            if cash_raised >= cash_to_raise:
                break
                
            symbol = position.get('symbol')
            position_value = position.get('current_value', 0.0)
            
            # Determine how much of this position to sell
            value_to_sell = min(position_value, cash_to_raise - cash_raised)
            sell_pct = value_to_sell / position_value
            
            # Calculate shares to sell
            shares = position.get('quantity', 0)
            shares_to_sell = int(shares * sell_pct)
            
            if shares_to_sell > 0:
                logger.info(f"Selling {shares_to_sell} shares of {symbol} to raise cash")
                # In a real implementation, this would call the trading system to execute the order
                # self.trading_system.create_order(symbol, 'sell', shares_to_sell, order_type='market')
                
                cash_raised += value_to_sell
                positions_sold.append(symbol)
        
        logger.info(f"Increased cash allocation by selling {len(positions_sold)} positions, raising ${cash_raised:.2f}")

    def _execute_suspend_trading(self):
        """
        Execute risk control to suspend trading.
        Halts all new order placement by setting a global trading suspension flag.
        """
        logger.info("Applying risk control: SUSPEND_TRADING")
        
        # Set risk status to SUSPENDED
        self._update_risk_status(RiskStatus.SUSPENDED)
        
        # Cancel all pending orders
        # In a real implementation, this would call the trading system to cancel orders
        # pending_orders = self.trading_system.get_pending_orders()
        # for order in pending_orders:
        #     self.trading_system.cancel_order(order.id)
        
        # Log suspension
        suspension_time = datetime.now()
        suspension_reason = "Risk limits exceeded - automatic trading suspension"
        
        logger.warning(f"TRADING SUSPENDED at {suspension_time.isoformat()}: {suspension_reason}")
        
        # In a real system, you might want to send notifications to administrators
        # self.notification_system.send_alert("Trading Suspended", suspension_reason, level="critical")
        
        # Record suspension in risk metrics
        self.current_risk_metrics['trading_suspended'] = True
        self.current_risk_metrics['suspension_time'] = suspension_time.isoformat()
        self.current_risk_metrics['suspension_reason'] = suspension_reason

    def _execute_hedge_portfolio(self):
        """
        Execute risk control to hedge the portfolio.
        Adds hedging positions (like short index futures or put options) to reduce market exposure.
        """
        logger.info("Applying risk control: HEDGE_PORTFOLIO")
        
        # Get portfolio beta if available
        portfolio_beta = self.current_risk_metrics.get('portfolio_beta', 1.0)
        
        # Get portfolio value
        positions = self.position_tracker.get_all_positions()
        portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
        
        if portfolio_value <= 0:
            logger.warning("Invalid portfolio value, cannot calculate hedge")
            return
        
        # Calculate hedge ratio based on portfolio risk
        # Higher portfolio volatility = higher hedge ratio
        base_hedge_ratio = 0.3  # Hedge 30% of portfolio by default
        volatility = self.current_risk_metrics.get('portfolio_volatility', 0.2)
        volatility_factor = min(2.0, max(0.5, volatility / 0.15))  # Scale based on volatility
        hedge_ratio = base_hedge_ratio * volatility_factor
        
        # Calculate hedge amount
        hedge_value = portfolio_value * hedge_ratio * portfolio_beta
        
        logger.info(f"Hedging {hedge_ratio:.2%} of portfolio (${hedge_value:.2f})")
        
        # Determine hedging instrument
        # Options: index futures, inverse ETFs, put options, etc.
        # For this example, we'll assume using S&P 500 futures (ES)
        
        # Calculate number of contracts
        # Assuming each ES contract is worth $50 * S&P 500 index
        # sp500_price = self.market_data_provider.get_price('SPX')
        sp500_price = 4000  # Example value
        contract_value = 50 * sp500_price
        num_contracts = int(hedge_value / contract_value)
        
        if num_contracts > 0:
            logger.info(f"Adding hedge: Short {num_contracts} S&P 500 futures contracts")
            # In a real implementation, this would call the trading system to execute the hedge
            # self.trading_system.create_futures_order('ES', 'sell', num_contracts)
        else:
            logger.warning("Hedge calculation resulted in zero contracts")

    def _execute_close_positions(self):
        """
        Execute risk control to close positions.
        Closes all or selected positions to reduce risk exposure.
        """
        logger.info("Applying risk control: CLOSE_POSITIONS")
        
        # Get all positions
        positions = self.position_tracker.get_all_positions()
        if not positions:
            logger.warning("No positions to close")
            return
        
        # Determine which positions to close
        # Options:
        # 1. Close all positions
        # 2. Close highest risk positions
        # 3. Close positions with largest losses
        
        # For this implementation, we'll close positions with the highest volatility
        # and largest drawdowns
        
        positions_to_close = []
        for position in positions:
            symbol = position.get('symbol')
            current_price = position.get('current_price', 0)
            avg_price = position.get('average_price', 0)
            
            # Skip if price data is invalid
            if current_price <= 0 or avg_price <= 0:
                continue
            
            # Calculate position metrics
            position_return = (current_price / avg_price) - 1
            position_volatility = position.get('volatility', 0.0)
            
            # Decision to close:
            # 1. Position is down more than 10%
            # 2. Position volatility is very high
            if position_return < -0.1 or position_volatility > 0.5:
                positions_to_close.append({
                    'symbol': symbol,
                    'quantity': position.get('quantity', 0),
                    'return': position_return,
                    'volatility': position_volatility
                })
        
        # Execute closing orders
        for position in positions_to_close:
            symbol = position['symbol']
            quantity = position['quantity']
            
            if quantity > 0:
                logger.info(f"Closing position: {symbol} ({quantity} shares), " +
                          f"Return: {position['return']:.2%}, Volatility: {position['volatility']:.2%}")
                # In a real implementation, this would call the trading system to execute the order
                # self.trading_system.create_order(symbol, 'sell', quantity, order_type='market')
        
        logger.info(f"Closed {len(positions_to_close)} high-risk positions")


    def _apply_risk_controls(self):
        """Apply automatic risk controls based on status."""
        # Define controls by risk status
        if self.risk_status == RiskStatus.HIGH:
            controls = {
                RiskControl.REDUCE_POSITION,
                RiskControl.ADD_STOP_LOSS,
                RiskControl.INCREASE_CASH
            }
        elif self.risk_status == RiskStatus.EXTREME:
            controls = {
                RiskControl.SUSPEND_TRADING,
                RiskControl.HEDGE_PORTFOLIO,
                RiskControl.CLOSE_POSITIONS
            }
        else:
            controls = set() # No controls for NORMAL or ELEVATED status by default
        # Apply the determined controls
        for control in controls:
            if control == RiskControl.REDUCE_POSITION:
                self._execute_reduce_position()
            elif control == RiskControl.ADD_STOP_LOSS:
                self._execute_add_stop_loss()
            elif control == RiskControl.INCREASE_CASH:
                self._execute_increase_cash()
            elif control == RiskControl.SUSPEND_TRADING:
                self._execute_suspend_trading()
            elif control == RiskControl.HEDGE_PORTFOLIO:
                self._execute_hedge_portfolio()
            elif control == RiskControl.CLOSE_POSITIONS:
                self._execute_close_positions()
            
            # Record the control application
            self.control_history.append({
                'timestamp': datetime.now().isoformat(),
                'control': control.value,
                'risk_status': self.risk_status.value
            })
            
            # Add to active controls set
            self.active_risk_controls.add(control)
