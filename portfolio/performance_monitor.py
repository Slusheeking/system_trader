#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Monitor
-----------------
Tracks portfolio performance metrics and provides analytics.
Calculates returns, drawdowns, alpha, beta, Sharpe ratio, and other performance metrics.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
from scipy import stats
import math
from collections import deque

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from portfolio.position_tracker import PositionTracker

# Setup logging
logger = setup_logger('performance_monitor')


class PerformanceMetrics:
    """
    Calculates and tracks performance metrics.
    """
    
    def __init__(self):
        """
        Initialize performance metrics.
        """
        # Performance time series
        self.equity_curve = {}  # Date to equity value
        self.daily_returns = {}  # Date to daily return
        self.drawdowns = {}  # Date to drawdown
        
        # Benchmark data
        self.benchmark_prices = {}  # Date to benchmark price
        self.benchmark_returns = {}  # Date to benchmark return
        
        # Calculated metrics
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.volatility = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.r_squared = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.calmar_ratio = 0.0
        
        # Performance periods
        self.mtd_return = 0.0  # Month to date
        self.qtd_return = 0.0  # Quarter to date
        self.ytd_return = 0.0  # Year to date
        self.one_month_return = 0.0
        self.three_month_return = 0.0
        self.six_month_return = 0.0
        self.one_year_return = 0.0
    
    def update_equity_value(self, date: str, value: float) -> None:
        """
        Update equity value for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            value: Equity value
        """
        self.equity_curve[date] = value
        
        # Sort equity curve by date
        self.equity_curve = dict(sorted(self.equity_curve.items()))
        
        # Calculate daily returns
        dates = list(self.equity_curve.keys())
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_value = self.equity_curve[prev_date]
            curr_value = self.equity_curve[curr_date]
            
            daily_return = (curr_value / prev_value) - 1
            self.daily_returns[curr_date] = daily_return
        
        # Calculate drawdowns
        peak = 0.0
        for date, value in self.equity_curve.items():
            if value > peak:
                peak = value
                self.drawdowns[date] = 0.0
            else:
                drawdown = (peak - value) / peak
                self.drawdowns[date] = drawdown
        
        # Update current drawdown
        self.current_drawdown = list(self.drawdowns.values())[-1] if self.drawdowns else 0.0
        
        # Update max drawdown
        self.max_drawdown = max(self.drawdowns.values()) if self.drawdowns else 0.0
    
    def update_benchmark(self, date: str, price: float) -> None:
        """
        Update benchmark price for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            price: Benchmark price
        """
        self.benchmark_prices[date] = price
        
        # Sort benchmark prices by date
        self.benchmark_prices = dict(sorted(self.benchmark_prices.items()))
        
        # Calculate benchmark returns
        dates = list(self.benchmark_prices.keys())
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_price = self.benchmark_prices[prev_date]
            curr_price = self.benchmark_prices[curr_date]
            
            daily_return = (curr_price / prev_price) - 1
            self.benchmark_returns[curr_date] = daily_return
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return self._empty_metrics()
        
        # Get dates
        dates = list(self.equity_curve.keys())
        
        if len(dates) < 2:
            return self._empty_metrics()
        
        # Calculate total return
        first_value = self.equity_curve[dates[0]]
        last_value = self.equity_curve[dates[-1]]
        
        self.total_return = (last_value / first_value) - 1
        
        # Calculate annualized return
        days = (datetime.strptime(dates[-1], '%Y-%m-%d') - 
                datetime.strptime(dates[0], '%Y-%m-%d')).days
        
        if days > 0:
            self.annualized_return = ((1 + self.total_return) ** (365 / days)) - 1
        
        # Calculate volatility
        if len(self.daily_returns) > 1:
            returns_array = np.array(list(self.daily_returns.values()))
            self.volatility = returns_array.std() * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        if self.volatility > 0:
            self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.volatility
        
        # Calculate Sortino ratio (downside volatility)
        if len(self.daily_returns) > 1:
            returns_array = np.array(list(self.daily_returns.values()))
            negative_returns = returns_array[returns_array < 0]
            
            if len(negative_returns) > 0:
                downside_vol = negative_returns.std() * np.sqrt(252)  # Annualized
                
                if downside_vol > 0:
                    self.sortino_ratio = (self.annualized_return - risk_free_rate) / downside_vol
        
        # Calculate Calmar ratio
        if self.max_drawdown > 0:
            self.calmar_ratio = self.annualized_return / self.max_drawdown
        
        # Calculate alpha, beta, and r-squared
        if self.benchmark_returns and len(self.daily_returns) > 1:
            # Find common dates
            common_dates = sorted(set(self.daily_returns.keys()) & set(self.benchmark_returns.keys()))
            
            if len(common_dates) > 1:
                port_returns = [self.daily_returns[date] for date in common_dates]
                bench_returns = [self.benchmark_returns[date] for date in common_dates]
                
                # Calculate beta
                covariance = np.cov(port_returns, bench_returns)[0, 1]
                benchmark_variance = np.var(bench_returns)
                
                if benchmark_variance > 0:
                    self.beta = covariance / benchmark_variance
                
                # Calculate alpha
                port_mean = np.mean(port_returns) * 252  # Annualized
                bench_mean = np.mean(bench_returns) * 252  # Annualized
                
                self.alpha = port_mean - (risk_free_rate + self.beta * (bench_mean - risk_free_rate))
                
                # Calculate r-squared
                correlation = np.corrcoef(port_returns, bench_returns)[0, 1]
                self.r_squared = correlation ** 2
        
        # Calculate win rate and profit factor
        if self.daily_returns:
            wins = sum(1 for r in self.daily_returns.values() if r > 0)
            losses = sum(1 for r in self.daily_returns.values() if r < 0)
            
            total_trades = wins + losses
            
            if total_trades > 0:
                self.win_rate = wins / total_trades
            
            total_gains = sum(r for r in self.daily_returns.values() if r > 0)
            total_losses = abs(sum(r for r in self.daily_returns.values() if r < 0))
            
            if total_losses > 0:
                self.profit_factor = total_gains / total_losses
        
        # Calculate period returns
        today = datetime.now().date()
        month_start = datetime(today.year, today.month, 1).date()
        
        # Determine quarter start
        quarter = (today.month - 1) // 3 + 1
        quarter_start = datetime(today.year, 3 * quarter - 2, 1).date()
        
        year_start = datetime(today.year, 1, 1).date()
        
        # Calculate period returns
        self._calculate_period_returns(month_start.isoformat(), today.isoformat(), 'mtd_return')
        self._calculate_period_returns(quarter_start.isoformat(), today.isoformat(), 'qtd_return')
        self._calculate_period_returns(year_start.isoformat(), today.isoformat(), 'ytd_return')
        
        one_month_ago = (today - timedelta(days=30)).isoformat()
        three_months_ago = (today - timedelta(days=90)).isoformat()
        six_months_ago = (today - timedelta(days=180)).isoformat()
        one_year_ago = (today - timedelta(days=365)).isoformat()
        
        self._calculate_period_returns(one_month_ago, today.isoformat(), 'one_month_return')
        self._calculate_period_returns(three_months_ago, today.isoformat(), 'three_month_return')
        self._calculate_period_returns(six_months_ago, today.isoformat(), 'six_month_return')
        self._calculate_period_returns(one_year_ago, today.isoformat(), 'one_year_return')
        
        # Build metrics dictionary
        metrics = {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'period_returns': {
                'mtd_return': self.mtd_return,
                'qtd_return': self.qtd_return,
                'ytd_return': self.ytd_return,
                'one_month_return': self.one_month_return,
                'three_month_return': self.three_month_return,
                'six_month_return': self.six_month_return,
                'one_year_return': self.one_year_return
            },
            'last_update': dates[-1]
        }
        
        return metrics
    
    def _calculate_period_returns(self, start_date: str, end_date: str, attr_name: str) -> None:
        """
        Calculate return for a specific period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            attr_name: Attribute name to store result
        """
        dates = list(self.equity_curve.keys())
        
        # Find closest dates
        start_idx = 0
        for i, date in enumerate(dates):
            if date >= start_date:
                start_idx = i
                break
        
        end_idx = len(dates) - 1
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= end_date:
                end_idx = i
                break
        
        if start_idx <= end_idx and start_idx < len(dates) and end_idx < len(dates):
            start_value = self.equity_curve[dates[start_idx]]
            end_value = self.equity_curve[dates[end_idx]]
            
            period_return = (end_value / start_value) - 1
            setattr(self, attr_name, period_return)
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """
        Return empty metrics dictionary.
        
        Returns:
            Dictionary with default metrics
        """
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'r_squared': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'period_returns': {
                'mtd_return': 0.0,
                'qtd_return': 0.0,
                'ytd_return': 0.0,
                'one_month_return': 0.0,
                'three_month_return': 0.0,
                'six_month_return': 0.0,
                'one_year_return': 0.0
            },
            'last_update': None
        }


class PerformanceMonitor:
    """
    Monitors and analyzes portfolio performance.
    """
    
    def __init__(self, position_tracker: Optional[PositionTracker] = None, config_path: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            position_tracker: PositionTracker instance or None to create a new one
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Use provided position tracker or create a new one
        self.position_tracker = position_tracker
        
        # Performance settings
        self.benchmark_symbol = self.config.get('benchmark_symbol', 'SPY')
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.target_sharpe = self.config.get('target_sharpe', 1.0)
        self.target_drawdown = self.config.get('target_drawdown', 0.15)  # 15% max drawdown
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Performance history
        self.daily_metrics = {}  # Date to metrics dictionary
        self.monthly_metrics = {}  # Month to metrics dictionary
        
        # Trade metrics
        self.trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_holding_period': 0.0,
            'avg_win_holding_period': 0.0,
            'avg_loss_holding_period': 0.0
        }
        
        # Performance alerts
        self.drawdown_alerts = []
        self.sharpe_alerts = []
        self.return_alerts = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Performance monitor initialized")
    
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
    
    def update_portfolio_value(self, date: Optional[str] = None, value: Optional[float] = None) -> None:
        """
        Update portfolio value for a date.
        
        Args:
            date: Date string (YYYY-MM-DD) or None for today
            value: Portfolio value or None to use position tracker
        """
        with self.lock:
            # Use today's date if not provided
            if date is None:
                date = datetime.now().date().isoformat()
            
            # Use position tracker to get portfolio value if not provided
            if value is None and self.position_tracker is not None:
                value = self.position_tracker.get_portfolio_value()
            
            if value is not None:
                # Update equity curve
                self.metrics.update_equity_value(date, value)
                
                # Log update
                logger.debug(f"Updated portfolio value for {date}: {value}")
                
                # Calculate daily metrics
                self._update_daily_metrics(date)
                
                # Check for alerts
                self._check_alerts()
            else:
                logger.warning(f"No portfolio value provided or available for {date}")
    
    def update_benchmark(self, date: Optional[str] = None, price: float = None) -> None:
        """
        Update benchmark price for a date.
        
        Args:
            date: Date string (YYYY-MM-DD) or None for today
            price: Benchmark price
        """
        with self.lock:
            # Use today's date if not provided
            if date is None:
                date = datetime.now().date().isoformat()
            
            if price is not None:
                # Update benchmark price
                self.metrics.update_benchmark(date, price)
                
                # Log update
                logger.debug(f"Updated benchmark price for {date}: {price}")
            else:
                logger.warning(f"No benchmark price provided for {date}")
    
    def update_trade_metrics(self, closed_positions: List[Dict[str, Any]]) -> None:
        """
        Update trade metrics based on closed positions.
        
        Args:
            closed_positions: List of closed position dictionaries
        """
        with self.lock:
            if not closed_positions:
                return
            
            # Calculate trade metrics
            total_trades = len(closed_positions)
            winning_trades = sum(1 for p in closed_positions if p.get('profit', 0) > 0)
            losing_trades = total_trades - winning_trades
            
            # Calculate profit/loss metrics
            winning_profits = [p.get('profit', 0) for p in closed_positions if p.get('profit', 0) > 0]
            losing_profits = [p.get('profit', 0) for p in closed_positions if p.get('profit', 0) <= 0]
            
            avg_win = np.mean(winning_profits) if winning_profits else 0.0
            avg_loss = np.mean(losing_profits) if losing_profits else 0.0
            largest_win = max(winning_profits) if winning_profits else 0.0
            largest_loss = min(losing_profits) if losing_profits else 0.0
            
            # Calculate holding periods
            all_holding_periods = [p.get('holding_period_days', 0) for p in closed_positions]
            win_holding_periods = [p.get('holding_period_days', 0) for p in closed_positions if p.get('profit', 0) > 0]
            loss_holding_periods = [p.get('holding_period_days', 0) for p in closed_positions if p.get('profit', 0) <= 0]
            
            avg_holding_period = np.mean(all_holding_periods) if all_holding_periods else 0.0
            avg_win_holding_period = np.mean(win_holding_periods) if win_holding_periods else 0.0
            avg_loss_holding_period = np.mean(loss_holding_periods) if loss_holding_periods else 0.0
            
            # Update trade metrics
            self.trade_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_holding_period': avg_holding_period,
                'avg_win_holding_period': avg_win_holding_period,
                'avg_loss_holding_period': avg_loss_holding_period
            }
            
            # Log update
            logger.debug(f"Updated trade metrics for {total_trades} trades")
    
    def _update_daily_metrics(self, date: str) -> None:
        """
        Update daily metrics for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
        """
        # Calculate metrics
        metrics = self.metrics.calculate_metrics()
        
        # Store daily metrics
        self.daily_metrics[date] = metrics
        
        # Extract month
        month = date[:7]  # YYYY-MM
        
        # Update monthly metrics
        self.monthly_metrics[month] = metrics
    
    def _check_alerts(self) -> None:
        """
        Check for performance alerts.
        """
        metrics = self.metrics.calculate_metrics()
        
        # Check drawdown alert
        current_drawdown = metrics.get('current_drawdown', 0.0)
        
        if current_drawdown > self.target_drawdown:
            alert = {
                'type': 'drawdown',
                'value': current_drawdown,
                'threshold': self.target_drawdown,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if we already have a similar alert
            if not self.drawdown_alerts or self.drawdown_alerts[-1]['value'] < current_drawdown:
                self.drawdown_alerts.append(alert)
                logger.warning(f"Drawdown alert: {current_drawdown:.2%} exceeds target of {self.target_drawdown:.2%}")
        
        # Check Sharpe ratio alert
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        
        if sharpe_ratio < 0:
            alert = {
                'type': 'sharpe_negative',
                'value': sharpe_ratio,
                'threshold': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if we already have a similar alert
            if not self.sharpe_alerts or self.sharpe_alerts[-1]['value'] > sharpe_ratio:
                self.sharpe_alerts.append(alert)
                logger.warning(f"Sharpe ratio alert: Negative Sharpe ratio ({sharpe_ratio:.2f})")
        
        # Check return alert
        mtd_return = metrics.get('period_returns', {}).get('mtd_return', 0.0)
        
        if mtd_return < -0.05:  # 5% loss in a month
            alert = {
                'type': 'monthly_return',
                'value': mtd_return,
                'threshold': -0.05,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if we already have a similar alert
            if not self.return_alerts or self.return_alerts[-1]['value'] > mtd_return:
                self.return_alerts.append(alert)
                logger.warning(f"Monthly return alert: {mtd_return:.2%} is below threshold of -5%")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            metrics = self.metrics.calculate_metrics()
            
            # Add trade metrics
            metrics['trade_metrics'] = self.trade_metrics
            
            # Add alert counts
            metrics['alerts'] = {
                'drawdown_alerts': len(self.drawdown_alerts),
                'sharpe_alerts': len(self.sharpe_alerts),
                'return_alerts': len(self.return_alerts)
            }
            
            return metrics
    
    def get_equity_curve(self) -> Dict[str, float]:
        """
        Get equity curve data.
        
        Returns:
            Dictionary of date to equity value
        """
        with self.lock:
            return self.metrics.equity_curve
    
    def get_drawdown_curve(self) -> Dict[str, float]:
        """
        Get drawdown curve data.
        
        Returns:
            Dictionary of date to drawdown value
        """
        with self.lock:
            return self.metrics.drawdowns
    
    def get_returns_curve(self) -> Dict[str, float]:
        """
        Get daily returns curve data.
        
        Returns:
            Dictionary of date to daily return
        """
        with self.lock:
            return self.metrics.daily_returns
    
    def get_benchmark_comparison(self) -> Dict[str, Any]:
        """
        Get benchmark comparison data.
        
        Returns:
            Dictionary with benchmark comparison
        """
        with self.lock:
            # Find common dates
            common_dates = sorted(set(self.metrics.daily_returns.keys()) & set(self.metrics.benchmark_returns.keys()))
            
            if not common_dates:
                return {
                    'alpha': 0.0,
                    'beta': 0.0,
                    'r_squared': 0.0,
                    'tracking_error': 0.0,
                    'information_ratio': 0.0,
                    'correlation': 0.0,
                    'outperformance': 0.0,
                    'common_dates': []
                }
            
            # Calculate cumulative returns for portfolio and benchmark
            portfolio_cumulative = {}
            benchmark_cumulative = {}
            
            port_value = 1.0
            bench_value = 1.0
            
            for date in common_dates:
                port_value *= (1 + self.metrics.daily_returns[date])
                bench_value *= (1 + self.metrics.benchmark_returns[date])
                
                portfolio_cumulative[date] = port_value - 1  # Convert to return
                benchmark_cumulative[date] = bench_value - 1  # Convert to return
            
            # Calculate relative performance
            outperformance = portfolio_cumulative[common_dates[-1]] - benchmark_cumulative[common_dates[-1]]
            
            # Calculate tracking error
            tracking_differences = [self.metrics.daily_returns[date] - self.metrics.benchmark_returns[date] for date in common_dates]
            tracking_error = np.std(tracking_differences) * np.sqrt(252) if tracking_differences else 0.0
            
            # Calculate information ratio
            information_ratio = outperformance / tracking_error if tracking_error > 0 else 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(
                [self.metrics.daily_returns[date] for date in common_dates],
                [self.metrics.benchmark_returns[date] for date in common_dates]
            )[0, 1] if len(common_dates) > 1 else 0.0
            
            return {
                'alpha': self.metrics.alpha,
                'beta': self.metrics.beta,
                'r_squared': self.metrics.r_squared,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'correlation': correlation,
                'outperformance': outperformance,
                'portfolio_cumulative': portfolio_cumulative,
                'benchmark_cumulative': benchmark_cumulative,
                'common_dates': common_dates
            }
    
    def get_monthly_returns(self) -> Dict[str, Dict[str, float]]:
        """
        Get monthly returns organized by year and month.
        
        Returns:
            Dictionary with monthly returns
        """
        with self.lock:
            # Group returns by year and month
            monthly_returns = {}
            
            for date, return_value in self.metrics.daily_returns.items():
                year = date[:4]
                month = date[5:7]
                
                if year not in monthly_returns:
                    monthly_returns[year] = {m: 0.0 for m in [f"{i:02d}" for i in range(1, 13)]}
                
                monthly_returns[year][month] = (1 + monthly_returns[year][month]) * (1 + return_value) - 1
            
            return monthly_returns
    
    def get_rolling_metrics(self, window_days: int = 90) -> Dict[str, Dict[str, float]]:
        """
        Get rolling performance metrics.
        
        Args:
            window_days: Rolling window size in days
            
        Returns:
            Dictionary with rolling metrics
        """
        with self.lock:
            # Get dates
            dates = sorted(self.metrics.daily_returns.keys())
            
            if len(dates) <= window_days:
                return {
                    'rolling_return': {},
                    'rolling_volatility': {},
                    'rolling_sharpe': {},
                    'rolling_drawdown': {}
                }
            
            rolling_return = {}
            rolling_volatility = {}
            rolling_sharpe = {}
            rolling_drawdown = {}
            
            for i in range(window_days, len(dates)):
                end_date = dates[i]
                start_date = dates[i - window_days]
                
                # Calculate rolling return
                window_returns = [self.metrics.daily_returns[dates[j]] for j in range(i - window_days, i + 1)]
                
                cumulative_return = np.prod([1 + r for r in window_returns]) - 1
                annualized_return = ((1 + cumulative_return) ** (252 / len(window_returns))) - 1
                
                rolling_return[end_date] = annualized_return
                
                # Calculate rolling volatility
                rolling_volatility[end_date] = np.std(window_returns) * np.sqrt(252)
                
                # Calculate rolling Sharpe ratio
                if rolling_volatility[end_date] > 0:
                    rolling_sharpe[end_date] = (annualized_return - self.risk_free_rate) / rolling_volatility[end_date]
                else:
                    rolling_sharpe[end_date] = 0.0
                
                # Calculate rolling max drawdown
                window_equity = 1.0
                peak = 1.0
                max_drawdown = 0.0
                
                for r in window_returns:
                    window_equity *= (1 + r)
                    peak = max(peak, window_equity)
                    drawdown = (peak - window_equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                rolling_drawdown[end_date] = max_drawdown
            
            return {
                'rolling_return': rolling_return,
                'rolling_volatility': rolling_volatility,
                'rolling_sharpe': rolling_sharpe,
                'rolling_drawdown': rolling_drawdown
            }
    
    def get_performance_attribution(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get performance attribution by position and sector.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with performance attribution
        """
        with self.lock:
            if not positions:
                return {
                    'position_attribution': {},
                    'sector_attribution': {},
                    'total_return': 0.0
                }
            
            # Get total portfolio value
            portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
            
            if portfolio_value == 0:
                return {
                    'position_attribution': {},
                    'sector_attribution': {},
                    'total_return': 0.0
                }
            
            # Calculate attribution by position
            position_attribution = {}
            sector_attribution = {}
            
            for position in positions:
                symbol = position.get('symbol')
                weight = position.get('current_value', 0.0) / portfolio_value
                return_pct = position.get('unrealized_pnl_pct', 0.0) / 100  # Convert from percentage
                
                # Calculate position attribution
                attribution = weight * return_pct
                position_attribution[symbol] = {
                    'weight': weight,
                    'return': return_pct,
                    'attribution': attribution
                }
                
                # Calculate sector attribution
                sector = position.get('sector', 'Unknown')
                
                if sector not in sector_attribution:
                    sector_attribution[sector] = {
                        'weight': 0.0,
                        'return': 0.0,
                        'attribution': 0.0
                    }
                
                sector_attribution[sector]['weight'] += weight
                sector_attribution[sector]['attribution'] += attribution
            
            # Calculate sector returns
            for sector in sector_attribution:
                if sector_attribution[sector]['weight'] > 0:
                    sector_attribution[sector]['return'] = sector_attribution[sector]['attribution'] / sector_attribution[sector]['weight']
            
            # Calculate total return
            total_return = sum(attr['attribution'] for attr in position_attribution.values())
            
            return {
                'position_attribution': position_attribution,
                'sector_attribution': sector_attribution,
                'total_return': total_return
            }
    
    def analyze_trading_patterns(self) -> Dict[str, Any]:
        """
        Analyze trading patterns from closed positions.
        
        Returns:
            Dictionary with trading pattern analysis
        """
        with self.lock:
            if not self.position_tracker:
                return {
                    'error': 'No position tracker available',
                    'patterns': {}
                }
            
            # Get closed positions
            closed_positions = self.position_tracker.get_closed_positions()
            
            if not closed_positions:
                return {
                    'patterns': {},
                    'strategy_performance': {},
                    'symbols_performance': {}
                }
            
            # Analyze performance by strategy
            strategy_performance = {}
            
            for position in closed_positions:
                strategy = position.get('strategy', 'Unknown')
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'count': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_profit': 0.0,
                        'total_loss': 0.0,
                        'avg_holding_period': 0.0
                    }
                
                strategy_performance[strategy]['count'] += 1
                
                profit = position.get('profit', 0.0)
                
                if profit > 0:
                    strategy_performance[strategy]['wins'] += 1
                    strategy_performance[strategy]['total_profit'] += profit
                else:
                    strategy_performance[strategy]['losses'] += 1
                    strategy_performance[strategy]['total_loss'] += abs(profit)
                
                strategy_performance[strategy]['avg_holding_period'] += position.get('holding_period_days', 0.0)
            
            # Calculate averages
            for strategy in strategy_performance:
                count = strategy_performance[strategy]['count']
                if count > 0:
                    strategy_performance[strategy]['avg_holding_period'] /= count
                    strategy_performance[strategy]['win_rate'] = strategy_performance[strategy]['wins'] / count
                    
                    wins = strategy_performance[strategy]['wins']
                    losses = strategy_performance[strategy]['losses']
                    
                    if wins > 0:
                        strategy_performance[strategy]['avg_win'] = strategy_performance[strategy]['total_profit'] / wins
                    
                    if losses > 0:
                        strategy_performance[strategy]['avg_loss'] = strategy_performance[strategy]['total_loss'] / losses
                        
                    if strategy_performance[strategy]['total_loss'] > 0:
                        strategy_performance[strategy]['profit_factor'] = strategy_performance[strategy]['total_profit'] / strategy_performance[strategy]['total_loss']
                    else:
                        strategy_performance[strategy]['profit_factor'] = float('inf')
            
            # Analyze performance by symbol
            symbols_performance = {}
            
            for position in closed_positions:
                symbol = position.get('symbol', 'Unknown')
                
                if symbol not in symbols_performance:
                    symbols_performance[symbol] = {
                        'count': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_profit': 0.0,
                        'total_loss': 0.0,
                        'avg_holding_period': 0.0
                    }
                
                symbols_performance[symbol]['count'] += 1
                
                profit = position.get('profit', 0.0)
                
                if profit > 0:
                    symbols_performance[symbol]['wins'] += 1
                    symbols_performance[symbol]['total_profit'] += profit
                else:
                    symbols_performance[symbol]['losses'] += 1
                    symbols_performance[symbol]['total_loss'] += abs(profit)
                
                symbols_performance[symbol]['avg_holding_period'] += position.get('holding_period_days', 0.0)
            
            # Calculate averages for symbols
            for symbol in symbols_performance:
                count = symbols_performance[symbol]['count']
                if count > 0:
                    symbols_performance[symbol]['avg_holding_period'] /= count
                    symbols_performance[symbol]['win_rate'] = symbols_performance[symbol]['wins'] / count
                    
                    wins = symbols_performance[symbol]['wins']
                    losses = symbols_performance[symbol]['losses']
                    
                    if wins > 0:
                        symbols_performance[symbol]['avg_win'] = symbols_performance[symbol]['total_profit'] / wins
                    
                    if losses > 0:
                        symbols_performance[symbol]['avg_loss'] = symbols_performance[symbol]['total_loss'] / losses
                        
                    if symbols_performance[symbol]['total_loss'] > 0:
                        symbols_performance[symbol]['profit_factor'] = symbols_performance[symbol]['total_profit'] / symbols_performance[symbol]['total_loss']
                    else:
                        symbols_performance[symbol]['profit_factor'] = float('inf')
            
            # Analyze day of week patterns
            day_performance = {
                0: {'name': 'Monday', 'count': 0, 'wins': 0, 'total_profit': 0.0},
                1: {'name': 'Tuesday', 'count': 0, 'wins': 0, 'total_profit': 0.0},
                2: {'name': 'Wednesday', 'count': 0, 'wins': 0, 'total_profit': 0.0},
                3: {'name': 'Thursday', 'count': 0, 'wins': 0, 'total_profit': 0.0},
                4: {'name': 'Friday', 'count': 0, 'wins': 0, 'total_profit': 0.0}
            }
            
            for position in closed_positions:
                exit_time_str = position.get('exit_time')
                if exit_time_str:
                    try:
                        exit_time = datetime.fromisoformat(exit_time_str)
                        day_of_week = exit_time.weekday()
                        
                        if day_of_week < 5:  # Only weekdays
                            day_performance[day_of_week]['count'] += 1
                            
                            profit = position.get('profit', 0.0)
                            day_performance[day_of_week]['total_profit'] += profit
                            
                            if profit > 0:
                                day_performance[day_of_week]['wins'] += 1
                    except ValueError:
                        pass
            
            # Calculate win rate and average profit for each day
            for day, data in day_performance.items():
                if data['count'] > 0:
                    data['win_rate'] = data['wins'] / data['count']
                    data['avg_profit'] = data['total_profit'] / data['count']
                else:
                    data['win_rate'] = 0.0
                    data['avg_profit'] = 0.0
            
            # Analyze holding period patterns
            holding_periods = [position.get('holding_period_days', 0.0) for position in closed_positions]
            
            # Create bins for holding periods
            bins = {
                'intraday': {'count': 0, 'wins': 0, 'total_profit': 0.0},
                '1-3_days': {'count': 0, 'wins': 0, 'total_profit': 0.0},
                '4-7_days': {'count': 0, 'wins': 0, 'total_profit': 0.0},
                '8-14_days': {'count': 0, 'wins': 0, 'total_profit': 0.0},
                '15-30_days': {'count': 0, 'wins': 0, 'total_profit': 0.0},
                '30+_days': {'count': 0, 'wins': 0, 'total_profit': 0.0}
            }
            
            for position in closed_positions:
                holding_period = position.get('holding_period_days', 0.0)
                profit = position.get('profit', 0.0)
                
                bin_key = None
                if holding_period < 1:
                    bin_key = 'intraday'
                elif holding_period < 4:
                    bin_key = '1-3_days'
                elif holding_period < 8:
                    bin_key = '4-7_days'
                elif holding_period < 15:
                    bin_key = '8-14_days'
                elif holding_period < 31:
                    bin_key = '15-30_days'
                else:
                    bin_key = '30+_days'
                
                bins[bin_key]['count'] += 1
                bins[bin_key]['total_profit'] += profit
                
                if profit > 0:
                    bins[bin_key]['wins'] += 1
            
            # Calculate win rate and average profit for each bin
            for bin_key, data in bins.items():
                if data['count'] > 0:
                    data['win_rate'] = data['wins'] / data['count']
                    data['avg_profit'] = data['total_profit'] / data['count']
                else:
                    data['win_rate'] = 0.0
                    data['avg_profit'] = 0.0
            
            return {
                'patterns': {
                    'day_of_week': day_performance,
                    'holding_period': bins
                },
                'strategy_performance': strategy_performance,
                'symbols_performance': symbols_performance
            }
    
    def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """
        Calculate detailed drawdown metrics.
        
        Returns:
            Dictionary with drawdown metrics
        """
        with self.lock:
            if not self.metrics.drawdowns:
                return {
                    'max_drawdown': 0.0,
                    'current_drawdown': 0.0,
                    'avg_drawdown': 0.0,
                    'drawdown_periods': []
                }
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_date = None
            peak_date = None
            peak_value = 0.0
            max_drawdown = 0.0
            recovery_date = None
            
            dates = sorted(self.metrics.equity_curve.keys())
            
            for i, date in enumerate(dates):
                value = self.metrics.equity_curve[date]
                drawdown = self.metrics.drawdowns[date]
                
                if not in_drawdown and drawdown > 0.01:  # Start of drawdown > 1%
                    in_drawdown = True
                    start_date = date
                    
                    # Find peak date (look backward)
                    for j in range(i-1, -1, -1):
                        prev_date = dates[j]
                        prev_value = self.metrics.equity_curve[prev_date]
                        
                        if prev_value > peak_value:
                            peak_value = prev_value
                            peak_date = prev_date
                        
                        if self.metrics.drawdowns[prev_date] == 0:
                            break
                
                elif in_drawdown and drawdown < 0.01:  # End of drawdown
                    in_drawdown = False
                    recovery_date = date
                    
                    # Calculate drawdown metrics
                    if peak_date and start_date:
                        # Find valley date (minimum equity during drawdown)
                        valley_date = None
                        valley_value = float('inf')
                        
                        for j in range(dates.index(start_date), dates.index(recovery_date) + 1):
                            check_date = dates[j]
                            check_value = self.metrics.equity_curve[check_date]
                            
                            if check_value < valley_value:
                                valley_value = check_value
                                valley_date = check_date
                        
                        # Calculate drawdown amount
                        drawdown_amount = (peak_value - valley_value) / peak_value
                        
                        # Calculate recovery time
                        recovery_days = (datetime.strptime(recovery_date, '%Y-%m-%d') - 
                                      datetime.strptime(valley_date, '%Y-%m-%d')).days
                        
                        # Calculate drawdown duration
                        drawdown_days = (datetime.strptime(recovery_date, '%Y-%m-%d') - 
                                       datetime.strptime(start_date, '%Y-%m-%d')).days
                        
                        drawdown_periods.append({
                            'start_date': start_date,
                            'peak_date': peak_date,
                            'valley_date': valley_date,
                            'recovery_date': recovery_date,
                            'drawdown_amount': drawdown_amount,
                            'recovery_days': recovery_days,
                            'drawdown_days': drawdown_days
                        })
                    
                    # Reset tracking variables
                    start_date = None
                    peak_date = None
                    peak_value = 0.0
                    recovery_date = None
            
            # Handle ongoing drawdown
            if in_drawdown:
                # Find valley date (minimum equity during drawdown)
                valley_date = None
                valley_value = float('inf')
                
                for j in range(dates.index(start_date), len(dates)):
                    check_date = dates[j]
                    check_value = self.metrics.equity_curve[check_date]
                    
                    if check_value < valley_value:
                        valley_value = check_value
                        valley_date = check_date
                
                # Calculate drawdown amount
                drawdown_amount = (peak_value - valley_value) / peak_value
                
                # Calculate drawdown duration so far
                drawdown_days = (datetime.strptime(dates[-1], '%Y-%m-%d') - 
                               datetime.strptime(start_date, '%Y-%m-%d')).days
                
                drawdown_periods.append({
                    'start_date': start_date,
                    'peak_date': peak_date,
                    'valley_date': valley_date,
                    'recovery_date': None,  # Ongoing
                    'drawdown_amount': drawdown_amount,
                    'recovery_days': None,  # Ongoing
                    'drawdown_days': drawdown_days,
                    'ongoing': True
                })
            
            # Calculate average drawdown
            if drawdown_periods:
                avg_drawdown = sum(period['drawdown_amount'] for period in drawdown_periods) / len(drawdown_periods)
            else:
                avg_drawdown = 0.0
            
            # Sort drawdown periods by amount (largest first)
            drawdown_periods.sort(key=lambda x: x['drawdown_amount'], reverse=True)
            
            return {
                'max_drawdown': self.metrics.max_drawdown,
                'current_drawdown': self.metrics.current_drawdown,
                'avg_drawdown': avg_drawdown,
                'drawdown_periods': drawdown_periods
            }
    
    def load_portfolio_history(self, history_data: Dict[str, float]) -> None:
        """
        Load portfolio history data.
        
        Args:
            history_data: Dictionary of date to portfolio value
        """
        with self.lock:
            for date, value in sorted(history_data.items()):
                self.metrics.update_equity_value(date, value)
            
            logger.info(f"Loaded portfolio history with {len(history_data)} data points")
    
    def load_benchmark_history(self, history_data: Dict[str, float]) -> None:
        """
        Load benchmark history data.
        
        Args:
            history_data: Dictionary of date to benchmark price
        """
        with self.lock:
            for date, price in sorted(history_data.items()):
                self.metrics.update_benchmark(date, price)
            
            logger.info(f"Loaded benchmark history with {len(history_data)} data points")
    
    def save_performance_data(self, file_path: str) -> bool:
        """
        Save performance data to file.
        
        Args:
            file_path: Path to save data
            
        Returns:
            Boolean indicating success
        """
        with self.lock:
            try:
                # Prepare data to save
                data = {
                    'equity_curve': self.metrics.equity_curve,
                    'daily_returns': self.metrics.daily_returns,
                    'drawdowns': self.metrics.drawdowns,
                    'benchmark_prices': self.metrics.benchmark_prices,
                    'benchmark_returns': self.metrics.benchmark_returns,
                    'performance_metrics': self.get_performance_metrics(),
                    'monthly_metrics': self.monthly_metrics,
                    'trade_metrics': self.trade_metrics,
                    'alerts': {
                        'drawdown_alerts': self.drawdown_alerts,
                        'sharpe_alerts': self.sharpe_alerts,
                        'return_alerts': self.return_alerts
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Saved performance data to {file_path}")
                return True
            
            except Exception as e:
                logger.error(f"Error saving performance data: {str(e)}")
                return False
    
    def load_performance_data(self, file_path: str) -> bool:
        """
        Load performance data from file.
        
        Args:
            file_path: Path to load data from
            
        Returns:
            Boolean indicating success
        """
        with self.lock:
            try:
                # Load from file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Load equity curve
                for date, value in data.get('equity_curve', {}).items():
                    self.metrics.update_equity_value(date, value)
                
                # Load benchmark prices
                for date, price in data.get('benchmark_prices', {}).items():
                    self.metrics.update_benchmark(date, price)
                
                # Load monthly metrics
                self.monthly_metrics = data.get('monthly_metrics', {})
                
                # Load trade metrics
                self.trade_metrics = data.get('trade_metrics', {})
                
                # Load alerts
                self.drawdown_alerts = data.get('alerts', {}).get('drawdown_alerts', [])
                self.sharpe_alerts = data.get('alerts', {}).get('sharpe_alerts', [])
                self.return_alerts = data.get('alerts', {}).get('return_alerts', [])
                
                logger.info(f"Loaded performance data from {file_path}")
                return True
            
            except Exception as e:
                logger.error(f"Error loading performance data: {str(e)}")
                return False