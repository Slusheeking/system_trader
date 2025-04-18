#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Analyzer Module
--------------------------
This module provides utilities for analyzing trading strategy performance.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes trading strategy performance by computing metrics and generating reports.
    """

    def __init__(self, strategy_name: str = "unnamed_strategy"):
        """
        Initialize the performance analyzer.

        Args:
            strategy_name: Name of the trading strategy being analyzed
        """
        self.strategy_name = strategy_name
        self.metrics = {}
        self.trades_df = None

    def compute_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute performance metrics from a dataframe of trades.

        Args:
            trades_df: DataFrame containing trade information with at least
                      'timestamp', 'pnl', and 'cumulative_pnl' columns

        Returns:
            Dictionary containing performance metrics
        """
        try:
            if trades_df.empty:
                logger.warning("Empty trades dataframe provided. Cannot compute metrics.")
                return {"error": "No trades data available"}

            self.trades_df = trades_df.copy()
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'pnl']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Sort by timestamp if not already sorted
            if not pd.api.types.is_datetime64_dtype(trades_df['timestamp']):
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Calculate returns if not provided
            if 'returns' not in trades_df.columns:
                # Assuming initial capital of 100,000 if not provided
                initial_capital = 100000
                if 'cumulative_pnl' in trades_df.columns:
                    equity_curve = initial_capital + trades_df['cumulative_pnl']
                else:
                    # Calculate cumulative P&L
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    equity_curve = initial_capital + trades_df['cumulative_pnl']
                
                trades_df['equity'] = equity_curve
                trades_df['returns'] = trades_df['equity'].pct_change().fillna(0)
            
            # Calculate key metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate returns metrics
            total_return = trades_df['returns'].sum()
            annualized_return = self._calculate_annualized_return(trades_df['returns'])
            
            # Calculate risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio(trades_df['returns'])
            sortino_ratio = self._calculate_sortino_ratio(trades_df['returns'])
            
            # Calculate drawdown metrics
            drawdown_series, max_drawdown = self._calculate_drawdown(trades_df)
            trades_df['drawdown'] = drawdown_series
            
            # Store metrics
            self.metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': self._calculate_profit_factor(trades_df),
                'average_trade': trades_df['pnl'].mean(),
                'average_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
                'average_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
                'max_consecutive_wins': self._calculate_max_consecutive(trades_df, 'wins'),
                'max_consecutive_losses': self._calculate_max_consecutive(trades_df, 'losses')
            }
            
            logger.info(f"Computed performance metrics for {self.strategy_name}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {"error": str(e)}

    def generate_reports(self, output_dir: str) -> Dict[str, str]:
        """
        Generate performance reports and save to the specified directory.

        Args:
            output_dir: Directory where reports will be saved

        Returns:
            Dictionary with paths to generated reports
        """
        try:
            if self.trades_df is None or self.trades_df.empty:
                logger.warning("No trade data available for generating reports")
                return {"error": "No trade data available"}
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            report_files = {}
            
            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, f"{self.strategy_name}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            report_files['metrics_json'] = metrics_file
            
            # Generate equity curve chart
            equity_curve_file = os.path.join(output_dir, f"{self.strategy_name}_equity_curve.png")
            self._plot_equity_curve(equity_curve_file)
            report_files['equity_curve'] = equity_curve_file
            
            # Generate drawdown chart
            drawdown_file = os.path.join(output_dir, f"{self.strategy_name}_drawdown.png")
            self._plot_drawdown(drawdown_file)
            report_files['drawdown_curve'] = drawdown_file
            
            # Generate monthly returns heatmap
            monthly_returns_file = os.path.join(output_dir, f"{self.strategy_name}_monthly_returns.png")
            self._plot_monthly_returns(monthly_returns_file)
            report_files['monthly_returns'] = monthly_returns_file
            
            # Try to log to MLflow if available
            self._log_to_mlflow(output_dir)
            
            logger.info(f"Generated performance reports in {output_dir}")
            return report_files
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return {"error": str(e)}

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate the Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods in a year (default: 252 trading days)

        Returns:
            Sharpe ratio value
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate the Sortino ratio (similar to Sharpe but only considers downside deviation).

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default: 0.0)
            periods_per_year: Number of periods in a year (default: 252 trading days)

        Returns:
            Sortino ratio value
        """
        if returns.empty:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()

    def _calculate_drawdown(self, trades_df: pd.DataFrame) -> Tuple[pd.Series, float]:
        """
        Calculate drawdown series and maximum drawdown.

        Args:
            trades_df: DataFrame with equity or cumulative P&L

        Returns:
            Tuple of (drawdown series, maximum drawdown value)
        """
        if 'equity' in trades_df.columns:
            equity = trades_df['equity']
        elif 'cumulative_pnl' in trades_df.columns:
            # Assuming initial capital of 100,000 if not provided
            equity = 100000 + trades_df['cumulative_pnl']
        else:
            # If neither is available, calculate from pnl
            equity = 100000 + trades_df['pnl'].cumsum()
        
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return drawdown, max_drawdown

    def _calculate_annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return from a series of returns.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year (default: 252 trading days)

        Returns:
            Annualized return as a float
        """
        if returns.empty:
            return 0.0
        
        total_periods = len(returns)
        if total_periods < 2:
            return returns.sum()
        
        # Calculate compound return
        compound_return = (1 + returns).prod() - 1
        
        # Annualize
        years = total_periods / periods_per_year
        annualized_return = (1 + compound_return) ** (1 / years) - 1
        
        return annualized_return

    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Args:
            trades_df: DataFrame with trade information

        Returns:
            Profit factor as a float
        """
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss

    def _calculate_max_consecutive(self, trades_df: pd.DataFrame, win_or_loss: str) -> int:
        """
        Calculate maximum consecutive wins or losses.

        Args:
            trades_df: DataFrame with trade information
            win_or_loss: Either 'wins' or 'losses'

        Returns:
            Maximum consecutive count
        """
        if trades_df.empty:
            return 0
        
        # Create a series indicating wins (1) or losses (-1)
        result_series = np.where(trades_df['pnl'] > 0, 1, -1)
        
        # Determine what we're looking for
        target = 1 if win_or_loss == 'wins' else -1
        
        # Count consecutive occurrences
        max_consecutive = 0
        current_streak = 0
        
        for result in result_series:
            if result == target:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0
                
        return max_consecutive

    def _plot_equity_curve(self, output_file: str) -> None:
        """
        Plot equity curve and save to file.

        Args:
            output_file: Path where the plot will be saved
        """
        plt.figure(figsize=(12, 6))
        
        if 'equity' in self.trades_df.columns:
            equity = self.trades_df['equity']
        elif 'cumulative_pnl' in self.trades_df.columns:
            equity = 100000 + self.trades_df['cumulative_pnl']
        else:
            equity = 100000 + self.trades_df['pnl'].cumsum()
        
        plt.plot(self.trades_df['timestamp'], equity, linewidth=2)
        plt.title(f'Equity Curve - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _plot_drawdown(self, output_file: str) -> None:
        """
        Plot drawdown curve and save to file.

        Args:
            output_file: Path where the plot will be saved
        """
        plt.figure(figsize=(12, 6))
        
        if 'drawdown' not in self.trades_df.columns:
            drawdown, _ = self._calculate_drawdown(self.trades_df)
        else:
            drawdown = self.trades_df['drawdown']
        
        plt.fill_between(self.trades_df['timestamp'], 0, drawdown, color='red', alpha=0.3)
        plt.plot(self.trades_df['timestamp'], drawdown, color='red', linewidth=1)
        plt.title(f'Drawdown - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _plot_monthly_returns(self, output_file: str) -> None:
        """
        Plot monthly returns heatmap and save to file.

        Args:
            output_file: Path where the plot will be saved
        """
        if 'returns' not in self.trades_df.columns:
            logger.warning("Returns data not available for monthly returns plot")
            return
        
        # Resample to daily returns if we have intraday data
        daily_returns = self.trades_df.set_index('timestamp')['returns'].resample('D').sum()
        
        # Create monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_table = monthly_returns.unstack(level=0)
        
        plt.figure(figsize=(12, 8))
        
        # If we have enough data for a heatmap
        if len(monthly_returns) >= 4:  # At least a few months of data
            monthly_returns_pivot = monthly_returns.to_frame()
            monthly_returns_pivot['Year'] = monthly_returns_pivot.index.year
            monthly_returns_pivot['Month'] = monthly_returns_pivot.index.month
            monthly_returns_pivot = monthly_returns_pivot.pivot_table(
                index='Year', columns='Month', values='returns'
            )
            
            plt.imshow(monthly_returns_pivot, cmap='RdYlGn', aspect='auto')
            plt.colorbar(label='Returns')
            plt.title(f'Monthly Returns Heatmap - {self.strategy_name}')
            plt.xlabel('Month')
            plt.ylabel('Year')
            
            # Set x-axis labels as month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(range(12), month_names)
            
            # Set y-axis labels as years
            plt.yticks(range(len(monthly_returns_pivot.index)), monthly_returns_pivot.index)
            
        else:
            # If not enough data, just plot the monthly returns as a bar chart
            monthly_returns.plot(kind='bar')
            plt.title(f'Monthly Returns - {self.strategy_name}')
            plt.xlabel('Month')
            plt.ylabel('Return')
            
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _log_to_mlflow(self, artifacts_dir: str) -> None:
        """
        Log metrics and artifacts to MLflow if available.

        Args:
            artifacts_dir: Directory containing artifacts to log
        """
        try:
            # Try to import MLflow
            import mlflow
            
            # Check if MLflow tracking module exists in the project
            try:
                import sys
                sys.path.append('/home/ubuntu/system_trader')
                from mlflow import tracking
                use_project_tracking = True
            except ImportError:
                use_project_tracking = False
            
            # Start a run
            with mlflow.start_run(run_name=f"backtest_{self.strategy_name}"):
                # Log metrics
                for key, value in self.metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log artifacts
                for file in os.listdir(artifacts_dir):
                    if file.startswith(self.strategy_name) and (file.endswith('.png') or file.endswith('.json')):
                        mlflow.log_artifact(os.path.join(artifacts_dir, file))
                
                logger.info("Successfully logged metrics and artifacts to MLflow")
                
        except ImportError:
            logger.info("MLflow not available, skipping MLflow logging")
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {str(e)}")