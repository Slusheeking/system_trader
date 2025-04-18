#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics Module
------------
This module provides utilities for calculating performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any


def calculate_metrics(returns: Union[List[float], np.ndarray, pd.Series],
                     benchmark_returns: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
                     risk_free_rate: float = 0.0,
                     annualization_factor: int = 252) -> Dict[str, float]:
    """
    Calculate performance metrics for a series of returns.

    Args:
        returns: List or array of returns
        benchmark_returns: Optional list or array of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)

    Returns:
        Dictionary with calculated metrics
    """
    # Convert to numpy array
    if isinstance(returns, (list, pd.Series)):
        returns = np.array(returns)
    
    if benchmark_returns is not None and isinstance(benchmark_returns, (list, pd.Series)):
        benchmark_returns = np.array(benchmark_returns)
    
    # Calculate metrics
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (annualization_factor / len(returns)) - 1
    volatility = returns.std() * np.sqrt(annualization_factor)
    
    # Sharpe ratio
    excess_returns = returns - risk_free_rate / annualization_factor
    sharpe_ratio = np.mean(excess_returns) / returns.std() * np.sqrt(annualization_factor)
    
    # Maximum drawdown
    cum_returns = np.cumprod(1 + returns)
    max_drawdown = np.min(cum_returns / np.maximum.accumulate(cum_returns) - 1)
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = downside_returns.std() * np.sqrt(annualization_factor)
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(annualization_factor)
    else:
        sortino_ratio = np.nan
    
    # Calmar ratio
    if max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = np.nan
    
    # Alpha and beta (if benchmark provided)
    alpha, beta = np.nan, np.nan
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        # Beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        
        # Alpha (annualized)
        alpha = annualized_return - risk_free_rate - beta * (
            (np.prod(1 + benchmark_returns) - 1) ** (annualization_factor / len(benchmark_returns)) - 1 - risk_free_rate
        )
    
    # Information ratio
    information_ratio = np.nan
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(annualization_factor)
        if tracking_error != 0:
            information_ratio = (annualized_return - 
                               ((np.prod(1 + benchmark_returns) - 1) ** 
                                (annualization_factor / len(benchmark_returns)) - 1)) / tracking_error
    
    # Return metrics dictionary
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio
    }
    
    return metrics
