#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diversification Engine
--------------------
Analyzes and manages portfolio diversification.
Provides tools for sector balance, asset allocation, and correlation management.
"""

import logging
import time
import json
import yaml # Import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
from scipy import stats, optimize

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from portfolio.risk_calculator import RiskCalculator

# Setup logging
logger = setup_logger('diversification_engine')


class DiversificationEngine:
    """
    Analyzes and manages portfolio diversification.
    """
    
    def __init__(self, risk_calculator: Optional[RiskCalculator] = None, config_path: Optional[str] = None):
        """
        Initialize diversification engine.
        
        Args:
            risk_calculator: RiskCalculator instance or None to create a new one
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Use provided risk calculator or create a new one
        self.risk_calculator = risk_calculator or RiskCalculator(config_path)
        
        # Diversification parameters
        self.max_sector_allocation = self.config.get('max_sector_allocation', 0.30)  # 30% max per sector
        self.max_industry_allocation = self.config.get('max_industry_allocation', 0.20)  # 20% max per industry
        self.max_single_stock_allocation = self.config.get('max_single_stock_allocation', 0.10)  # 10% max per stock
        self.min_position_count = self.config.get('min_position_count', 10)  # Minimum 10 positions for diversification
        self.max_correlation_threshold = self.config.get('max_correlation_threshold', 0.70)  # Maximum correlation between positions
        
        # Target allocations
        self.target_sector_allocations = self.config.get('target_sector_allocations', {})
        self.target_factor_exposures = self.config.get('target_factor_exposures', {})
        
        # Sector and industry mappings
        self.sector_mappings = {}  # Symbol to sector mapping
        self.industry_mappings = {}  # Symbol to industry mapping
        self.factor_exposures = {}  # Symbol to factor exposures
        
        # Current allocations
        self.current_sector_allocations = {}
        self.current_industry_allocations = {}
        self.current_factor_exposures = {}
        
        # Diversification metrics
        self.herfindahl_index = 1.0  # 1.0 = completely concentrated, lower is more diversified
        self.effective_position_count = 1
        
        # Last updated timestamp
        self.last_updated = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Diversification engine initialized")
    
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
                config_data = yaml.safe_load(f) # Use yaml.safe_load
            
            # Extract the 'diversification' section if it exists
            config = config_data.get('diversification', {}) if config_data else {}
            
            logger.info(f"Loaded diversification configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def load_sector_mappings(self, mappings: Dict[str, str]):
        """
        Load sector mappings for symbols.
        
        Args:
            mappings: Dictionary of symbol to sector
        """
        with self.lock:
            self.sector_mappings = mappings
            logger.info(f"Loaded sector mappings for {len(mappings)} symbols")
    
    def load_industry_mappings(self, mappings: Dict[str, str]):
        """
        Load industry mappings for symbols.
        
        Args:
            mappings: Dictionary of symbol to industry
        """
        with self.lock:
            self.industry_mappings = mappings
            logger.info(f"Loaded industry mappings for {len(mappings)} symbols")
    
    def load_factor_exposures(self, factor_data: Dict[str, Dict[str, float]]):
        """
        Load factor exposures for symbols.
        
        Args:
            factor_data: Dictionary of symbol to factor exposures
        """
        with self.lock:
            self.factor_exposures = factor_data
            logger.info(f"Loaded factor exposures for {len(factor_data)} symbols")
    
    def analyze_portfolio_diversity(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze portfolio diversification.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with diversification metrics
        """
        with self.lock:
            if not positions:
                return {
                    'herfindahl_index': 1.0,
                    'effective_position_count': 1,
                    'sector_allocations': {},
                    'industry_allocations': {},
                    'factor_exposures': {},
                    'diversification_score': 0.0,
                    'warnings': ['No positions in portfolio']
                }
            
            # Calculate portfolio value
            portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
            
            if portfolio_value == 0:
                return {
                    'herfindahl_index': 1.0,
                    'effective_position_count': 1,
                    'sector_allocations': {},
                    'industry_allocations': {},
                    'factor_exposures': {},
                    'diversification_score': 0.0,
                    'warnings': ['Portfolio value is zero']
                }
            
            # Calculate position weights
            position_weights = {}
            for position in positions:
                symbol = position.get('symbol')
                weight = position.get('current_value', 0.0) / portfolio_value
                position_weights[symbol] = weight
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = sum(weight**2 for weight in position_weights.values())
            self.herfindahl_index = hhi
            
            # Calculate effective number of positions (inverse of HHI)
            self.effective_position_count = 1 / hhi if hhi > 0 else 1
            
            # Calculate sector allocations
            sector_allocations = {}
            for position in positions:
                symbol = position.get('symbol')
                weight = position.get('current_value', 0.0) / portfolio_value
                
                sector = self.sector_mappings.get(symbol, 'Unknown')
                if sector not in sector_allocations:
                    sector_allocations[sector] = 0
                sector_allocations[sector] += weight
            
            self.current_sector_allocations = sector_allocations
            
            # Calculate industry allocations
            industry_allocations = {}
            for position in positions:
                symbol = position.get('symbol')
                weight = position.get('current_value', 0.0) / portfolio_value
                
                industry = self.industry_mappings.get(symbol, 'Unknown')
                if industry not in industry_allocations:
                    industry_allocations[industry] = 0
                industry_allocations[industry] += weight
            
            self.current_industry_allocations = industry_allocations
            
            # Calculate factor exposures
            factor_exposures = {}
            for position in positions:
                symbol = position.get('symbol')
                weight = position.get('current_value', 0.0) / portfolio_value
                
                if symbol in self.factor_exposures:
                    for factor, exposure in self.factor_exposures[symbol].items():
                        if factor not in factor_exposures:
                            factor_exposures[factor] = 0
                        factor_exposures[factor] += weight * exposure
            
            self.current_factor_exposures = factor_exposures
            
            # Check for diversification warnings
            warnings = []
            
            # Check sector concentration
            for sector, allocation in sector_allocations.items():
                if allocation > self.max_sector_allocation:
                    warnings.append(f"Sector concentration: {sector} at {allocation:.1%} exceeds {self.max_sector_allocation:.1%} limit")
            
            # Check industry concentration
            for industry, allocation in industry_allocations.items():
                if allocation > self.max_industry_allocation:
                    warnings.append(f"Industry concentration: {industry} at {allocation:.1%} exceeds {self.max_industry_allocation:.1%} limit")
            
            # Check individual position sizes
            for symbol, weight in position_weights.items():
                if weight > self.max_single_stock_allocation:
                    warnings.append(f"Position size: {symbol} at {weight:.1%} exceeds {self.max_single_stock_allocation:.1%} limit")
            
            # Check minimum position count
            if len(positions) < self.min_position_count:
                warnings.append(f"Position count: {len(positions)} positions is below minimum of {self.min_position_count}")
            
            # Calculate overall diversification score (0-100)
            # Lower HHI = better diversification
            hhi_score = max(0, min(100, 100 * (1 - self.herfindahl_index)))
            
            # More sectors = better diversification
            sector_count = len([s for s, a in sector_allocations.items() if a > 0.02])  # Sectors with >2% allocation
            sector_score = max(0, min(100, 100 * (sector_count / 11)))  # Assuming 11 major sectors
            
            # Factor balance
            factor_balance_score = 50  # Default middle score
            if factor_exposures:
                # Calculate variance of factor exposures
                factor_values = list(factor_exposures.values())
                if len(factor_values) > 1:
                    factor_variance = np.var(factor_values)
                    # Lower variance = better balance
                    factor_balance_score = max(0, min(100, 100 * (1 - factor_variance)))
            
            # Combine scores with weights
            diversification_score = 0.50 * hhi_score + 0.30 * sector_score + 0.20 * factor_balance_score
            
            # Update timestamp
            self.last_updated = datetime.now()
            
            return {
                'herfindahl_index': self.herfindahl_index,
                'effective_position_count': self.effective_position_count,
                'sector_allocations': self.current_sector_allocations,
                'industry_allocations': self.current_industry_allocations,
                'factor_exposures': self.current_factor_exposures,
                'position_weights': position_weights,
                'diversification_score': diversification_score,
                'warnings': warnings,
                'timestamp': self.last_updated.isoformat()
            }
    
    def calculate_correlation_matrix(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate correlation matrix for current positions.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with correlation analysis
        """
        with self.lock:
            # Get symbols from positions
            symbols = [position.get('symbol') for position in positions if position.get('symbol')]
            
            if len(symbols) <= 1:
                return {
                    'correlation_matrix': {},
                    'avg_correlation': 0.0,
                    'high_correlation_pairs': [],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Use risk calculator to get correlation data
            correlation_data = self.risk_calculator.calculate_portfolio_correlation(symbols)
            
            # Extract correlation matrix
            correlation_matrix = correlation_data.get('correlation_matrix', {})
            
            # Find highly correlated pairs
            high_correlation_pairs = []
            
            if correlation_matrix:
                for i, symbol1 in enumerate(symbols):
                    for j, symbol2 in enumerate(symbols):
                        if i < j:  # Only check each pair once
                            if symbol1 in correlation_matrix and symbol2 in correlation_matrix.get(symbol1, {}):
                                correlation = correlation_matrix[symbol1][symbol2]
                                
                                if abs(correlation) > self.max_correlation_threshold:
                                    high_correlation_pairs.append({
                                        'symbol1': symbol1,
                                        'symbol2': symbol2,
                                        'correlation': correlation
                                    })
            
            return {
                'correlation_matrix': correlation_matrix,
                'avg_correlation': correlation_data.get('average_correlation', 0.0),
                'high_correlation_pairs': high_correlation_pairs,
                'timestamp': datetime.now().isoformat()
            }
    
    def recommend_position_adjustments(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recommend position adjustments to improve diversification.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with recommended adjustments
        """
        with self.lock:
            # First analyze current diversification
            diversification = self.analyze_portfolio_diversity(positions)
            correlation_analysis = self.calculate_correlation_matrix(positions)
            
            # Check if we need to make adjustments
            warnings = diversification.get('warnings', [])
            if not warnings and not correlation_analysis.get('high_correlation_pairs', []):
                return {
                    'adjustments_needed': False,
                    'recommendations': [],
                    'diversification': diversification,
                    'correlation_analysis': correlation_analysis,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Extract current position weights
            position_weights = diversification.get('position_weights', {})
            portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
            
            recommendations = []
            
            # Handle sector concentration issues
            for sector, allocation in diversification.get('sector_allocations', {}).items():
                if allocation > self.max_sector_allocation:
                    excess = allocation - self.max_sector_allocation
                    
                    # Find positions in this sector
                    sector_positions = [p for p in positions if self.sector_mappings.get(p.get('symbol')) == sector]
                    
                    # Sort by position size (largest first)
                    sector_positions.sort(key=lambda x: x.get('current_value', 0), reverse=True)
                    
                    for position in sector_positions:
                        symbol = position.get('symbol')
                        weight = position.get('current_value', 0) / portfolio_value
                        
                        # Calculate reduction
                        reduction_pct = min(0.5, excess / allocation)  # Cap at 50% reduction
                        reduction_value = position.get('current_value', 0) * reduction_pct
                        
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'reduce',
                            'reason': f'Sector concentration in {sector}',
                            'reduction_pct': reduction_pct,
                            'reduction_value': reduction_value
                        })
                        
                        # Update tracking variables
                        excess -= weight * reduction_pct
                        if excess <= 0:
                            break
            
            # Handle individual position concentration
            for symbol, weight in position_weights.items():
                if weight > self.max_single_stock_allocation:
                    # Check if we already have a recommendation for this symbol
                    if not any(r.get('symbol') == symbol for r in recommendations):
                        excess = weight - self.max_single_stock_allocation
                        reduction_pct = excess / weight
                        
                        # Find position data
                        position_data = next((p for p in positions if p.get('symbol') == symbol), None)
                        
                        if position_data:
                            reduction_value = position_data.get('current_value', 0) * reduction_pct
                            
                            recommendations.append({
                                'symbol': symbol,
                                'action': 'reduce',
                                'reason': 'Position size too large',
                                'reduction_pct': reduction_pct,
                                'reduction_value': reduction_value
                            })
            
            # Handle high correlations
            for pair in correlation_analysis.get('high_correlation_pairs', []):
                symbol1 = pair.get('symbol1')
                symbol2 = pair.get('symbol2')
                correlation = pair.get('correlation')
                
                # Find position data
                pos1 = next((p for p in positions if p.get('symbol') == symbol1), None)
                pos2 = next((p for p in positions if p.get('symbol') == symbol2), None)
                
                if pos1 and pos2:
                    # Decide which position to reduce (smaller one typically)
                    if pos1.get('current_value', 0) <= pos2.get('current_value', 0):
                        target_pos = pos1
                        target_symbol = symbol1
                    else:
                        target_pos = pos2
                        target_symbol = symbol2
                    
                    # Check if we already have a recommendation for this symbol
                    if not any(r.get('symbol') == target_symbol for r in recommendations):
                        reduction_pct = 0.25  # Reduce by 25%
                        reduction_value = target_pos.get('current_value', 0) * reduction_pct
                        
                        recommendations.append({
                            'symbol': target_symbol,
                            'action': 'reduce',
                            'reason': f'High correlation ({correlation:.2f}) with {symbol1 if target_symbol == symbol2 else symbol2}',
                            'reduction_pct': reduction_pct,
                            'reduction_value': reduction_value
                        })
            
            # Limit number of recommendations
            recommendations = sorted(recommendations, key=lambda x: x.get('reduction_value', 0), reverse=True)
            recommendations = recommendations[:5]  # Limit to top 5 recommendations
            
            return {
                'adjustments_needed': bool(recommendations),
                'recommendations': recommendations,
                'diversification': diversification,
                'correlation_analysis': correlation_analysis,
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_portfolio(self, positions: List[Dict[str, Any]], target_risk: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights for better diversification and risk/return.
        
        Args:
            positions: List of position dictionaries
            target_risk: Optional target risk level (annualized volatility)
            
        Returns:
            Dictionary with optimized portfolio weights
        """
        with self.lock:
            if not positions or len(positions) < 2:
                return {
                    'success': False,
                    'error': 'Not enough positions for optimization',
                    'optimal_weights': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            try:
                # Get symbols from positions
                symbols = [position.get('symbol') for position in positions if position.get('symbol')]
                
                # Get current weights
                portfolio_value = sum(position.get('current_value', 0.0) for position in positions)
                current_weights = {p.get('symbol'): p.get('current_value', 0.0) / portfolio_value for p in positions}
                
                # Get expected returns (use historical returns as a proxy)
                expected_returns = {}
                for symbol in symbols:
                    if symbol in self.risk_calculator.historical_returns:
                        returns = self.risk_calculator.historical_returns[symbol]
                        if isinstance(returns, pd.DataFrame) and 'returns' in returns.columns:
                            expected_returns[symbol] = returns['returns'].mean() * 252  # Annualized
                        elif isinstance(returns, pd.Series):
                            expected_returns[symbol] = returns.mean() * 252  # Annualized
                
                # If we don't have returns for all symbols, use a default value
                for symbol in symbols:
                    if symbol not in expected_returns:
                        expected_returns[symbol] = 0.05  # Default 5% expected return
                
                # Convert to numpy arrays for optimization
                symbols_array = np.array(symbols)
                returns_array = np.array([expected_returns[s] for s in symbols])
                
                # Get correlation matrix
                correlation_data = self.calculate_correlation_matrix(positions)
                correlation_matrix = correlation_data.get('correlation_matrix', {})
                
                # Convert to numpy array
                cov_matrix = np.zeros((len(symbols), len(symbols)))
                
                if correlation_matrix:
                    for i, symbol1 in enumerate(symbols):
                        for j, symbol2 in enumerate(symbols):
                            if symbol1 in correlation_matrix and symbol2 in correlation_matrix.get(symbol1, {}):
                                vol1 = self.risk_calculator.volatility_estimates.get(symbol1, 0.25)
                                vol2 = self.risk_calculator.volatility_estimates.get(symbol2, 0.25)
                                corr = correlation_matrix[symbol1][symbol2]
                                cov_matrix[i, j] = corr * vol1 * vol2
                else:
                    # If no correlation matrix available, use identity matrix
                    np.fill_diagonal(cov_matrix, 0.04)  # Default variance
                
                # Define optimization function
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                def portfolio_return(weights):
                    return np.sum(returns_array * weights)
                
                def negative_sharpe_ratio(weights):
                    return -portfolio_return(weights) / portfolio_volatility(weights)
                
                # Constraints
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
                
                # Bounds (0-20% per position)
                bounds = tuple((0, self.max_single_stock_allocation) for _ in range(len(symbols)))
                
                # Initial guess (equal weight)
                initial_weights = np.array([1.0/len(symbols)] * len(symbols))
                
                # Run optimization
                if target_risk is not None:
                    # Target risk optimization
                    def risk_objective(weights):
                        return (portfolio_volatility(weights) - target_risk)**2
                    
                    result = optimize.minimize(risk_objective, initial_weights, method='SLSQP',
                                             bounds=bounds, constraints=constraints)
                else:
                    # Maximize Sharpe ratio
                    result = optimize.minimize(negative_sharpe_ratio, initial_weights, method='SLSQP',
                                             bounds=bounds, constraints=constraints)
                
                if result['success']:
                    optimal_weights = result['x']
                    
                    # Convert to dictionary
                    weight_dict = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
                    
                    # Calculate expected metrics
                    expected_return = portfolio_return(optimal_weights)
                    expected_volatility = portfolio_volatility(optimal_weights)
                    expected_sharpe = expected_return / expected_volatility if expected_volatility > 0 else 0
                    
                    # Calculate changes from current weights
                    weight_changes = {s: weight_dict.get(s, 0) - current_weights.get(s, 0) for s in symbols}
                    
                    # Calculate diversification improvement
                    current_hhi = sum(w**2 for w in current_weights.values())
                    optimal_hhi = sum(w**2 for w in weight_dict.values())
                    diversification_improvement = (current_hhi - optimal_hhi) / current_hhi if current_hhi > 0 else 0
                    
                    return {
                        'success': True,
                        'optimal_weights': weight_dict,
                        'expected_return': expected_return,
                        'expected_volatility': expected_volatility,
                        'expected_sharpe': expected_sharpe,
                        'weight_changes': weight_changes,
                        'diversification_improvement': diversification_improvement,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Optimization failed: {result.get('message', 'Unknown error')}",
                        'optimal_weights': {},
                        'timestamp': datetime.now().isoformat()
                    }
            
            except Exception as e:
                logger.error(f"Error optimizing portfolio: {str(e)}")
                return {
                    'success': False,
                    'error': f"Optimization error: {str(e)}",
                    'optimal_weights': {},
                    'timestamp': datetime.now().isoformat()
                }
    
    def analyze_sector_balance(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sector balance against market benchmarks.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with sector balance analysis
        """
        with self.lock:
            # Calculate current sector allocations
            diversification = self.analyze_portfolio_diversity(positions)
            current_allocations = diversification.get('sector_allocations', {})
            
            # Compare to target allocations
            deviations = {}
            
            for sector, target in self.target_sector_allocations.items():
                current = current_allocations.get(sector, 0.0)
                deviation = current - target
                deviations[sector] = {
                    'current': current,
                    'target': target,
                    'deviation': deviation,
                    'is_overweight': deviation > 0.05,  # >5% overweight
                    'is_underweight': deviation < -0.05  # >5% underweight
                }
            
            # Check for sectors in portfolio but not in targets
            for sector in current_allocations:
                if sector not in deviations and sector != 'Unknown':
                    deviations[sector] = {
                        'current': current_allocations[sector],
                        'target': 0.0,
                        'deviation': current_allocations[sector],
                        'is_overweight': current_allocations[sector] > 0.05,
                        'is_underweight': False
                    }
            
            # Calculate total deviation
            total_deviation = sum(abs(d['deviation']) for d in deviations.values()) / 2
            
            return {
                'current_allocations': current_allocations,
                'target_allocations': self.target_sector_allocations,
                'deviations': deviations,
                'total_deviation': total_deviation,
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_factor_exposures(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze factor exposures of the portfolio.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with factor exposure analysis
        """
        with self.lock:
            # Calculate current factor exposures
            diversification = self.analyze_portfolio_diversity(positions)
            current_exposures = diversification.get('factor_exposures', {})
            
            # Compare to target exposures
            deviations = {}
            
            for factor, target in self.target_factor_exposures.items():
                current = current_exposures.get(factor, 0.0)
                deviation = current - target
                deviations[factor] = {
                    'current': current,
                    'target': target,
                    'deviation': deviation,
                    'is_overexposed': deviation > 0.2,  # >0.2 overexposed
                    'is_underexposed': deviation < -0.2  # >0.2 underexposed
                }
            
            # Check for factors in portfolio but not in targets
            for factor in current_exposures:
                if factor not in deviations:
                    deviations[factor] = {
                        'current': current_exposures[factor],
                        'target': 0.0,
                        'deviation': current_exposures[factor],
                        'is_overexposed': current_exposures[factor] > 0.2,
                        'is_underexposed': False
                    }
            
            # Calculate total deviation
            total_deviation = sum(abs(d['deviation']) for d in deviations.values()) / 2
            
            return {
                'current_exposures': current_exposures,
                'target_exposures': self.target_factor_exposures,
                'deviations': deviations,
                'total_deviation': total_deviation,
                'timestamp': datetime.now().isoformat()
            }