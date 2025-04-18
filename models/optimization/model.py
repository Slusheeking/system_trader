#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Optimization Model
---------------------------
This module implements various portfolio optimization strategies including
Efficient Frontier, Risk Parity, and Black-Litterman approaches.
"""

import logging
import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import optimization libraries
import cvxpy as cp
from scipy.optimize import minimize

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('optimization_model')


class PortfolioOptimizationModel:
    """
    Portfolio optimization model implementing various optimization strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the portfolio optimization model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Optimization method
        self.optimization_method = config.get('optimization_method', 'efficient_frontier')
        
        # Risk and return parameters
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # Annual risk-free rate
        self.target_return = config.get('target_return', 0.15)    # Annual target return
        
        # Position constraints
        self.max_position_size = config.get('max_position_size', 0.2)  # Maximum allocation to a single asset
        self.min_position_size = config.get('min_position_size', 0.01) # Minimum allocation if included
        
        # Sector constraints
        self.max_sector_allocation = config.get('max_sector_allocation', 0.3)  # Maximum allocation to a sector
        
        # Portfolio size constraints
        self.min_stocks = config.get('min_stocks', 10)  # Minimum number of stocks in portfolio
        self.max_stocks = config.get('max_stocks', 30)  # Maximum number of stocks in portfolio
        
        # Black-Litterman specific parameters
        self.bl_params = config.get('black_litterman', {
            'tau': 0.05,  # Scaling factor for prior covariance
            'risk_aversion': 2.5  # Risk aversion parameter
        })
        
        # Risk parity specific parameters
        self.rp_params = config.get('risk_parity', {
            'risk_budget_equal': True,  # Equal risk contribution if True
            'custom_risk_budget': None  # Custom risk budget if not equal
        })
        
        # Model state
        self.returns = None
        self.covariance_matrix = None
        self.weights = None
        self.expected_return = None
        self.expected_volatility = None
        self.sharpe_ratio = None
        self.efficient_frontier = None
        self.asset_names = None
        self.sector_mapping = None
        
        logger.info(f"Portfolio Optimization Model initialized with method: {self.optimization_method}")
    
    def prepare_data(self, returns_data: pd.DataFrame, sector_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for optimization.
        
        Args:
            returns_data: DataFrame with asset returns (columns are assets, rows are time periods)
            sector_data: Optional DataFrame mapping assets to sectors
            
        Returns:
            Tuple of (expected returns array, covariance matrix, asset names list)
        """
        logger.info("Preparing data for optimization")
        
        # Store asset names
        self.asset_names = returns_data.columns.tolist()
        
        # Calculate expected returns (annualized)
        expected_returns = returns_data.mean() * 252
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns_data.cov() * 252
        
        # Store returns and covariance matrix
        self.returns = expected_returns.values
        self.covariance_matrix = cov_matrix.values
        
        # Store sector mapping if provided
        if sector_data is not None:
            self.sector_mapping = {}
            for asset in self.asset_names:
                if asset in sector_data.index:
                    self.sector_mapping[asset] = sector_data.loc[asset, 'sector']
        
        logger.info(f"Prepared data for {len(self.asset_names)} assets")
        return self.returns, self.covariance_matrix, self.asset_names
    
    def optimize(self, returns_data: pd.DataFrame, sector_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run portfolio optimization.
        
        Args:
            returns_data: DataFrame with asset returns
            sector_data: Optional DataFrame mapping assets to sectors
            
        Returns:
            Dictionary with optimization results
        """
        # Prepare data
        self.prepare_data(returns_data, sector_data)
        
        # Run optimization based on selected method
        if self.optimization_method == 'efficient_frontier':
            return self._optimize_efficient_frontier()
        elif self.optimization_method == 'risk_parity':
            return self._optimize_risk_parity()
        elif self.optimization_method == 'black_litterman':
            return self._optimize_black_litterman()
        else:
            logger.error(f"Unknown optimization method: {self.optimization_method}")
            return {'error': f"Unknown optimization method: {self.optimization_method}"}
    
    def _optimize_efficient_frontier(self) -> Dict[str, Any]:
        """
        Optimize portfolio using mean-variance optimization (Efficient Frontier).
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Running Efficient Frontier optimization")
        
        n_assets = len(self.asset_names)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        returns = self.returns
        cov_matrix = self.covariance_matrix
        
        # Define objective function (maximize Sharpe ratio)
        portfolio_return = returns @ weights
        portfolio_risk = cp.sqrt(cp.quad_form(weights, cov_matrix))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= self.min_position_size,  # Minimum position size
            weights <= self.max_position_size   # Maximum position size
        ]
        
        # Add sector constraints if sector mapping is available
        if self.sector_mapping:
            sectors = set(self.sector_mapping.values())
            for sector in sectors:
                sector_indices = [i for i, asset in enumerate(self.asset_names) 
                                if self.sector_mapping.get(asset) == sector]
                if sector_indices:
                    sector_weights = cp.sum([weights[i] for i in sector_indices])
                    constraints.append(sector_weights <= self.max_sector_allocation)
        
        # Define and solve the problem
        problem = cp.Problem(cp.Maximize(sharpe_ratio), constraints)
        try:
            problem.solve()
            
            if problem.status == 'optimal':
                # Store results
                self.weights = weights.value
                self.expected_return = float(returns @ self.weights)
                self.expected_volatility = float(np.sqrt(self.weights @ cov_matrix @ self.weights))
                self.sharpe_ratio = (self.expected_return - self.risk_free_rate) / self.expected_volatility
                
                # Apply cardinality constraints as post-processing
                self._apply_cardinality_constraints()
                
                # Generate efficient frontier
                self._generate_efficient_frontier()
                
                # Create results dictionary
                results = {
                    'weights': {asset: float(weight) for asset, weight in zip(self.asset_names, self.weights)},
                    'expected_annual_return': float(self.expected_return),
                    'expected_annual_volatility': float(self.expected_volatility),
                    'sharpe_ratio': float(self.sharpe_ratio),
                    'status': 'optimal'
                }
                
                # Add sector allocations if available
                if self.sector_mapping:
                    sector_allocations = {}
                    for sector in set(self.sector_mapping.values()):
                        sector_weight = sum(self.weights[i] for i, asset in enumerate(self.asset_names)
                                          if self.sector_mapping.get(asset) == sector)
                        sector_allocations[sector] = float(sector_weight)
                    
                    results['sector_allocations'] = sector_allocations
                
                logger.info(f"Optimization successful: Sharpe ratio = {self.sharpe_ratio:.4f}")
                return results
            else:
                logger.error(f"Optimization failed with status: {problem.status}")
                return {'error': f"Optimization failed with status: {problem.status}"}
        
        except Exception as e:
            logger.error(f"Error in efficient frontier optimization: {str(e)}")
            return {'error': str(e)}
    
    def _optimize_risk_parity(self) -> Dict[str, Any]:
        """
        Optimize portfolio using risk parity approach.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Running Risk Parity optimization")
        
        n_assets = len(self.asset_names)
        cov_matrix = self.covariance_matrix
        
        # Define risk budget
        if self.rp_params.get('risk_budget_equal', True):
            risk_budget = np.ones(n_assets) / n_assets
        else:
            risk_budget = np.array(self.rp_params.get('custom_risk_budget', [1/n_assets] * n_assets))
            risk_budget = risk_budget / risk_budget.sum()  # Normalize
        
        # Define objective function for risk parity
        def risk_parity_objective(weights, cov_matrix, risk_budget):
            weights = np.array(weights)
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            risk_contribution = weights * (cov_matrix @ weights) / portfolio_risk
            risk_target = portfolio_risk * risk_budget
            return np.sum((risk_contribution - risk_target)**2)
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Constraints
        bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        
        def weight_sum_constraint(weights):
            return np.sum(weights) - 1.0
        
        constraints = ({'type': 'eq', 'fun': weight_sum_constraint})
        
        # Solve the optimization problem
        try:
            result = minimize(
                risk_parity_objective,
                initial_weights,
                args=(cov_matrix, risk_budget),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )
            
            if result.success:
                # Store results
                self.weights = result.x
                self.expected_return = float(self.returns @ self.weights)
                self.expected_volatility = float(np.sqrt(self.weights @ cov_matrix @ self.weights))
                self.sharpe_ratio = (self.expected_return - self.risk_free_rate) / self.expected_volatility
                
                # Apply cardinality constraints as post-processing
                self._apply_cardinality_constraints()
                
                # Create results dictionary
                results = {
                    'weights': {asset: float(weight) for asset, weight in zip(self.asset_names, self.weights)},
                    'expected_annual_return': float(self.expected_return),
                    'expected_annual_volatility': float(self.expected_volatility),
                    'sharpe_ratio': float(self.sharpe_ratio),
                    'risk_contribution': {
                        asset: float(rc) for asset, rc in zip(
                            self.asset_names, 
                            self.weights * (cov_matrix @ self.weights) / self.expected_volatility
                        )
                    },
                    'status': 'optimal'
                }
                
                # Add sector allocations if available
                if self.sector_mapping:
                    sector_allocations = {}
                    for sector in set(self.sector_mapping.values()):
                        sector_weight = sum(self.weights[i] for i, asset in enumerate(self.asset_names)
                                          if self.sector_mapping.get(asset) == sector)
                        sector_allocations[sector] = float(sector_weight)
                    
                    results['sector_allocations'] = sector_allocations
                
                logger.info(f"Risk Parity optimization successful: Sharpe ratio = {self.sharpe_ratio:.4f}")
                return results
            else:
                logger.error(f"Risk Parity optimization failed: {result.message}")
                return {'error': f"Optimization failed: {result.message}"}
        
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {'error': str(e)}
    
    def _optimize_black_litterman(self) -> Dict[str, Any]:
        """
        Optimize portfolio using Black-Litterman approach.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Running Black-Litterman optimization")
        
        n_assets = len(self.asset_names)
        cov_matrix = self.covariance_matrix
        
        # Parameters for Black-Litterman
        tau = self.bl_params.get('tau', 0.05)
        risk_aversion = self.bl_params.get('risk_aversion', 2.5)
        
        # Calculate market-implied returns (reverse optimization)
        # Assume market weights are proportional to market cap (equal for simplicity)
        market_weights = np.ones(n_assets) / n_assets
        implied_returns = risk_aversion * (cov_matrix @ market_weights)
        
        # Incorporate views if provided in config
        views = self.config.get('views', [])
        if views:
            # Process views
            P = []  # View matrix
            q = []  # View returns
            omega = []  # View uncertainty
            
            for view in views:
                view_vector = np.zeros(n_assets)
                for asset, weight in view.get('assets', {}).items():
                    if asset in self.asset_names:
                        asset_idx = self.asset_names.index(asset)
                        view_vector[asset_idx] = weight
                
                P.append(view_vector)
                q.append(view.get('expected_return', 0))
                omega.append(view.get('confidence', 1) ** -2)  # Convert confidence to variance
            
            # Convert to numpy arrays
            P = np.array(P)
            q = np.array(q)
            omega = np.diag(omega)
            
            # Calculate posterior returns
            prior_covariance = tau * cov_matrix
            posterior_covariance = np.linalg.inv(
                np.linalg.inv(prior_covariance) + P.T @ np.linalg.inv(omega) @ P
            )
            posterior_returns = posterior_covariance @ (
                np.linalg.inv(prior_covariance) @ implied_returns + P.T @ np.linalg.inv(omega) @ q
            )
        else:
            # Without views, use implied returns
            posterior_returns = implied_returns
        
        # Now use mean-variance optimization with Black-Litterman returns
        # Define optimization variables
        weights = cp.Variable(n_assets)
        returns = posterior_returns
        
        # Define objective function (maximize Sharpe ratio)
        portfolio_return = returns @ weights
        portfolio_risk = cp.sqrt(cp.quad_form(weights, cov_matrix))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= self.min_position_size,  # Minimum position size
            weights <= self.max_position_size   # Maximum position size
        ]
        
        # Add sector constraints if sector mapping is available
        if self.sector_mapping:
            sectors = set(self.sector_mapping.values())
            for sector in sectors:
                sector_indices = [i for i, asset in enumerate(self.asset_names) 
                                if self.sector_mapping.get(asset) == sector]
                if sector_indices:
                    sector_weights = cp.sum([weights[i] for i in sector_indices])
                    constraints.append(sector_weights <= self.max_sector_allocation)
        
        # Define and solve the problem
        problem = cp.Problem(cp.Maximize(sharpe_ratio), constraints)
        try:
            problem.solve()
            
            if problem.status == 'optimal':
                # Store results
                self.weights = weights.value
                self.expected_return = float(returns @ self.weights)
                self.expected_volatility = float(np.sqrt(self.weights @ cov_matrix @ self.weights))
                self.sharpe_ratio = (self.expected_return - self.risk_free_rate) / self.expected_volatility
                
                # Apply cardinality constraints as post-processing
                self._apply_cardinality_constraints()
                
                # Create results dictionary
                results = {
                    'weights': {asset: float(weight) for asset, weight in zip(self.asset_names, self.weights)},
                    'expected_annual_return': float(self.expected_return),
                    'expected_annual_volatility': float(self.expected_volatility),
                    'sharpe_ratio': float(self.sharpe_ratio),
                    'black_litterman_returns': {asset: float(ret) for asset, ret in zip(self.asset_names, posterior_returns)},
                    'status': 'optimal'
                }
                
                # Add sector allocations if available
                if self.sector_mapping:
                    sector_allocations = {}
                    for sector in set(self.sector_mapping.values()):
                        sector_weight = sum(self.weights[i] for i, asset in enumerate(self.asset_names)
                                          if self.sector_mapping.get(asset) == sector)
                        sector_allocations[sector] = float(sector_weight)
                    
                    results['sector_allocations'] = sector_allocations
                
                logger.info(f"Black-Litterman optimization successful: Sharpe ratio = {self.sharpe_ratio:.4f}")
                return results
            else:
                logger.error(f"Black-Litterman optimization failed with status: {problem.status}")
                return {'error': f"Optimization failed with status: {problem.status}"}
        
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {'error': str(e)}
    
    def _apply_cardinality_constraints(self) -> None:
        """
        Apply cardinality constraints (min and max number of stocks) as post-processing.
        """
        # Sort weights in descending order
        sorted_indices = np.argsort(self.weights)[::-1]
        
        # Initialize new weights
        new_weights = np.zeros_like(self.weights)
        
        # Determine number of stocks to include
        n_stocks = min(max(self.min_stocks, np.sum(self.weights > self.min_position_size)), self.max_stocks)
        
        # Select top n_stocks
        selected_indices = sorted_indices[:n_stocks]
        
        # Allocate weights to selected stocks
        selected_weights = self.weights[selected_indices]
        new_weights[selected_indices] = selected_weights / np.sum(selected_weights)  # Normalize
        
        # Update weights
        self.weights = new_weights
        
        # Recalculate expected return and volatility
        self.expected_return = float(self.returns @ self.weights)
        self.expected_volatility = float(np.sqrt(self.weights @ self.covariance_matrix @ self.weights))
        self.sharpe_ratio = (self.expected_return - self.risk_free_rate) / self.expected_volatility
    
    def _generate_efficient_frontier(self, points: int = 100) -> None:
        """
        Generate the efficient frontier.
        
        Args:
            points: Number of points on the efficient frontier
        """
        n_assets = len(self.asset_names)
        returns = self.returns
        cov_matrix = self.covariance_matrix
        
        # Calculate minimum variance portfolio
        weights = cp.Variable(n_assets)
        portfolio_risk = cp.sqrt(cp.quad_form(weights, cov_matrix))
        
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= self.min_position_size,  # Minimum position size
            weights <= self.max_position_size   # Maximum position size
        ]
        
        problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
        problem.solve()
        
        min_risk = portfolio_risk.value
        min_return = returns @ weights.value
        
        # Calculate maximum return portfolio
        max_return_weights = np.zeros(n_assets)
        max_return_idx = np.argmax(returns)
        max_return_weights[max_return_idx] = 1
        max_return = returns[max_return_idx]
        
        # Generate efficient frontier
        target_returns = np.linspace(min_return, max_return, points)
        risks = []
        frontier_weights = []
        
        for target_return in target_returns:
            weights = cp.Variable(n_assets)
            portfolio_risk = cp.sqrt(cp.quad_form(weights, cov_matrix))
            
            constraints = [
                cp.sum(weights) == 1,  # Fully invested
                returns @ weights >= target_return,  # Target return
                weights >= self.min_position_size,  # Minimum position size
                weights <= self.max_position_size   # Maximum position size
            ]
            
            problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
            try:
                problem.solve()
                if problem.status == 'optimal':
                    risks.append(portfolio_risk.value)
                    frontier_weights.append(weights.value)
                else:
                    # If optimization fails, use previous weights
                    if frontier_weights:
                        risks.append(np.sqrt(frontier_weights[-1] @ cov_matrix @ frontier_weights[-1]))
                        frontier_weights.append(frontier_weights[-1])
                    else:
                        # For the first point, use minimum variance weights
                        min_var_weights = np.zeros(n_assets)
                        min_var_weights[np.argmin(np.diag(cov_matrix))] = 1
                        risks.append(np.sqrt(min_var_weights @ cov_matrix @ min_var_weights))
                        frontier_weights.append(min_var_weights)
            except:
                # If optimization fails, use previous weights
                if frontier_weights:
                    risks.append(np.sqrt(frontier_weights[-1] @ cov_matrix @ frontier_weights[-1]))
                    frontier_weights.append(frontier_weights[-1])
                else:
                    # For the first point, use minimum variance weights
                    min_var_weights = np.zeros(n_assets)
                    min_var_weights[np.argmin(np.diag(cov_matrix))] = 1
                    risks.append(np.sqrt(min_var_weights @ cov_matrix @ min_var_weights))
                    frontier_weights.append(min_var_weights)
        
        # Store efficient frontier
        self.efficient_frontier = {
            'returns': target_returns,
            'risks': risks,
            'weights': frontier_weights
        }
    
    def plot_efficient_frontier(self, save_path: Optional[str] = None) -> None:
        """
        Plot the efficient frontier.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.efficient_frontier is None:
            logger.warning("Efficient frontier not generated yet")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot efficient frontier
        plt.plot(
            self.efficient_frontier['risks'],
            self.efficient_frontier['returns'],
            'b-', linewidth=2, label='Efficient Frontier'
        )
        
        # Plot current portfolio
        if self.weights is not None:
            plt.scatter(
                self.expected_volatility,
                self.expected_return,
                marker='*', s=100, color='r', label='Optimal Portfolio'
            )
        
        # Plot risk-free rate
        plt.axhline(y=self.risk_free_rate, color='g', linestyle='--', label=f'Risk-Free Rate ({self.risk_free_rate:.2%})')
        
        # Plot individual assets
        for i, asset in enumerate(self.asset_names):
            plt.scatter(
                np.sqrt(self.covariance_matrix[i, i]),
                self.returns[i],
                marker='o', s=50, label=asset
            )
        
        # Add labels and title
        plt.xlabel('Expected Volatility (Annualized)')
        plt.ylabel('Expected Return (Annualized)')
        plt.title('Efficient Frontier')
        
        # Add legend (limit to 10 items for readability)
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 12:
            plt.legend(handles[:12], labels[:12], loc='best')
        else:
            plt.legend(loc='best')
        
        # Format axes as percentages
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Efficient frontier plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_weights(self, save_path: Optional[str] = None) -> None:
        """
        Plot portfolio weights.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.weights is None:
            logger.warning("Portfolio weights not available")
            return
        
        # Filter out assets with very small weights for readability
        threshold = 0.005  # 0.5%
        significant_weights = {asset: weight for asset, weight in zip(self.asset_names, self.weights) if weight > threshold}
        
        # Sort by weight
        sorted_weights = sorted(significant_weights.items(), key=lambda x: x[1], reverse=True)
        assets, weights = zip(*sorted_weights) if sorted_weights else ([], [])
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(assets, weights)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.005,
                f'{height:.1%}',
                ha='center', va='bottom', rotation=0
            )
        
        # Add labels and title
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.title('Portfolio Weights')
        
        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Portfolio weights plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_sector_allocation(self, save_path: Optional[str] = None) -> None:
        """
        Plot sector allocation.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.weights is None or self.sector_mapping is None:
            logger.warning("Portfolio weights or sector mapping not available")
            return
        
        # Calculate sector weights
        sector_weights = {}
        for i, asset in enumerate(self.asset_names):
            sector = self.sector_mapping.get(asset, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + self.weights[i]
        
        # Sort by weight
        sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        sectors, weights = zip(*sorted_sectors)
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(
            weights,
            labels=sectors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            explode=[0.05] * len(sectors)
        )
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Portfolio Sector Allocation')
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Sector allocation plot saved to {save_path}")
        else:
            plt.show()
    
    def save(self, path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Boolean indicating success
        """
        try:
            model_data = {
                'config': self.config,
                'weights': self.weights,
                'expected_return': self.expected_return,
                'expected_volatility': self.expected_volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'efficient_frontier': self.efficient_frontier,
                'asset_names': self.asset_names,
                'sector_mapping': self.sector_mapping,
                'returns': self.returns,
                'covariance_matrix': self.covariance_matrix
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.weights = model_data['weights']
            self.expected_return = model_data['expected_return']
            self.expected_volatility = model_data['expected_volatility']
            self.sharpe_ratio = model_data['sharpe_ratio']
            self.efficient_frontier = model_data['efficient_frontier']
            self.asset_names = model_data['asset_names']
            self.sector_mapping = model_data['sector_mapping']
            self.returns = model_data['returns']
            self.covariance_matrix = model_data['covariance_matrix']
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage (requires data)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Dummy data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    data = []
    for symbol in symbols:
        prices = np.random.rand(len(dates)) * 100 + 50
        symbol_data = pd.DataFrame({'timestamp': dates, 'symbol': symbol, 'close': prices})
        data.append(symbol_data)
        
    price_data = pd.concat(data, ignore_index=True)
    
    # Calculate returns
    returns_data = price_data.pivot_table(index='timestamp', columns='symbol', values='close').pct_change().dropna()
    
    # Dummy sector data
    sector_data = pd.DataFrame({
        'symbol': symbols,
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary']
    }).set_index('symbol')
    
    config = {
        'optimization_method': 'efficient_frontier',
        'risk_free_rate': 0.02,
        'max_position_size': 0.2,
        'min_position_size': 0.01,
        'max_sector_allocation': 0.3,
        'min_stocks': 3,
        'max_stocks': 5
    }
    
    model = PortfolioOptimizationModel(config)
    results = model.optimize(returns_data, sector_data)
    
    if 'error' not in results:
        print("Optimization Results:")
        for asset, weight in results['weights'].items():
            print(f"{asset}: {weight:.2%}")
        print(f"Expected Annual Return: {results['expected_annual_return']:.2%}")
        print(f"Expected Annual Volatility: {results['expected_annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        
        # Generate and save plots
        model.plot_efficient_frontier('efficient_frontier.png')
        model.plot_weights('portfolio_weights.png')
        model.plot_sector_allocation('sector_allocation.png')
