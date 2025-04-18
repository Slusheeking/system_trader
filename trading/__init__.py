"""
Trading package for system_trader.

This package contains modules related to trading operations and strategies.
"""

from trading.strategy import StrategyComposer, TradingStrategy
from trading.manager import TradingManager

__all__ = [
    'StrategyComposer',
    'TradingStrategy',
    'TradingManager',
]
