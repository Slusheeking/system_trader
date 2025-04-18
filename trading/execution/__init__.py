"""
Trading execution package for system_trader.

This package contains modules related to trade execution functionality.
"""

from trading.execution.execution_monitor import get_execution_monitor, ExecutionMonitor
from trading.execution.circuit_breaker import get_circuit_breaker, CircuitBreaker
from trading.execution.order_router import OrderRouter

__all__ = [
    'get_execution_monitor',
    'ExecutionMonitor',
    'get_circuit_breaker',
    'CircuitBreaker',
    'OrderRouter'
]
