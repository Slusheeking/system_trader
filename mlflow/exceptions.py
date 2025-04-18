"""
MLflow Exceptions Module for Day Trading System

This module provides mock implementations of MLflow exceptions
for the day trading system.
"""

class MlflowException(Exception):
    """
    Mock implementation of MlflowException.
    
    This exception is raised when MLflow encounters an error.
    """
    def __init__(self, message, error_code=None, **kwargs):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __str__(self):
        return self.message