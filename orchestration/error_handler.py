#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Error Handler
------------
Centralized error handling system for the trading platform.
Provides consistent error handling, logging, and recovery strategies.
"""

import logging
import traceback
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Type, Tuple
from enum import Enum
import threading
import queue
from functools import wraps

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger, log_structured

# Setup logging with error category
logger = setup_logger('error_handler', category='errors')


class ErrorSeverity(Enum):
    """
    Enumeration of error severity levels.
    """
    INFO = 0       # Informational, no action needed
    WARNING = 1    # Warning, may need attention
    ERROR = 2      # Error, needs attention
    CRITICAL = 3   # Critical, immediate attention needed
    FATAL = 4      # Fatal, system shutdown required


class ErrorCategory(Enum):
    """
    Enumeration of error categories.
    """
    DATA_COLLECTION = 'data_collection'
    DATA_PROCESSING = 'data_processing'
    MODEL_PREDICTION = 'model_prediction'
    TRADING_EXECUTION = 'trading_execution'
    SYSTEM = 'system'
    NETWORK = 'network'
    DATABASE = 'database'
    AUTHENTICATION = 'authentication'
    CONFIGURATION = 'configuration'
    UNKNOWN = 'unknown'


class ErrorContext:
    """
    Context information for an error.
    """
    
    def __init__(self, component: str, operation: str, inputs: Optional[Dict[str, Any]] = None,
               timestamp: Optional[datetime] = None):
        """
        Initialize error context.
        
        Args:
            component: Component where the error occurred
            operation: Operation being performed
            inputs: Inputs to the operation
            timestamp: Time when the error occurred
        """
        self.component = component
        self.operation = operation
        self.inputs = inputs or {}
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'component': self.component,
            'operation': self.operation,
            'inputs': {k: str(v) for k, v in self.inputs.items()},  # Convert all values to strings
            'timestamp': self.timestamp.isoformat()
        }


class ErrorRecord:
    """
    Record of an error.
    """
    
    def __init__(self, error: Exception, context: ErrorContext, 
               severity: ErrorSeverity, category: ErrorCategory,
               traceback_str: Optional[str] = None, 
               recovery_attempted: bool = False,
               recovery_successful: bool = False):
        """
        Initialize error record.
        
        Args:
            error: Exception that occurred
            context: Context information
            severity: Error severity
            category: Error category
            traceback_str: Traceback string
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery was successful
        """
        self.error = error
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.context = context
        self.severity = severity
        self.category = category
        self.traceback_str = traceback_str or ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        self.recovery_attempted = recovery_attempted
        self.recovery_successful = recovery_successful
        self.timestamp = datetime.now()
        self.id = f"{self.timestamp.strftime('%Y%m%d%H%M%S')}_{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'context': self.context.to_dict(),
            'severity': self.severity.name,
            'category': self.category.value,
            'traceback': self.traceback_str,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'timestamp': self.timestamp.isoformat()
        }


class RecoveryStrategy:
    """
    Strategy for recovering from an error.
    """
    
    def __init__(self, name: str, strategy_func: Callable[[ErrorRecord], bool], 
               applicable_errors: List[Type[Exception]], max_attempts: int = 3,
               cooldown_seconds: int = 5):
        """
        Initialize recovery strategy.
        
        Args:
            name: Strategy name
            strategy_func: Function to execute for recovery
            applicable_errors: List of exception types this strategy applies to
            max_attempts: Maximum number of recovery attempts
            cooldown_seconds: Seconds to wait between attempts
        """
        self.name = name
        self.strategy_func = strategy_func
        self.applicable_errors = applicable_errors
        self.max_attempts = max_attempts
        self.cooldown_seconds = cooldown_seconds
    
    def applies_to(self, error: Exception) -> bool:
        """
        Check if this strategy applies to the given error.
        
        Args:
            error: Exception to check
            
        Returns:
            Boolean indicating if strategy applies
        """
        return any(isinstance(error, err_type) for err_type in self.applicable_errors)
    
    def execute(self, error_record: ErrorRecord) -> bool:
        """
        Execute the recovery strategy.
        
        Args:
            error_record: Error record to recover from
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Executing recovery strategy '{self.name}' for error {error_record.id}")
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(f"Recovery attempt {attempt}/{self.max_attempts}")
                success = self.strategy_func(error_record)
                
                if success:
                    logger.info(f"Recovery strategy '{self.name}' succeeded on attempt {attempt}")
                    return True
                
                logger.warning(f"Recovery strategy '{self.name}' failed on attempt {attempt}")
                
                if attempt < self.max_attempts:
                    logger.info(f"Waiting {self.cooldown_seconds} seconds before next attempt")
                    time.sleep(self.cooldown_seconds)
            
            except Exception as e:
                logger.error(f"Error executing recovery strategy '{self.name}': {str(e)}")
                if attempt < self.max_attempts:
                    logger.info(f"Waiting {self.cooldown_seconds} seconds before next attempt")
                    time.sleep(self.cooldown_seconds)
        
        logger.error(f"Recovery strategy '{self.name}' failed after {self.max_attempts} attempts")
        return False


class ErrorHandler:
    """
    Centralized error handling system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize error handler.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Error records
        self.error_records: List[ErrorRecord] = []
        self.max_records = self.config.get('max_records', 1000)
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        
        # Error notification
        self.notification_threshold = self.config.get('notification_threshold', ErrorSeverity.ERROR)
        self.notification_cooldown = self.config.get('notification_cooldown', 300)  # 5 minutes
        self.last_notification_time: Dict[str, datetime] = {}
        
        # Error handling queue
        self.error_queue = queue.Queue()
        self.error_thread = threading.Thread(target=self._process_error_queue, daemon=True)
        self.error_thread.start()
        
        # Initialize recovery strategies
        self._init_recovery_strategies()
        
        logger.info("Error handler initialized")
    
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
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _init_recovery_strategies(self):
        """
        Initialize recovery strategies.
        """
        # Add default recovery strategies
        
        # Network retry strategy
        self.add_recovery_strategy(
            RecoveryStrategy(
                name="network_retry",
                strategy_func=self._network_retry_strategy,
                applicable_errors=[
                    ConnectionError, 
                    TimeoutError, 
                    ConnectionRefusedError,
                    ConnectionResetError
                ],
                max_attempts=3,
                cooldown_seconds=5
            )
        )
        
        # Database reconnect strategy
        self.add_recovery_strategy(
            RecoveryStrategy(
                name="db_reconnect",
                strategy_func=self._db_reconnect_strategy,
                applicable_errors=[
                    Exception  # Replace with specific database exceptions
                ],
                max_attempts=3,
                cooldown_seconds=5
            )
        )
        
        # Data revalidation strategy
        self.add_recovery_strategy(
            RecoveryStrategy(
                name="data_revalidation",
                strategy_func=self._data_revalidation_strategy,
                applicable_errors=[
                    ValueError, 
                    TypeError,
                    KeyError,
                    IndexError
                ],
                max_attempts=2,
                cooldown_seconds=1
            )
        )
        
        # Model fallback strategy
        self.add_recovery_strategy(
            RecoveryStrategy(
                name="model_fallback",
                strategy_func=self._model_fallback_strategy,
                applicable_errors=[
                    Exception  # Replace with specific model exceptions
                ],
                max_attempts=1,
                cooldown_seconds=0
            )
        )
    
    def _network_retry_strategy(self, error_record: ErrorRecord) -> bool:
        """
        Network retry recovery strategy.
        
        Args:
            error_record: Error record
            
        Returns:
            Boolean indicating success
        """
        # This would implement logic to retry network operations
        logger.info(f"Executing network retry strategy for {error_record.error_type}")
        
        # Example implementation (placeholder)
        # In a real implementation, you would extract the operation from the context
        # and retry it with appropriate parameters
        
        # Simulate success/failure
        import random
        success = random.random() > 0.3  # 70% success rate
        
        return success
    
    def _db_reconnect_strategy(self, error_record: ErrorRecord) -> bool:
        """
        Database reconnect recovery strategy.
        
        Args:
            error_record: Error record
            
        Returns:
            Boolean indicating success
        """
        # This would implement logic to reconnect to the database
        logger.info(f"Executing database reconnect strategy for {error_record.error_type}")
        
        # Example implementation (placeholder)
        try:
            # Get database client based on context
            component = error_record.context.component
            
            if 'timeseries_db' in component:
                from data.database.timeseries_db import TimeseriesDBClient
                client = TimeseriesDBClient()
                # Test connection
                client.ping()
                return True
            
            elif 'redis' in component:
                from data.database.redis_client import RedisClient
                client = RedisClient()
                # Test connection
                client.ping()
                return True
            
            else:
                logger.warning(f"Unknown database component: {component}")
                return False
        
        except Exception as e:
            logger.error(f"Error in database reconnect strategy: {str(e)}")
            return False
    
    def _data_revalidation_strategy(self, error_record: ErrorRecord) -> bool:
        """
        Data revalidation recovery strategy.
        
        Args:
            error_record: Error record
            
        Returns:
            Boolean indicating success
        """
        # This would implement logic to revalidate and clean data
        logger.info(f"Executing data revalidation strategy for {error_record.error_type}")
        
        # Example implementation (placeholder)
        try:
            # Get data from context
            inputs = error_record.context.inputs
            
            if 'data' in inputs:
                # In a real implementation, you would clean and validate the data
                # For now, just simulate success/failure
                import random
                success = random.random() > 0.2  # 80% success rate
                return success
            
            else:
                logger.warning("No data found in error context")
                return False
        
        except Exception as e:
            logger.error(f"Error in data revalidation strategy: {str(e)}")
            return False
    
    def _model_fallback_strategy(self, error_record: ErrorRecord) -> bool:
        """
        Model fallback recovery strategy.
        
        Args:
            error_record: Error record
            
        Returns:
            Boolean indicating success
        """
        # This would implement logic to fall back to a simpler model
        logger.info(f"Executing model fallback strategy for {error_record.error_type}")
        
        # Example implementation (placeholder)
        try:
            # Get model information from context
            component = error_record.context.component
            
            if 'model' in component:
                # In a real implementation, you would load a fallback model
                # For now, just simulate success/failure
                import random
                success = random.random() > 0.1  # 90% success rate
                return success
            
            else:
                logger.warning(f"Not a model component: {component}")
                return False
        
        except Exception as e:
            logger.error(f"Error in model fallback strategy: {str(e)}")
            return False
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """
        Add a recovery strategy.
        
        Args:
            strategy: Recovery strategy to add
        """
        self.recovery_strategies.append(strategy)
        logger.info(f"Added recovery strategy: {strategy.name}")
    
    def handle_error(self, error: Exception, context: ErrorContext, 
                   severity: ErrorSeverity, category: ErrorCategory,
                   attempt_recovery: bool = True) -> ErrorRecord:
        """
        Handle an error.
        
        Args:
            error: Exception that occurred
            context: Context information
            severity: Error severity
            category: Error category
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Error record
        """
        # Create error record
        error_record = ErrorRecord(
            error=error,
            context=context,
            severity=severity,
            category=category
        )
        
        # Add to queue for processing
        self.error_queue.put((error_record, attempt_recovery))
        
        return error_record
    
    def _process_error_queue(self):
        """
        Process errors from the queue.
        """
        while True:
            try:
                # Get error from queue
                error_record, attempt_recovery = self.error_queue.get()
                
                # Log error
                self._log_error(error_record)
                
                # Store error record
                self._store_error_record(error_record)
                
                # Send notification if needed
                self._send_notification(error_record)
                
                # Attempt recovery if requested
                if attempt_recovery:
                    self._attempt_recovery(error_record)
                
                # Mark task as done
                self.error_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error processing error queue: {str(e)}")
                time.sleep(1)  # Avoid tight loop if there's a persistent error
    
    def _log_error(self, error_record: ErrorRecord):
        """
        Log an error.
        
        Args:
            error_record: Error record to log
        """
        # Determine log level based on severity
        if error_record.severity == ErrorSeverity.INFO:
            log_func = logger.info
        elif error_record.severity == ErrorSeverity.WARNING:
            log_func = logger.warning
        elif error_record.severity == ErrorSeverity.ERROR:
            log_func = logger.error
        else:  # CRITICAL or FATAL
            log_func = logger.critical
        
        # Get the appropriate log level
        if error_record.severity == ErrorSeverity.INFO:
            level = logging.INFO
        elif error_record.severity == ErrorSeverity.WARNING:
            level = logging.WARNING
        elif error_record.severity == ErrorSeverity.ERROR:
            level = logging.ERROR
        else:  # CRITICAL or FATAL
            level = logging.CRITICAL
            
        # Use structured logging
        log_structured(
            logger,
            level,
            f"[{error_record.category.value}] {error_record.error_type}: {error_record.error_message} in {error_record.context.component}.{error_record.context.operation}",
            error_id=error_record.id,
            error_type=error_record.error_type,
            error_message=error_record.error_message,
            component=error_record.context.component,
            operation=error_record.context.operation,
            category=error_record.category.value,
            severity=error_record.severity.name,
            timestamp=error_record.timestamp.isoformat()
        )
        
        # Log traceback for ERROR and above
        if error_record.severity.value >= ErrorSeverity.ERROR.value:
            logger.error(f"Traceback:\n{error_record.traceback_str}")
    
    def _store_error_record(self, error_record: ErrorRecord):
        """
        Store an error record.
        
        Args:
            error_record: Error record to store
        """
        # Add to in-memory list
        self.error_records.append(error_record)
        
        # Trim list if needed
        if len(self.error_records) > self.max_records:
            self.error_records = self.error_records[-self.max_records:]
        
        # Store to persistent storage if configured
        if self.config.get('persistent_storage', False):
            self._store_to_persistent_storage(error_record)
    
    def _store_to_persistent_storage(self, error_record: ErrorRecord):
        """
        Store an error record to persistent storage.
        
        Args:
            error_record: Error record to store
        """
        # This would implement logic to store to a database or file
        storage_type = self.config.get('storage_type', 'file')
        
        if storage_type == 'file':
            # Store to JSON file
            try:
                storage_dir = self.config.get('storage_dir', 'logs/errors')
                
                # Create directory if it doesn't exist
                os.makedirs(storage_dir, exist_ok=True)
                
                # Create errors subdirectory for the category
                category_dir = os.path.join(storage_dir, error_record.category.value)
                os.makedirs(category_dir, exist_ok=True)
                
                # Create filename based on timestamp and ID
                filename = f"{error_record.timestamp.strftime('%Y%m%d')}_{error_record.category.value}.json"
                filepath = os.path.join(category_dir, filename)
                
                # Append to file
                with open(filepath, 'a') as f:
                    f.write(json.dumps(error_record.to_dict()) + '\n')
            
            except Exception as e:
                logger.error(f"Error storing to file: {str(e)}")
        
        elif storage_type == 'database':
            # Store to database
            try:
                # This would implement database storage
                pass
            
            except Exception as e:
                logger.error(f"Error storing to database: {str(e)}")
    
    def _send_notification(self, error_record: ErrorRecord):
        """
        Send a notification for an error.
        
        Args:
            error_record: Error record to notify about
        """
        # Check if notification is needed based on severity
        if error_record.severity.value < self.notification_threshold.value:
            return
        
        # Check cooldown
        category_key = f"{error_record.category.value}_{error_record.error_type}"
        last_time = self.last_notification_time.get(category_key)
        
        if last_time and (datetime.now() - last_time).total_seconds() < self.notification_cooldown:
            logger.debug(f"Skipping notification for {category_key} due to cooldown")
            return
        
        # Update last notification time
        self.last_notification_time[category_key] = datetime.now()
        
        # Send notification
        notification_type = self.config.get('notification_type', 'log')
        
        if notification_type == 'log':
            # Just log the notification
            logger.warning(
                f"NOTIFICATION: {error_record.severity.name} {error_record.category.value} error: "
                f"{error_record.error_type} in {error_record.context.component}"
            )
        
        elif notification_type == 'email':
            # Send email notification
            self._send_email_notification(error_record)
        
        elif notification_type == 'slack':
            # Send Slack notification
            self._send_slack_notification(error_record)
    
    def _send_email_notification(self, error_record: ErrorRecord):
        """
        Send an email notification.
        
        Args:
            error_record: Error record to notify about
        """
        # This would implement email notification
        logger.info(f"Would send email notification for {error_record.id}")
    
    def _send_slack_notification(self, error_record: ErrorRecord):
        """
        Send a Slack notification.
        
        Args:
            error_record: Error record to notify about
        """
        # This would implement Slack notification
        logger.info(f"Would send Slack notification for {error_record.id}")
    
    def _attempt_recovery(self, error_record: ErrorRecord):
        """
        Attempt to recover from an error.
        
        Args:
            error_record: Error record to recover from
        """
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if strategy.applies_to(error_record.error)
        ]
        
        if not applicable_strategies:
            logger.info(f"No applicable recovery strategies for {error_record.error_type}")
            return
        
        # Update error record
        error_record.recovery_attempted = True
        
        # Try each strategy in order
        for strategy in applicable_strategies:
            logger.info(f"Attempting recovery with strategy: {strategy.name}")
            
            try:
                success = strategy.execute(error_record)
                
                if success:
                    logger.info(f"Recovery successful with strategy: {strategy.name}")
                    error_record.recovery_successful = True
                    break
                
                logger.warning(f"Recovery failed with strategy: {strategy.name}")
            
            except Exception as e:
                logger.error(f"Error executing recovery strategy {strategy.name}: {str(e)}")
        
        # Log final recovery status
        if error_record.recovery_successful:
            logger.info(f"Successfully recovered from error {error_record.id}")
        else:
            logger.warning(f"Failed to recover from error {error_record.id}")
    
    def get_error_records(self, severity: Optional[ErrorSeverity] = None, 
                        category: Optional[ErrorCategory] = None,
                        component: Optional[str] = None,
                        limit: int = 100) -> List[ErrorRecord]:
        """
        Get error records matching the given criteria.
        
        Args:
            severity: Filter by severity
            category: Filter by category
            component: Filter by component
            limit: Maximum number of records to return
            
        Returns:
            List of matching error records
        """
        # Filter records
        filtered_records = self.error_records
        
        if severity is not None:
            filtered_records = [r for r in filtered_records if r.severity == severity]
        
        if category is not None:
            filtered_records = [r for r in filtered_records if r.category == category]
        
        if component is not None:
            filtered_records = [r for r in filtered_records if r.context.component == component]
        
        # Sort by timestamp (newest first) and limit
        sorted_records = sorted(filtered_records, key=lambda r: r.timestamp, reverse=True)
        
        return sorted_records[:limit]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of errors.
        
        Returns:
            Dictionary with error summary
        """
        # Count errors by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.name] = len([r for r in self.error_records if r.severity == severity])
        
        # Count errors by category
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = len([r for r in self.error_records if r.category == category])
        
        # Count errors by component
        component_counts = {}
        for record in self.error_records:
            component = record.context.component
            component_counts[component] = component_counts.get(component, 0) + 1
        
        # Count recovery attempts and successes
        recovery_attempted = len([r for r in self.error_records if r.recovery_attempted])
        recovery_successful = len([r for r in self.error_records if r.recovery_successful])
        
        # Calculate recovery success rate
        recovery_rate = recovery_successful / recovery_attempted if recovery_attempted > 0 else 0
        
        # Get recent errors
        recent_errors = [
            {
                'id': r.id,
                'error_type': r.error_type,
                'error_message': r.error_message,
                'component': r.context.component,
                'severity': r.severity.name,
                'category': r.category.value,
                'timestamp': r.timestamp.isoformat(),
                'recovery_successful': r.recovery_successful
            }
            for r in sorted(self.error_records, key=lambda r: r.timestamp, reverse=True)[:10]
        ]
        
        return {
            'total_errors': len(self.error_records),
            'severity_counts': severity_counts,
            'category_counts': category_counts,
            'component_counts': component_counts,
            'recovery_attempted': recovery_attempted,
            'recovery_successful': recovery_successful,
            'recovery_rate': recovery_rate,
            'recent_errors': recent_errors
        }


def handle_errors(component: str, operation: str, severity: ErrorSeverity = ErrorSeverity.ERROR,
                category: ErrorCategory = ErrorCategory.UNKNOWN, attempt_recovery: bool = True):
    """
    Decorator for handling errors in a function.
    
    Args:
        component: Component name
        operation: Operation name
        severity: Error severity
        category: Error category
        attempt_recovery: Whether to attempt recovery
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    inputs={'args': args, 'kwargs': kwargs}
                )
                
                # Handle error
                error_handler = get_error_handler()
                error_record = error_handler.handle_error(
                    error=e,
                    context=context,
                    severity=severity,
                    category=category,
                    attempt_recovery=attempt_recovery
                )
                
                # Re-raise if recovery failed or wasn't attempted
                if not error_record.recovery_successful:
                    raise
                
                # Return None if recovery was successful
                return None
        
        return wrapper
    
    return decorator


# Default error handler instance
default_error_handler = None


def get_error_handler(config_path: Optional[str] = None) -> ErrorHandler:
    """
    Get or create the default error handler.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ErrorHandler instance
    """
    global default_error_handler
    
    if default_error_handler is None:
        default_error_handler = ErrorHandler(config_path)
    
    return default_error_handler


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Error Handler for Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--summary', action='store_true', help='Print error summary')
    parser.add_argument('--test', action='store_true', help='Run a test error')
    
    args = parser.parse_args()
    
    # Create error handler
    error_handler = ErrorHandler(args.config)
    
    if args.test:
        # Create a test error
        try:
            # Simulate an error
            raise ValueError("This is a test error")
        except Exception as e:
            # Handle the error
            context = ErrorContext(
                component="error_handler",
                operation="test",
                inputs={"test": True}
            )
            
            error_handler.handle_error(
                error=e,
                context=context,
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.SYSTEM,
                attempt_recovery=True
            )
        
        print("Test error created")
    
    if args.summary:
        # Print error summary
        summary = error_handler.get_error_summary()
        print(json.dumps(summary, indent=2))
