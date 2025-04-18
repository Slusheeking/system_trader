#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging Module
-------------
This module provides centralized logging functionality for the system trader application.
"""

import logging
import os
import sys
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from utils.config_loader import load_config

# Load system configuration
try:
    system_config = load_config('config/system_config.yaml')
    logging_config = system_config.get('logging', {})
except Exception:
    # Default configuration if loading fails
    logging_config = {
        'level': 'INFO',
        'console': True,
        'component_files': True,
        'central_file': True,
        'structured': True,
        'central_log_level': 'WARNING',
        'categories': {
            'system': {'file': 'system.log', 'level': 'INFO'},
            'model': {'file': 'model.log', 'level': 'INFO'},
            'data': {'file': 'data.log', 'level': 'INFO'},
            'trading': {'file': 'trading.log', 'level': 'INFO'},
            'default': {'file': 'system_trader.log', 'level': 'INFO'}
        },
        'retention': {
            'days': 30,
            'compress_after_days': 7,
            'archive_after_days': 30
        }
    }

# Convert string log level to logging constant
def get_log_level(level_str):
    """Convert string log level to logging constant."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def setup_logger(name, log_level=None, log_to_console=None, log_to_file=None, 
                log_dir='logs', max_file_size=10*1024*1024, backup_count=30, 
                category=None):
    """
    Set up and configure a logger with the given name.
    
    Args:
        name (str): Name of the logger
        log_level (int/str): Logging level (default: from config)
        log_to_console (bool): Whether to log to console (default: from config)
        log_to_file (bool): Whether to log to category-specific file (default: from config)
        log_dir (str): Directory to store log files (default: 'logs')
        max_file_size (int): Maximum size of log file before rotation in bytes (default: 10MB)
        backup_count (int): Number of backup files to keep (default: 30)
        category (str): Log category (system, model, data, trading, default)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Use config values if parameters are None
    if log_level is None:
        log_level = logging_config.get('level', 'INFO')
    if log_to_console is None:
        log_to_console = logging_config.get('console', True)
    if log_to_file is None:
        log_to_file = logging_config.get('component_files', True)
    
    # Use default category if not specified
    if category is None:
        category = 'default'
    
    # Get category config
    categories = logging_config.get('categories', {})
    category_config = categories.get(category, categories.get('default', {'file': 'system_trader.log', 'level': 'INFO'}))
    
    # Get category-specific log level if available
    if log_level is None and 'level' in category_config:
        log_level = category_config.get('level', 'INFO')
    
    # Convert string log level to logging constant if needed
    if isinstance(log_level, str):
        log_level = get_log_level(log_level)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Category-specific file handler
    if log_to_file:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Get category log file name
        category_log_file = category_config.get('file', f"{category}.log")
        
        # Create log file path
        log_file = os.path.join(log_dir, category_log_file)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent messages from propagating to the root logger
    logger.propagate = False
    
    return logger

def log_structured(logger, level, message, **kwargs):
    """
    Log a structured message with additional context.
    
    Args:
        logger: Logger instance
        level: Log level (e.g., logging.INFO)
        message: Log message
        **kwargs: Additional context fields
    """
    # Only add structured logging if enabled in config
    if logging_config.get('structured', True) and kwargs:
        # Format additional context as JSON
        try:
            context = json.dumps(kwargs)
            # Log with context
            logger.log(level, f"{message} | {context}")
        except Exception as e:
            # Fall back to regular logging if JSON serialization fails
            logger.log(level, message)
            logger.warning(f"Failed to serialize structured log data: {str(e)}")
    else:
        # Regular logging
        logger.log(level, message)

def setup_log_directories():
    """
    Set up the log directory structure.
    """
    base_dir = 'logs'
    
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    return True

# Set up log directories when module is imported
setup_log_directories()
