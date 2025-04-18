#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Logging Script
-------------------
This script sets up the log directory structure and demonstrates the new logging capabilities.
"""

import os
import logging
import argparse
import shutil
from datetime import datetime, timedelta
import gzip
import time

from utils.logging import setup_logger, log_structured, setup_log_directories

def create_log_structure(force_recreate=False):
    """
    Create the log directory structure.
    
    Args:
        force_recreate (bool): Whether to force recreation of the structure
    
    Returns:
        bool: Success status
    """
    base_dir = 'logs'
    
    # If force_recreate is True, remove existing log directory
    if force_recreate and os.path.exists(base_dir):
        print(f"Removing existing log directory: {base_dir}")
        shutil.rmtree(base_dir)
    
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base log directory: {base_dir}")
    
    # Create category directories
    categories = ['system', 'models', 'data', 'trading', 'errors']
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            print(f"Created category directory: {category_dir}")
    
    return True

def setup_log_rotation():
    """
    Set up log rotation for existing log files.
    
    Returns:
        bool: Success status
    """
    base_dir = 'logs'
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Log directory does not exist: {base_dir}")
        return False
    
    # Get all log files
    log_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    
    print(f"Found {len(log_files)} log files")
    
    # Current time for comparison
    now = datetime.now()
    
    # Process each log file
    for log_file in log_files:
        try:
            # Get file stats
            stats = os.stat(log_file)
            modified_time = datetime.fromtimestamp(stats.st_mtime)
            
            # Calculate age in days
            age_days = (now - modified_time).days
            
            # Compress files older than 7 days
            if age_days >= 7:
                # Check if already compressed
                if not log_file.endswith('.gz'):
                    print(f"Compressing log file: {log_file}")
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    # Remove original file
                    os.remove(log_file)
            
            # Archive files older than 30 days (move to archive directory)
            if age_days >= 30:
                archive_dir = os.path.join(base_dir, 'archive')
                if not os.path.exists(archive_dir):
                    os.makedirs(archive_dir)
                
                # Move file to archive
                archive_file = os.path.join(archive_dir, os.path.basename(log_file))
                print(f"Archiving log file: {log_file} -> {archive_file}")
                shutil.move(log_file, archive_file)
        
        except Exception as e:
            print(f"Error processing log file {log_file}: {str(e)}")
    
    return True

def demonstrate_logging():
    """
    Demonstrate the new logging capabilities.
    
    Returns:
        bool: Success status
    """
    # Create loggers for different categories
    system_logger = setup_logger('system_demo', category='system')
    model_logger = setup_logger('model_demo', category='models')
    data_logger = setup_logger('data_demo', category='data')
    trading_logger = setup_logger('trading_demo', category='trading')
    error_logger = setup_logger('error_demo', category='errors')
    
    # Log messages at different levels
    system_logger.info("System startup initiated")
    system_logger.warning("System resource usage high")
    
    model_logger.info("Model training started")
    model_logger.debug("Model hyperparameters: learning_rate=0.001, batch_size=64")
    
    data_logger.info("Data collection started")
    data_logger.error("Failed to connect to data source")
    
    trading_logger.info("Trading session started")
    trading_logger.warning("Order execution delayed")
    
    error_logger.error("Critical error in component X")
    error_logger.critical("System shutdown required")
    
    # Demonstrate structured logging
    log_structured(
        system_logger, 
        logging.INFO, 
        "System configuration loaded",
        config_file="system_config.yaml",
        components=["trading", "data", "models"],
        startup_time=datetime.now().isoformat()
    )
    
    log_structured(
        model_logger,
        logging.INFO,
        "Model evaluation completed",
        model_name="stock_selection",
        accuracy=0.87,
        precision=0.82,
        recall=0.79,
        f1_score=0.80,
        training_time=120.5
    )
    
    log_structured(
        trading_logger,
        logging.INFO,
        "Order executed",
        order_id="ORD123456",
        symbol="AAPL",
        quantity=100,
        price=185.75,
        side="buy",
        execution_time=0.35
    )
    
    log_structured(
        error_logger,
        logging.ERROR,
        "Database connection failed",
        db_host="localhost",
        db_port=5432,
        error_code="CONN_REFUSED",
        retry_count=3,
        backoff_seconds=5
    )
    
    print("Logging demonstration completed. Check the logs directory for output.")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup logging for the trading system')
    parser.add_argument('--recreate', action='store_true', help='Force recreation of log directories')
    parser.add_argument('--rotate', action='store_true', help='Set up log rotation')
    parser.add_argument('--demo', action='store_true', help='Demonstrate logging capabilities')
    
    args = parser.parse_args()
    
    # Create log structure
    if args.recreate:
        create_log_structure(force_recreate=True)
    else:
        setup_log_directories()
    
    # Set up log rotation
    if args.rotate:
        setup_log_rotation()
    
    # Demonstrate logging
    if args.demo:
        demonstrate_logging()
    
    # If no arguments provided, show help
    if not (args.recreate or args.rotate or args.demo):
        parser.print_help()

if __name__ == '__main__':
    main()
