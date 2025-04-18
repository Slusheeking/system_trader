#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging Example
--------------
This example demonstrates the usage of the centralized logging system
with different categories and log levels.
"""

import time
import random
from datetime import datetime, timedelta
import traceback
from utils.logging import setup_logger

# Create loggers for different categories
data_logger = setup_logger('data_example', category='data')
model_logger = setup_logger('model_example', category='model')
trading_logger = setup_logger('trading_example', category='trading')
system_logger = setup_logger('system_example', category='system')
default_logger = setup_logger('default_example')  # Uses default category

def simulate_data_collection():
    """Simulate data collection process with logging."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    data_logger.info(f"Starting data collection for {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            # Simulate data collection
            data_logger.debug(f"Collecting data for {symbol}")
            
            # Simulate random processing time
            process_time = random.uniform(0.1, 0.5)
            time.sleep(process_time)
            
            # Simulate random data points
            data_points = random.randint(100, 500)
            
            # Log successful collection
            data_logger.info(f"Collected {data_points} data points for {symbol} in {process_time:.2f}s")
            
            # Simulate occasional warning
            if random.random() < 0.3:
                data_logger.warning(f"Missing some historical data for {symbol}")
                
        except Exception as e:
            # Log error with exception info
            data_logger.error(f"Error collecting data for {symbol}: {str(e)}", exc_info=True)
    
    data_logger.info("Data collection completed")

def simulate_model_training():
    """Simulate model training process with logging."""
    model_types = ['stock_selection', 'entry_timing', 'market_regime']
    
    for model_type in model_types:
        model_logger.info(f"Starting training for {model_type} model")
        
        try:
            # Simulate model training
            epochs = random.randint(50, 200)
            model_logger.debug(f"Training {model_type} model with {epochs} epochs")
            
            # Simulate training progress
            for epoch in range(1, epochs + 1, 10):
                loss = random.uniform(0.01, 1.0) * (1 - epoch/epochs)
                accuracy = random.uniform(0.7, 0.99)
                model_logger.debug(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}")
                
                # Simulate occasional warning
                if random.random() < 0.1:
                    model_logger.warning(f"High validation loss detected in epoch {epoch}")
            
            # Log model metrics
            final_accuracy = random.uniform(0.85, 0.98)
            model_logger.info(f"Model {model_type} trained successfully: accuracy={final_accuracy:.4f}")
            
            # Simulate model saving
            model_logger.info(f"Saving {model_type} model to model registry")
            
        except Exception as e:
            # Log error with exception info
            model_logger.error(f"Error training {model_type} model: {str(e)}", exc_info=True)
    
    model_logger.info("Model training completed")

def simulate_trading():
    """Simulate trading process with logging."""
    strategies = ['momentum', 'mean_reversion', 'trend_following']
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    trading_logger.info(f"Starting trading simulation with {len(strategies)} strategies")
    
    for strategy in strategies:
        trading_logger.info(f"Executing {strategy} strategy")
        
        for symbol in symbols:
            try:
                # Simulate signal generation
                signal = random.choice(['BUY', 'SELL', 'HOLD'])
                confidence = random.uniform(0.5, 1.0)
                
                trading_logger.debug(f"Generated {signal} signal for {symbol} with confidence {confidence:.2f}")
                
                # Simulate order execution
                if signal in ['BUY', 'SELL']:
                    quantity = random.randint(10, 100)
                    price = random.uniform(100, 500)
                    order_id = f"ORD-{random.randint(10000, 99999)}"
                    
                    trading_logger.info(
                        f"Executing {signal} order for {quantity} shares of {symbol} "
                        f"at ${price:.2f} (ID: {order_id})"
                    )
                    
                    # Simulate execution
                    fill_price = price * (1 + random.uniform(-0.01, 0.01))
                    slippage = (fill_price - price) / price * 100
                    
                    trading_logger.info(
                        f"Order {order_id} filled at ${fill_price:.2f} "
                        f"with slippage of {slippage:.2f}%"
                    )
                    
                    # Simulate occasional warning
                    if abs(slippage) > 0.5:
                        trading_logger.warning(f"High slippage detected for order {order_id}")
                
            except Exception as e:
                # Log error with exception info
                trading_logger.error(f"Error executing {strategy} for {symbol}: {str(e)}", exc_info=True)
    
    # Log portfolio summary
    portfolio_value = random.uniform(100000, 500000)
    daily_pnl = random.uniform(-5000, 5000)
    trading_logger.info(f"Trading day completed: Portfolio value=${portfolio_value:.2f}, Daily P&L=${daily_pnl:.2f}")

def simulate_system_operations():
    """Simulate system operations with logging."""
    system_logger.info("Starting system operations")
    
    # Simulate system checks
    system_logger.debug("Performing system health checks")
    
    # Simulate database connection
    try:
        system_logger.debug("Connecting to database")
        if random.random() < 0.1:
            raise Exception("Database connection timeout")
        system_logger.info("Database connection established")
    except Exception as e:
        system_logger.error(f"Database connection error: {str(e)}", exc_info=True)
    
    # Simulate API health checks
    apis = ['market_data', 'execution', 'authentication']
    for api in apis:
        try:
            system_logger.debug(f"Checking {api} API health")
            
            # Simulate API response time
            response_time = random.uniform(50, 500)
            
            if response_time > 200:
                system_logger.warning(f"{api} API response time high: {response_time:.2f}ms")
            else:
                system_logger.info(f"{api} API health check passed: {response_time:.2f}ms")
                
        except Exception as e:
            system_logger.error(f"{api} API health check failed: {str(e)}", exc_info=True)
    
    # Simulate resource usage
    cpu_usage = random.uniform(10, 90)
    memory_usage = random.uniform(20, 80)
    disk_usage = random.uniform(30, 70)
    
    system_logger.info(f"Resource usage - CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%, Disk: {disk_usage:.1f}%")
    
    if cpu_usage > 80:
        system_logger.warning(f"High CPU usage detected: {cpu_usage:.1f}%")
    
    if memory_usage > 80:
        system_logger.warning(f"High memory usage detected: {memory_usage:.1f}%")
    
    if disk_usage > 80:
        system_logger.warning(f"High disk usage detected: {disk_usage:.1f}%")
    
    system_logger.info("System operations completed")

def simulate_error_scenarios():
    """Simulate various error scenarios for logging demonstration."""
    # Simulate data processing error
    try:
        data_logger.debug("Processing market data batch")
        if random.random() < 0.5:
            raise ValueError("Invalid data format in market data batch")
    except Exception as e:
        data_logger.error(f"Data processing error: {str(e)}", exc_info=True)
    
    # Simulate model prediction error
    try:
        model_logger.debug("Making predictions with market_regime model")
        if random.random() < 0.5:
            raise RuntimeError("Model prediction failed due to incompatible input shape")
    except Exception as e:
        model_logger.error(f"Model prediction error: {str(e)}", exc_info=True)
    
    # Simulate trading execution error
    try:
        trading_logger.debug("Executing market order")
        if random.random() < 0.5:
            raise ConnectionError("Order execution failed: connection to broker lost")
    except Exception as e:
        trading_logger.error(f"Order execution error: {str(e)}", exc_info=True)
    
    # Simulate critical system error
    try:
        system_logger.debug("Performing database backup")
        if random.random() < 0.5:
            raise IOError("Critical error: database backup failed due to disk failure")
    except Exception as e:
        system_logger.critical(f"Critical system error: {str(e)}", exc_info=True)

def main():
    """Run the logging example."""
    default_logger.info("Starting logging example")
    
    # Simulate different components
    simulate_data_collection()
    simulate_model_training()
    simulate_trading()
    simulate_system_operations()
    simulate_error_scenarios()
    
    default_logger.info("Logging example completed")
    
    # Print information about log locations
    print("\nLog files:")
    print("  - Data logs: logs/data.log")
    print("  - Model logs: logs/model.log")
    print("  - Trading logs: logs/trading.log")
    print("  - System logs: logs/system.log")
    print("  - Default logs: logs/system_trader.log")

if __name__ == "__main__":
    main()
