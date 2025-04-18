# Logging Strategy

## Overview

This document outlines the logging strategy for the System Trader platform. The platform uses a centralized logging approach with categorized logs to ensure efficient log management, monitoring, and troubleshooting.

## Logging Architecture

The logging system is built on Python's standard `logging` module with custom enhancements to support:

1. **Categorized Logs**: Logs are organized by functional categories
2. **Centralized Configuration**: All logging is configured through a single utility
3. **Consistent Formatting**: Uniform log format across all components
4. **Log Rotation**: Automatic rotation of log files to prevent excessive disk usage
5. **Multiple Outputs**: Logs are sent to both console and files

## Log Categories

Logs are organized into the following categories:

| Category | Description | Log File |
|----------|-------------|----------|
| `data` | Data collection, processing, and storage | `logs/data.log` |
| `model` | Model training, evaluation, and inference | `logs/model.log` |
| `trading` | Trading strategy, execution, and monitoring | `logs/trading.log` |
| `system` | System-level operations and infrastructure | `logs/system.log` |
| `default` | Uncategorized logs | `logs/system_trader.log` |

## Usage

### Basic Usage

```python
from utils.logging import setup_logger

# Create a logger with a specific category
logger = setup_logger('component_name', category='data')

# Use the logger
logger.info("This is an informational message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.debug("This is a debug message")
```

### Configuration

Logging configuration is managed through the `utils/setup_logging.py` module. The configuration includes:

- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log formats
- Log file locations
- Log rotation settings

## Log Format

The standard log format includes:

- Timestamp
- Log Level
- Logger Name
- Message

Example:
```
2025-04-17 10:15:23,456 - INFO - alpaca_collector - Collecting data for AAPL from 2025-04-01 to 2025-04-17
```

## Best Practices

1. **Use Appropriate Log Levels**:
   - `DEBUG`: Detailed information for debugging
   - `INFO`: Confirmation that things are working as expected
   - `WARNING`: Indication that something unexpected happened, but the application is still working
   - `ERROR`: Due to a more serious problem, the application has not been able to perform a function
   - `CRITICAL`: A serious error indicating that the program itself may be unable to continue running

2. **Include Contextual Information**:
   - Include relevant identifiers (symbol, order ID, etc.)
   - Include quantitative information where applicable
   - For errors, include exception details

3. **Structured Logging**:
   - Use consistent terminology
   - Keep messages concise but informative
   - Include relevant metrics and measurements

4. **Performance Considerations**:
   - Avoid excessive logging in performance-critical paths
   - Use debug level for high-volume logs
   - Consider using lazy evaluation for complex log messages

## Log Management

### Log Rotation

Log files are automatically rotated:
- When they reach a certain size (10MB)
- Daily at midnight
- A maximum of 30 backup files are kept

### Log Analysis

Logs can be analyzed using:
- Standard Unix tools (grep, awk, etc.)
- Log analysis tools (ELK stack, Grafana Loki, etc.)
- Custom scripts in the `monitoring` module

## Integration with Monitoring

The logging system integrates with the monitoring system:
- Critical errors are forwarded to alerting systems
- Log metrics (error rates, etc.) are exposed to Prometheus
- Log dashboards are available in Grafana

## Example Implementation

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.logging import setup_logger

# Create a logger with a specific category
logger = setup_logger('data_processor', category='data')

def process_data(symbol, start_date, end_date):
    """Process market data for a symbol."""
    logger.info(f"Processing data for {symbol} from {start_date} to {end_date}")
    
    try:
        # Data processing logic here
        records_processed = 1000  # Example count
        logger.info(f"Successfully processed {records_processed} records for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {str(e)}", exc_info=True)
        return False
