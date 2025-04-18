#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Connection Test Script
------------------------------
Tests the connection to TimescaleDB using the system configuration.
"""

import sys
import yaml
from data.database.timeseries_db import TimeseriesDBClient

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None

def test_connection():
    """Test connection to TimescaleDB."""
    # Load configuration
    config = load_config('config/system_config.yaml')
    if not config:
        print("Failed to load configuration")
        return False
    
    # Extract TimescaleDB config
    if 'database' not in config or 'timescaledb' not in config['database']:
        print("TimescaleDB configuration not found in system_config.yaml")
        return False
    
    db_config = config['database']['timescaledb']
    
    # Create client
    try:
        print("Connecting to TimescaleDB...")
        db_client = TimeseriesDBClient(db_config)
        
        # Test connection with health check
        health = db_client.health_check()
        print(f"Connection status: {health['status']}")
        print(f"Message: {health['message']}")
        print("\nDetails:")
        for key, value in health['details'].items():
            print(f"  {key}: {value}")
        
        # Test query
        print("\nTesting query...")
        result = db_client.execute_query("SELECT COUNT(*) FROM market_data")
        print(f"Market data count: {result[0]['count']}")
        
        # Close connection
        db_client.close()
        print("\nConnection test completed successfully")
        return True
    
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)