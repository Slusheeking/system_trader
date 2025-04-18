#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Database Connection Test Script
-------------------------------------
Tests the connection to TimescaleDB using direct psycopg2 connection.
"""

import sys
import yaml
import psycopg2
from psycopg2.extras import DictCursor

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
    
    # Create connection string
    conn_string = f"host={db_config['host']} port={db_config['port']} dbname={db_config['dbname']} user={db_config['user']} password={db_config['password']}"
    
    # Connect to database
    try:
        print("Connecting to TimescaleDB...")
        conn = psycopg2.connect(conn_string)
        
        # Test connection
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            # Check TimescaleDB extension
            cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb'")
            ts_result = cursor.fetchone()
            
            if ts_result:
                print(f"TimescaleDB extension: {ts_result['extname']} version {ts_result['extversion']}")
            else:
                print("TimescaleDB extension not found")
            
            # Check tables
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row['table_name'] for row in cursor.fetchall()]
            print(f"Tables in database: {', '.join(tables)}")
            
            # Check hypertables
            cursor.execute("SELECT hypertable_schema, hypertable_name FROM timescaledb_information.hypertables")
            hypertables = [f"{row['hypertable_schema']}.{row['hypertable_name']}" for row in cursor.fetchall()]
            print(f"Hypertables: {', '.join(hypertables) if hypertables else 'None'}")
            
            # Check continuous aggregates
            cursor.execute("SELECT view_name FROM timescaledb_information.continuous_aggregates")
            caggs = [row['view_name'] for row in cursor.fetchall()]
            print(f"Continuous aggregates: {', '.join(caggs) if caggs else 'None'}")
            
            # Check market data
            cursor.execute("SELECT COUNT(*) FROM market_data")
            count = cursor.fetchone()['count']
            print(f"Market data count: {count}")
            
            if count > 0:
                cursor.execute("SELECT * FROM market_data LIMIT 1")
                sample = cursor.fetchone()
                print("\nSample market data record:")
                for key, value in sample.items():
                    print(f"  {key}: {value}")
        
        # Close connection
        conn.close()
        print("\nConnection test completed successfully")
        return True
    
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)