#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Configuration Update Script
-----------------------------------
Updates the system configuration with TimescaleDB connection settings.
"""

import os
import yaml
import argparse

def update_config(config_path, host, port, dbname, user, password):
    """
    Update the system configuration file with database settings.
    
    Args:
        config_path: Path to the system config file
        host: Database host
        port: Database port
        dbname: Database name
        user: Database user
        password: Database password
    """
    # Read existing config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Create or update database section
    if 'database' not in config:
        config['database'] = {}
    
    # Update TimescaleDB config
    if 'timescaledb' not in config['database']:
        config['database']['timescaledb'] = {}
    
    config['database']['timescaledb'].update({
        'host': host,
        'port': port,
        'dbname': dbname,
        'user': user,
        'password': password,
        'schema': 'public',
        'min_connections': 1,
        'max_connections': 10,
        'max_retries': 3,
        'retry_delay': 1,
        'max_retry_delay': 30
    })
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated database configuration in {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update database configuration')
    parser.add_argument('--config', type=str, default='config/system_config.yaml',
                        help='Path to system config file')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Database host')
    parser.add_argument('--port', type=int, default=5432,
                        help='Database port')
    parser.add_argument('--dbname', type=str, default='timescaledb_test',
                        help='Database name')
    parser.add_argument('--user', type=str, default='timescaleuser',
                        help='Database user')
    parser.add_argument('--password', type=str, default='password',
                        help='Database password')
    
    args = parser.parse_args()
    
    update_config(
        args.config,
        args.host,
        args.port,
        args.dbname,
        args.user,
        args.password
    )