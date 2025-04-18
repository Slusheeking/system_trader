#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TimescaleDB Client
----------------
Provides integration with TimescaleDB for high-performance time-series data storage.
Handles connection, data insertion, and querying with optimizations for time-series operations.
"""

import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
import time
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values, DictCursor
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import contextlib
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from utils.config_loader import ConfigLoader

# Setup logging
logger = setup_logger('timeseries_db', category='data')


class TimeseriesDBClient:
    """
    Client for interacting with TimescaleDB database for time-series data.
    Provides resilient connection handling, retry logic, and health checks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize TimescaleDB client.
        
        Args:
            config: Database configuration settings. If None, loads from central config.
        """
        if config is None:
            # Load from central config loader
            try:
                # Check if CONFIG_PATH environment variable is set
                config_path = os.environ.get('CONFIG_PATH', 'config/system_config.yaml')
                logger.info(f"Loading database configuration from: {config_path}")
                config_loader = ConfigLoader()
                system_config = config_loader.load_yaml(config_path)
                config = system_config.get('database', {}).get('timescaledb', {})
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                config = {}
        
        self.config = config or {}
        
        # Database connection settings
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 5432)
        self.dbname = self.config.get('dbname', 'trading')
        self.user = self.config.get('user', 'postgres')
        self.password = self.config.get('password', '')
        self.schema = self.config.get('schema', 'public')
        
        # Pool settings
        self.min_connections = self.config.get('min_connections', 1)
        self.max_connections = self.config.get('max_connections', 10)
        
        # Retry settings
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        self.max_retry_delay = self.config.get('max_retry_delay', 30)
        
        # Connection pool
        self.connection_pool = None
        
        # SQLAlchemy engine and session
        self.engine = None
        self.Session = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialization
        self._initialize_connection_pool()
        self._initialize_sqlalchemy()
        
        logger.info(f"TimeseriesDBClient initialized for {self.host}:{self.port}/{self.dbname}")
    
    @retry(
        retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=lambda retry_state: logger.warning(
            f"Connection attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep} seconds"
        )
    )
    def _initialize_connection_pool(self):
        """Initialize the connection pool for psycopg2 with retry logic."""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connection pool initialized with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            logger.error(f"Error initializing connection pool: {str(e)}")
            raise
    
    def _initialize_sqlalchemy(self):
        """Initialize SQLAlchemy engine and session factory."""
        try:
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
            self.engine = sa.create_engine(connection_string, pool_size=self.max_connections, max_overflow=10)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("SQLAlchemy engine and session factory initialized")
        except Exception as e:
            logger.error(f"Error initializing SQLAlchemy: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Connection attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep} seconds"
        )
    )
    @contextlib.contextmanager
    def get_connection(self):
        """
        Get a connection from the pool with retry logic.
        
        Yields:
            Connection object
        """
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logger.error(f"Database connection error: {str(e)}")
            if connection:
                self.connection_pool.putconn(connection, close=True)
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    @contextlib.contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Get a cursor from a pooled connection.
        
        Args:
            cursor_factory: Optional cursor factory
            
        Yields:
            Database cursor
        """
        with self.get_connection() as connection:
            cursor = connection.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()
    
    def create_hypertable(self, table_name: str, time_column: str = 'time',
                         chunk_time_interval: str = '1 day'):
        """
        Convert a regular table to a TimescaleDB hypertable.
        
        Args:
            table_name: Name of the table to convert
            time_column: Name of the time column
            chunk_time_interval: Interval for chunks
        
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Check if table already exists as a hypertable
                cursor.execute(
                    "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = %s",
                    (table_name,)
                )
                if cursor.fetchone():
                    logger.info(f"Table {table_name} is already a hypertable")
                    return True
                
                # Convert to hypertable
                cursor.execute(
                    f"SELECT create_hypertable('{table_name}', '{time_column}', "
                    f"chunk_time_interval => INTERVAL '{chunk_time_interval}')"
                )
                logger.info(f"Created hypertable {table_name} with time column {time_column}")
                return True
        except Exception as e:
            logger.error(f"Error creating hypertable {table_name}: {str(e)}")
            return False
    
    def create_market_data_table(self, table_name: str = 'market_data'):
        """
        Create a table for market data.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Create table if not exists
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(16) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    vwap DOUBLE PRECISION,
                    num_trades INTEGER,
                    source VARCHAR(32),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """)
                
                # Create indexes
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)
                """)
                
                # Convert to hypertable
                self.create_hypertable(table_name, 'time')
                
                # Add compression policy (if supported)
                try:
                    cursor.execute(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol'
                    )
                    """)
                    
                    cursor.execute(f"""
                    SELECT add_compression_policy('{table_name}', INTERVAL '7 days')
                    """)
                    logger.info(f"Added compression policy to {table_name}")
                except Exception as e:
                    logger.warning(f"Could not add compression policy (may require enterprise version): {str(e)}")
                
                logger.info(f"Created market data table {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error creating market data table {table_name}: {str(e)}")
            return False
    
    def create_trade_data_table(self, table_name: str = 'trade_data'):
        """
        Create a table for trade execution data.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Create table if not exists
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    time TIMESTAMPTZ NOT NULL,
                    order_id VARCHAR(64) NOT NULL,
                    trade_id VARCHAR(64),
                    symbol VARCHAR(16) NOT NULL,
                    side VARCHAR(8) NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    order_type VARCHAR(16),
                    execution_venue VARCHAR(32),
                    strategy_id VARCHAR(64),
                    commission DOUBLE PRECISION,
                    slippage DOUBLE PRECISION,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """)
                
                # Create indexes
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_order_id ON {table_name} (order_id)
                """)
                
                # Convert to hypertable
                self.create_hypertable(table_name, 'time')
                
                logger.info(f"Created trade data table {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error creating trade data table {table_name}: {str(e)}")
            return False
    
    def create_analytics_table(self, table_name: str = 'analytics_data'):
        """
        Create a table for analytics data.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Create table if not exists
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    time TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(64) NOT NULL,
                    metric_value DOUBLE PRECISION,
                    symbol VARCHAR(16),
                    strategy_id VARCHAR(64),
                    dimension VARCHAR(32),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """)
                
                # Create indexes
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_metric_name ON {table_name} (metric_name)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_dimension ON {table_name} (dimension)
                """)
                
                # Convert to hypertable
                self.create_hypertable(table_name, 'time')
                
                logger.info(f"Created analytics data table {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error creating analytics data table {table_name}: {str(e)}")
            return False
    
    def insert_market_data(self, data: List[Dict[str, Any]], table_name: str = 'market_data'):
        """
        Insert market data into database.
        
        Args:
            data: List of market data dictionaries
            table_name: Target table name
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        try:
            with self.get_cursor() as cursor:
                # Prepare column names and values
                columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                         'vwap', 'num_trades', 'source', 'metadata']
                
                values = []
                for item in data:
                    # Ensure time is in the correct format
                    if isinstance(item.get('time'), str):
                        time_val = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    elif isinstance(item.get('time'), (int, float)):
                        time_val = datetime.fromtimestamp(item['time'] / 1000 if item['time'] > 1e10 else item['time'])
                    else:
                        time_val = item.get('time', datetime.now())
                    
                    # Convert metadata to JSON if needed
                    metadata = item.get('metadata', {})
                    if not isinstance(metadata, str):
                        try:
                            metadata = json.dumps(metadata)
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error converting metadata to JSON: {str(e)}")
                            metadata = "{}"
                    else:
                        # Ensure the string is valid JSON
                        try:
                            # Try to parse it as JSON to validate
                            json.loads(metadata)
                        except (json.JSONDecodeError, ValueError):
                            # If it's not valid JSON, try to fix common issues
                            try:
                                # Replace single quotes with double quotes for JSON compatibility
                                metadata = metadata.replace("'", '"')
                                # Handle None values which aren't valid JSON
                                metadata = metadata.replace("None", "null")
                                # Validate the fixed JSON
                                json.loads(metadata)
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"Could not convert metadata to valid JSON: {str(e)}")
                                metadata = "{}"
                    
                    row = (
                        time_val,
                        item.get('symbol'),
                        item.get('open'),
                        item.get('high'),
                        item.get('low'),
                        item.get('close'),
                        item.get('volume'),
                        item.get('vwap'),
                        item.get('num_trades'),
                        item.get('source'),
                        metadata
                    )
                    values.append(row)
                
                # Insert data
                query = f"""
                INSERT INTO {table_name} (
                    time, symbol, open, high, low, close, volume, 
                    vwap, num_trades, source, metadata
                ) VALUES %s
                """
                
                execute_values(cursor, query, values)
                
                count = len(values)
                logger.debug(f"Inserted {count} rows into {table_name}")
                return count
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return 0
    
    def insert_trade_data(self, data: List[Dict[str, Any]], table_name: str = 'trade_data'):
        """
        Insert trade execution data into database.
        
        Args:
            data: List of trade data dictionaries
            table_name: Target table name
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        try:
            with self.get_cursor() as cursor:
                # Prepare column names and values
                columns = ['time', 'order_id', 'trade_id', 'symbol', 'side', 'quantity', 
                         'price', 'order_type', 'execution_venue', 'strategy_id', 
                         'commission', 'slippage', 'metadata']
                
                values = []
                for item in data:
                    # Ensure time is in the correct format
                    if isinstance(item.get('time'), str):
                        time_val = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    elif isinstance(item.get('time'), (int, float)):
                        time_val = datetime.fromtimestamp(item['time'] / 1000 if item['time'] > 1e10 else item['time'])
                    else:
                        time_val = item.get('time', datetime.now())
                    
                    # Convert metadata to JSON if needed
                    metadata = item.get('metadata', {})
                    if not isinstance(metadata, str):
                        try:
                            metadata = json.dumps(metadata)
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error converting metadata to JSON: {str(e)}")
                            metadata = "{}"
                    else:
                        # Ensure the string is valid JSON
                        try:
                            # Try to parse it as JSON to validate
                            json.loads(metadata)
                        except (json.JSONDecodeError, ValueError):
                            # If it's not valid JSON, try to fix common issues
                            try:
                                # Replace single quotes with double quotes for JSON compatibility
                                metadata = metadata.replace("'", '"')
                                # Handle None values which aren't valid JSON
                                metadata = metadata.replace("None", "null")
                                # Validate the fixed JSON
                                json.loads(metadata)
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"Could not convert metadata to valid JSON: {str(e)}")
                                metadata = "{}"
                    
                    row = (
                        time_val,
                        item.get('order_id'),
                        item.get('trade_id'),
                        item.get('symbol'),
                        item.get('side'),
                        item.get('quantity'),
                        item.get('price'),
                        item.get('order_type'),
                        item.get('execution_venue'),
                        item.get('strategy_id'),
                        item.get('commission'),
                        item.get('slippage'),
                        metadata
                    )
                    values.append(row)
                
                # Insert data
                query = f"""
                INSERT INTO {table_name} (
                    time, order_id, trade_id, symbol, side, quantity, price, order_type, 
                    execution_venue, strategy_id, commission, slippage, metadata
                ) VALUES %s
                """
                
                execute_values(cursor, query, values)
                
                count = len(values)
                logger.debug(f"Inserted {count} rows into {table_name}")
                return count
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return 0
    
    def insert_analytics_data(self, data: List[Dict[str, Any]], table_name: str = 'analytics_data'):
        """
        Insert analytics data into database.
        
        Args:
            data: List of analytics data dictionaries
            table_name: Target table name
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        try:
            with self.get_cursor() as cursor:
                # Prepare column names and values
                columns = ['time', 'metric_name', 'metric_value', 'symbol', 
                         'strategy_id', 'dimension', 'metadata']
                
                values = []
                for item in data:
                    # Ensure time is in the correct format
                    if isinstance(item.get('time'), str):
                        time_val = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    elif isinstance(item.get('time'), (int, float)):
                        time_val = datetime.fromtimestamp(item['time'] / 1000 if item['time'] > 1e10 else item['time'])
                    else:
                        time_val = item.get('time', datetime.now())
                    
                    # Convert metadata to JSON if needed
                    metadata = item.get('metadata', {})
                    if not isinstance(metadata, str):
                        try:
                            metadata = json.dumps(metadata)
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error converting metadata to JSON: {str(e)}")
                            metadata = "{}"
                    else:
                        # Ensure the string is valid JSON
                        try:
                            # Try to parse it as JSON to validate
                            json.loads(metadata)
                        except (json.JSONDecodeError, ValueError):
                            # If it's not valid JSON, try to fix common issues
                            try:
                                # Replace single quotes with double quotes for JSON compatibility
                                metadata = metadata.replace("'", '"')
                                # Handle None values which aren't valid JSON
                                metadata = metadata.replace("None", "null")
                                # Validate the fixed JSON
                                json.loads(metadata)
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"Could not convert metadata to valid JSON: {str(e)}")
                                metadata = "{}"
                    
                    row = (
                        time_val,
                        item.get('metric_name'),
                        item.get('metric_value'),
                        item.get('symbol'),
                        item.get('strategy_id'),
                        item.get('dimension'),
                        metadata
                    )
                    values.append(row)
                
                # Insert data
                query = f"""
                INSERT INTO {table_name} (
                    time, metric_name, metric_value, symbol, strategy_id, dimension, metadata
                ) VALUES %s
                """
                
                execute_values(cursor, query, values)
                
                count = len(values)
                logger.debug(f"Inserted {count} rows into {table_name}")
                return count
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return 0
    
    def query_market_data(self, symbol: str, start_time: datetime, end_time: datetime,
                        interval: str = None, table_name: str = 'market_data') -> pd.DataFrame:
        """
        Query market data within a time range.
        
        Args:
            symbol: Ticker symbol
            start_time: Start time
            end_time: End time
            interval: Optional time bucket interval (e.g., '1 minute', '5 minutes', '1 hour')
            table_name: Source table name
            
        Returns:
            DataFrame with market data
        """
        try:
            if interval:
                # Time-bucketed query
                query = f"""
                SELECT 
                    time_bucket('{interval}', time) AS bucket,
                    symbol,
                    FIRST(open, time) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, time) AS close,
                    SUM(volume) AS volume,
                    AVG(vwap) AS vwap,
                    SUM(num_trades) AS num_trades
                FROM {table_name}
                WHERE symbol = %s AND time >= %s AND time <= %s
                GROUP BY bucket, symbol
                ORDER BY bucket
                """
                params = (symbol, start_time, end_time)
            else:
                # Raw data query
                query = f"""
                SELECT 
                    time, symbol, open, high, low, close, volume, vwap, num_trades, source
                FROM {table_name}
                WHERE symbol = %s AND time >= %s AND time <= %s
                ORDER BY time
                """
                params = (symbol, start_time, end_time)
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} rows for {symbol} from {start_time} to {end_time}")
                    return df
                else:
                    logger.warning(f"No data found for {symbol} from {start_time} to {end_time}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying market data: {str(e)}")
            return pd.DataFrame()
    
    def query_trade_data(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None,
                        start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                        table_name: str = 'trade_data') -> pd.DataFrame:
        """
        Query trade execution data with filters.
        
        Args:
            symbol: Optional ticker symbol filter
            strategy_id: Optional strategy ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            table_name: Source table name
            
        Returns:
            DataFrame with trade data
        """
        try:
            # Build query based on filters
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if strategy_id:
                query += " AND strategy_id = %s"
                params.append(strategy_id)
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            query += " ORDER BY time"
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} trade records")
                    return df
                else:
                    logger.warning("No trade data found matching criteria")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying trade data: {str(e)}")
            return pd.DataFrame()
    
    def query_analytics_data(self, metric_name: Optional[str] = None, symbol: Optional[str] = None,
                           start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                           dimension: Optional[str] = None, table_name: str = 'analytics_data') -> pd.DataFrame:
        """
        Query analytics data with filters.
        
        Args:
            metric_name: Optional metric name filter
            symbol: Optional ticker symbol filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            dimension: Optional dimension filter
            table_name: Source table name
            
        Returns:
            DataFrame with analytics data
        """
        try:
            # Build query based on filters
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            
            if metric_name:
                query += " AND metric_name = %s"
                params.append(metric_name)
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            if dimension:
                query += " AND dimension = %s"
                params.append(dimension)
            
            query += " ORDER BY time"
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} analytics records")
                    return df
                else:
                    logger.warning("No analytics data found matching criteria")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying analytics data: {str(e)}")
            return pd.DataFrame()
    
    def get_continuous_aggregate(self, symbol: str, start_time: datetime, end_time: datetime,
                               interval: str = '1 hour', view_name: str = 'market_data_hourly'):
        """
        Get data from a continuous aggregate view.
        
        Args:
            symbol: Ticker symbol
            start_time: Start time
            end_time: End time
            interval: Time interval
            view_name: Aggregate view name
            
        Returns:
            DataFrame with aggregated data
        """
        try:
            # Check if view exists, if not create it
            self.create_continuous_aggregate(interval, view_name)
            
            # Query the view
            query = f"""
            SELECT * FROM {view_name}
            WHERE symbol = %s AND bucket >= %s AND bucket <= %s
            ORDER BY bucket
            """
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, (symbol, start_time, end_time))
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} rows from continuous aggregate {view_name}")
                    return df
                else:
                    logger.warning(f"No data found in {view_name} for {symbol}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying continuous aggregate: {str(e)}")
            return pd.DataFrame()
    
    def create_continuous_aggregate(self, interval: str = '1 hour', view_name: str = 'market_data_hourly',
                                  source_table: str = 'market_data'):
        """
        Create a continuous aggregate view for faster queries.
        
        Args:
            interval: Time bucket interval 
            view_name: Name for the view
            source_table: Source table name
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Check if view already exists
                cursor.execute("SELECT to_regclass(%s)", (view_name,))
                if cursor.fetchone()[0] is not None:
                    logger.debug(f"Continuous aggregate {view_name} already exists")
                    return True
                
                # Create continuous aggregate view
                create_view_query = f"""
                CREATE MATERIALIZED VIEW {view_name}
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('{interval}', time) AS bucket,
                    symbol,
                    FIRST(open, time) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, time) AS close,
                    SUM(volume) AS volume,
                    AVG(vwap) AS vwap,
                    SUM(num_trades) AS num_trades
                FROM {source_table}
                GROUP BY bucket, symbol;
                """
                cursor.execute(create_view_query)
                
                # Set refresh policy (every hour)
                policy_query = f"""
                SELECT add_continuous_aggregate_policy('{view_name}',
                    start_offset => INTERVAL '3 days',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour');
                """
                cursor.execute(policy_query)
                
                logger.info(f"Created continuous aggregate {view_name} with {interval} intervals")
                return True
        except Exception as e:
            logger.error(f"Error creating continuous aggregate: {str(e)}")
            return False
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params or ())
                results = cursor.fetchall()
                
                if results:
                    # Convert to list of dicts
                    result_list = [dict(row) for row in results]
                    logger.debug(f"Executed custom query, returned {len(result_list)} rows")
                    return result_list
                else:
                    logger.debug("Custom query returned no results")
                    return []
        except Exception as e:
            logger.error(f"Error executing custom query: {str(e)}")
            return []
    
    def get_latest_price(self, symbol: str, table_name: str = 'market_data') -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Ticker symbol
            table_name: Source table name
            
        Returns:
            Latest price or None if not found
        """
        try:
            query = f"""
            SELECT close 
            FROM {table_name} 
            WHERE symbol = %s 
            ORDER BY time DESC 
            LIMIT 1
            """
            
            with self.get_cursor() as cursor:
                cursor.execute(query, (symbol,))
                result = cursor.fetchone()
                
                if result:
                    return result[0]
                else:
                    logger.warning(f"No price data found for {symbol}")
                    return None
        except Exception as e:
            logger.error(f"Error getting latest price: {str(e)}")
            return None
    
    def get_ohlcv_summary(self, symbols: List[str], timeframe: str = '1 day',
                        lookback: int = 30, table_name: str = 'market_data') -> pd.DataFrame:
        """
        Get OHLCV summary for multiple symbols over a lookback period.
        
        Args:
            symbols: List of ticker symbols
            timeframe: Time bucket interval
            lookback: Number of periods to look back
            table_name: Source table name
            
        Returns:
            DataFrame with OHLCV summary
        """
        try:
            end_time = datetime.now()
            
            # Calculate start time based on lookback and timeframe
            if timeframe == '1 day':
                start_time = end_time - timedelta(days=lookback)
            elif timeframe == '1 hour':
                start_time = end_time - timedelta(hours=lookback)
            elif timeframe == '1 minute':
                start_time = end_time - timedelta(minutes=lookback)
            else:
                # Default to 30 days
                start_time = end_time - timedelta(days=30)
            
            query = f"""
            SELECT 
                time_bucket('{timeframe}', time) AS bucket,
                symbol,
                FIRST(open, time) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close, time) AS close,
                SUM(volume) AS volume
            FROM {table_name}
            WHERE symbol = ANY(%s) AND time >= %s AND time <= %s
            GROUP BY bucket, symbol
            ORDER BY symbol, bucket
            """
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, (symbols, start_time, end_time))
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} rows for OHLCV summary")
                    return df
                else:
                    logger.warning(f"No OHLCV data found for {symbols}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting OHLCV summary: {str(e)}")
            return pd.DataFrame()
    
    def get_db_size(self) -> Dict[str, Any]:
        """
        Get database size information.
        
        Returns:
            Dictionary with size information
        """
        try:
            with self.get_cursor() as cursor:
                # Get total database size
                cursor.execute("SELECT pg_size_pretty(pg_database_size(%s))", (self.dbname,))
                total_size = cursor.fetchone()[0]
                
                # Get table sizes
                cursor.execute("""
                SELECT table_name, pg_size_pretty(total_bytes) as total,
                       pg_size_pretty(index_bytes) as index,
                       pg_size_pretty(toast_bytes) as toast,
                       pg_size_pretty(table_bytes) as table
                FROM (
                    SELECT *, total_bytes-index_bytes-COALESCE(toast_bytes,0) AS table_bytes FROM (
                        SELECT c.oid,nspname AS table_schema, relname AS table_name,
                               c.reltuples AS row_estimate,
                               pg_total_relation_size(c.oid) AS total_bytes,
                               pg_indexes_size(c.oid) AS index_bytes,
                               pg_total_relation_size(reltoastrelid) AS toast_bytes
                        FROM pg_class c
                        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE relkind = 'r' AND nspname = 'public'
                    ) a
                ) b
                ORDER BY total_bytes DESC
                LIMIT 10
                """)
                table_sizes = [dict(zip(["table_name", "total", "index", "toast", "table"], row)) for row in cursor.fetchall()]
                
                # Get chunk information
                cursor.execute("""
                SELECT chunk_schema || '.' || chunk_name as chunk, 
                       pg_size_pretty(pg_total_relation_size(chunk_schema || '.' || chunk_name::text)) as size
                FROM timescaledb_information.chunks
                ORDER BY pg_total_relation_size(chunk_schema || '.' || chunk_name::text) DESC
                LIMIT 10
                """)
                chunks = [dict(zip(["chunk", "size"], row)) for row in cursor.fetchall()]
                
                return {
                    "database_name": self.dbname,
                    "total_size": total_size,
                    "table_sizes": table_sizes,
                    "chunks": chunks
                }
        except Exception as e:
            logger.error(f"Error getting database size information: {str(e)}")
            return {"error": str(e)}
    
    def create_news_data_table(self, table_name: str = 'news_data'):
        """
        Create a table for news data with sentiment analysis.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Create table if not exists
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    news_id TEXT,
                    title TEXT,
                    author TEXT,
                    source TEXT,
                    url TEXT,
                    sentiment_score FLOAT,
                    sentiment_magnitude FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """)
                
                # Create indexes
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_time_symbol ON {table_name} (time, symbol)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_sentiment ON {table_name} (sentiment_score)
                """)
                
                # Convert to hypertable
                self.create_hypertable(table_name, 'time')
                
                logger.info(f"Created news data table {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error creating news data table {table_name}: {str(e)}")
            return False
    
    def create_social_data_table(self, table_name: str = 'social_data'):
        """
        Create a table for social media data with sentiment analysis.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.get_cursor() as cursor:
                # Create table if not exists
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    subreddit TEXT,
                    post_id TEXT,
                    parent_id TEXT,
                    author TEXT,
                    content_type TEXT,
                    sentiment_score FLOAT,
                    sentiment_magnitude FLOAT,
                    score INT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """)
                
                # Create indexes
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_time_symbol ON {table_name} (time, symbol)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_platform ON {table_name} (platform)
                """)
                cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_sentiment ON {table_name} (sentiment_score)
                """)
                
                # Convert to hypertable
                self.create_hypertable(table_name, 'time')
                
                logger.info(f"Created social data table {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error creating social data table {table_name}: {str(e)}")
            return False
    
    def insert_news_data(self, data: List[Dict[str, Any]], table_name: str = 'news_data') -> int:
        """
        Insert news data into database.
        
        Args:
            data: List of news data dictionaries
            table_name: Target table name
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        try:
            with self.get_cursor() as cursor:
                # Prepare column names and values
                columns = ['time', 'symbol', 'news_id', 'title', 'author', 'source', 
                         'url', 'sentiment_score', 'sentiment_magnitude', 'metadata']
                
                values = []
                for item in data:
                    # Ensure time is in the correct format
                    if isinstance(item.get('time'), str):
                        time_val = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    elif isinstance(item.get('time'), (int, float)):
                        time_val = datetime.fromtimestamp(item['time'] / 1000 if item['time'] > 1e10 else item['time'])
                    else:
                        time_val = item.get('time', datetime.now())
                    
                    # Convert metadata to JSON if needed
                    metadata = item.get('metadata', {})
                    if not isinstance(metadata, str):
                        try:
                            metadata = json.dumps(metadata)
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error converting metadata to JSON: {str(e)}")
                            metadata = "{}"
                    
                    row = (
                        time_val,
                        item.get('symbol'),
                        item.get('news_id'),
                        item.get('title'),
                        item.get('author'),
                        item.get('source'),
                        item.get('url'),
                        item.get('sentiment_score'),
                        item.get('sentiment_magnitude'),
                        metadata
                    )
                    values.append(row)
                
                # Insert data
                query = f"""
                INSERT INTO {table_name} (
                    time, symbol, news_id, title, author, source, url, 
                    sentiment_score, sentiment_magnitude, metadata
                ) VALUES %s
                RETURNING id
                """
                
                result = execute_values(cursor, query, values, fetch=True)
                
                count = len(result)
                logger.debug(f"Inserted {count} rows into {table_name}")
                return count
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return 0
    
    def insert_social_data(self, data: List[Dict[str, Any]], table_name: str = 'social_data') -> int:
        """
        Insert social media data into database.
        
        Args:
            data: List of social data dictionaries
            table_name: Target table name
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        try:
            with self.get_cursor() as cursor:
                # Prepare column names and values
                columns = ['time', 'symbol', 'source', 'platform', 'subreddit', 'post_id', 
                         'parent_id', 'author', 'content_type', 'sentiment_score', 
                         'sentiment_magnitude', 'score', 'metadata']
                
                values = []
                for item in data:
                    # Ensure time is in the correct format
                    if isinstance(item.get('time'), str):
                        time_val = datetime.fromisoformat(item['time'].replace('Z', '+00:00'))
                    elif isinstance(item.get('time'), (int, float)):
                        time_val = datetime.fromtimestamp(item['time'] / 1000 if item['time'] > 1e10 else item['time'])
                    else:
                        time_val = item.get('time', datetime.now())
                    
                    # Convert metadata to JSON if needed
                    metadata = item.get('metadata', {})
                    if not isinstance(metadata, str):
                        try:
                            metadata = json.dumps(metadata)
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error converting metadata to JSON: {str(e)}")
                            metadata = "{}"
                    
                    row = (
                        time_val,
                        item.get('symbol'),
                        item.get('source'),
                        item.get('platform'),
                        item.get('subreddit'),
                        item.get('post_id'),
                        item.get('parent_id'),
                        item.get('author'),
                        item.get('content_type'),
                        item.get('sentiment_score'),
                        item.get('sentiment_magnitude'),
                        item.get('score'),
                        metadata
                    )
                    values.append(row)
                
                # Insert data
                query = f"""
                INSERT INTO {table_name} (
                    time, symbol, source, platform, subreddit, post_id, parent_id,
                    author, content_type, sentiment_score, sentiment_magnitude, score, metadata
                ) VALUES %s
                RETURNING id
                """
                
                result = execute_values(cursor, query, values, fetch=True)
                
                count = len(result)
                logger.debug(f"Inserted {count} rows into {table_name}")
                return count
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return 0
    
    def query_news_data(self, symbol: Optional[str] = None, start_time: Optional[datetime] = None, 
                      end_time: Optional[datetime] = None, limit: int = 1000, 
                      min_sentiment: Optional[float] = None, table_name: str = 'news_data') -> pd.DataFrame:
        """
        Query news data with filters.
        
        Args:
            symbol: Optional ticker symbol filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            min_sentiment: Optional minimum sentiment score filter
            table_name: Source table name
            
        Returns:
            DataFrame with news data
        """
        try:
            # Build query based on filters
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            if min_sentiment is not None:
                query += " AND sentiment_score >= %s"
                params.append(min_sentiment)
            
            query += " ORDER BY time DESC LIMIT %s"
            params.append(limit)
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} news records")
                    return df
                else:
                    logger.warning("No news data found matching criteria")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying news data: {str(e)}")
            return pd.DataFrame()
    
    def query_social_data(self, symbol: Optional[str] = None, platform: Optional[str] = None,
                        start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                        limit: int = 1000, min_sentiment: Optional[float] = None, 
                        table_name: str = 'social_data') -> pd.DataFrame:
        """
        Query social media data with filters.
        
        Args:
            symbol: Optional ticker symbol filter
            platform: Optional platform filter (e.g., 'reddit')
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            min_sentiment: Optional minimum sentiment score filter
            table_name: Source table name
            
        Returns:
            DataFrame with social data
        """
        try:
            # Build query based on filters
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if platform:
                query += " AND platform = %s"
                params.append(platform)
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            if min_sentiment is not None:
                query += " AND sentiment_score >= %s"
                params.append(min_sentiment)
            
            query += " ORDER BY time DESC LIMIT %s"
            params.append(limit)
            
            with self.get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    logger.debug(f"Fetched {len(df)} social records")
                    return df
                else:
                    logger.warning("No social data found matching criteria")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying social data: {str(e)}")
            return pd.DataFrame()
    
    def get_sentiment_summary(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment summary for a symbol across news and social sources.
        
        Args:
            symbol: Ticker symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment summary
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Query news sentiment
            news_query = """
            SELECT 
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as mention_count,
                SUM(CASE WHEN sentiment_score > 0.05 THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN sentiment_score < -0.05 THEN 1 ELSE 0 END) as bearish_count
            FROM news_data
            WHERE symbol = %s AND time >= %s AND time <= %s
            """
            
            # Query social sentiment
            social_query = """
            SELECT 
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as mention_count,
                SUM(CASE WHEN sentiment_score > 0.05 THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN sentiment_score < -0.05 THEN 1 ELSE 0 END) as bearish_count
            FROM social_data
            WHERE symbol = %s AND time >= %s AND time <= %s
            """
            
            with self.get_cursor() as cursor:
                # Get news sentiment
                cursor.execute(news_query, (symbol, start_time, end_time))
                news_result = cursor.fetchone()
                
                # Get social sentiment
                cursor.execute(social_query, (symbol, start_time, end_time))
                social_result = cursor.fetchone()
                
                # Calculate combined sentiment
                news_avg = news_result[0] if news_result and news_result[0] is not None else 0
                news_count = news_result[1] if news_result and news_result[1] is not None else 0
                news_bullish = news_result[2] if news_result and news_result[2] is not None else 0
                news_bearish = news_result[3] if news_result and news_result[3] is not None else 0
                
                social_avg = social_result[0] if social_result and social_result[0] is not None else 0
                social_count = social_result[1] if social_result and social_result[1] is not None else 0
                social_bullish = social_result[2] if social_result and social_result[2] is not None else 0
                social_bearish = social_result[3] if social_result and social_result[3] is not None else 0
                
                total_count = news_count + social_count
                
                if total_count > 0:
                    combined_avg = (news_avg * news_count + social_avg * social_count) / total_count
                else:
                    combined_avg = 0
                
                return {
                    'symbol': symbol,
                    'period_days': days,
                    'combined': {
                        'avg_sentiment': combined_avg,
                        'mention_count': total_count,
                        'bullish_count': news_bullish + social_bullish,
                        'bearish_count': news_bearish + social_bearish
                    },
                    'news': {
                        'avg_sentiment': news_avg,
                        'mention_count': news_count,
                        'bullish_count': news_bullish,
                        'bearish_count': news_bearish
                    },
                    'social': {
                        'avg_sentiment': social_avg,
                        'mention_count': social_count,
                        'bullish_count': social_bullish,
                        'bearish_count': social_bearish
                    }
                }
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Dictionary with health status information
        """
        try:
            start_time = time.time()
            
            # Check connection
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if we can execute a simple query
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    
                    if result and result[0] == 1:
                        connection_status = "OK"
                    else:
                        connection_status = "ERROR"
                    
                    # Check TimescaleDB extension
                    cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb'")
                    ts_result = cursor.fetchone()
                    
                    if ts_result:
                        timescale_status = "OK"
                        timescale_version = ts_result[1]
                    else:
                        timescale_status = "ERROR"
                        timescale_version = "Not installed"
                    
                    # Check PostgreSQL version
                    cursor.execute("SHOW server_version")
                    pg_version = cursor.fetchone()[0]
                    
                    # Get connection count
                    cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = %s", (self.dbname,))
                    connection_count = cursor.fetchone()[0]
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if connection_status == "OK" and timescale_status == "OK" else "unhealthy",
                "message": "Database connection successful" if connection_status == "OK" else "Database connection failed",
                "details": {
                    "connection_status": connection_status,
                    "imtimescale_status": timescale_status,
                    "timescale_version": timescale_version,
                    "postgresql_version": pg_version,
                    "active_connections": connection_count,
                    "response_time_ms": round(response_time * 1000, 2),
                    "database": self.dbname,
                    "host": self.host,
                    "port": self.port
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "details": {
                    "error": str(e),
                    "database": self.dbname,
                    "host": self.host,
                    "port": self.port
                }
            }
    
    def close(self):
        """Close all database connections."""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
                logger.info("Closed all database connections")
            
            if self.engine:
                self.engine.dispose()
                logger.info("Disposed SQLAlchemy engine")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")




# Create a singleton instance
_db_client = None


def get_timescale_client(config: Dict[str, Any] = None) -> TimeseriesDBClient:
    """
    Get or create the TimeseriesDBClient instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TimeseriesDBClient instance
    """
    global _db_client
    
    if _db_client is None:
        _db_client = TimeseriesDBClient(config)
    
    return _db_client


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TimescaleDB Client')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--dbname', type=str, default='trading', help='Database name')
    parser.add_argument('--user', type=str, default='postgres', help='Database user')
    parser.add_argument('--password', type=str, default='', help='Database password')
    parser.add_argument('--setup', action='store_true', help='Setup tables')
    parser.add_argument('--info', action='store_true', help='Show database info')
    parser.add_argument('--health', action='store_true', help='Run health check')
    
    args = parser.parse_args()
    
    # Create config from args
    config = {
        'host': args.host,
        'port': args.port,
        'dbname': args.dbname,
        'user': args.user,
        'password': args.password
    }
    
    # Create client
    db_client = TimeseriesDBClient(config)
    
    if args.setup:
        # Create tables
        db_client.create_market_data_table()
        db_client.create_trade_data_table()
        db_client.create_analytics_table()
        db_client.create_news_data_table()
        db_client.create_social_data_table()
        
        # Create continuous aggregates
        db_client.create_continuous_aggregate('1 hour', 'market_data_hourly')
        db_client.create_continuous_aggregate('1 day', 'market_data_daily')
        
        print("Database setup complete")
    
    if args.info:
        # Show database info
        db_info = db_client.get_db_size()
        
        print(f"Database: {db_info['database_name']}")
        print(f"Total Size: {db_info['total_size']}")
        
        print("\nLargest Tables:")
        for table in db_info['table_sizes']:
            print(f"  {table['table_name']}: {table['total']}")
        
        print("\nLargest Chunks:")
        for chunk in db_info['chunks']:
            print(f"  {chunk['chunk']}: {chunk['size']}")
    
    if args.health:
        # Run health check
        health = db_client.health_check()
        print(f"Health Status: {health['status']}")
        print(f"Message: {health['message']}")
        print("\nDetails:")
        for key, value in health['details'].items():
            print(f"  {key}: {value}")
    
    # Close connections
    db_client.close()

# Alias for backward compatibility
TimeSeriesDB = TimeseriesDBClient
