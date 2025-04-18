#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Population Script
-------------------------
This script populates the TimescaleDB database with synthetic market data
and features required for model training.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_populator')

# Database connection parameters
DB_PARAMS = {
    'dbname': 'timescaledb_test',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        logger.info("Connected to the database")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {str(e)}")
        sys.exit(1)

def create_synthetic_price_data(symbol, start_date, end_date, initial_price=100.0, volatility=0.02):
    """
    Create synthetic price data for a symbol.
    
    Args:
        symbol: Stock symbol
        start_date: Start date for data generation
        end_date: End date for data generation
        initial_price: Starting price
        volatility: Daily volatility
        
    Returns:
        DataFrame with synthetic price data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random returns
    np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
    returns = np.random.normal(0.0005, volatility, len(date_range))
    
    # Calculate price series
    prices = initial_price * (1 + returns).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': date_range,
        'symbol': symbol,
        'open': prices * (1 + np.random.normal(0, 0.005, len(date_range))),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, len(date_range))),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, len(date_range))),
        'close': prices,
        'volume': np.random.randint(100000, 10000000, len(date_range)),
        'vwap': prices * (1 + np.random.normal(0, 0.002, len(date_range))),
        'num_trades': np.random.randint(1000, 50000, len(date_range)),
        'source': 'synthetic',
        'data_category': 'OHLCV',  # Add data_category for ml_training_engine.py
        'interval': '1d'  # Add interval for ml_training_engine.py
    })
    
    # Ensure high > open, close, low and low < open, close, high
    df['high'] = df[['high', 'open', 'close']].max(axis=1) * (1 + np.random.uniform(0.001, 0.005, len(df)))
    df['low'] = df[['low', 'open', 'close']].min(axis=1) * (1 - np.random.uniform(0.001, 0.005, len(df)))
    
    # Add timestamp column (alias for time) for model compatibility
    df['timestamp'] = df['time']
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the price data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
    """
    # Price momentum at different timeframes
    for window in [5, 10, 20]:
        df[f'return_{window}d'] = df['close'].pct_change(window)
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle_20'] = df['close'].rolling(20).mean()
    df['bb_std_20'] = df['close'].rolling(20).std()
    df['bb_upper_20'] = df['bb_middle_20'] + 2 * df['bb_std_20']
    df['bb_lower_20'] = df['bb_middle_20'] - 2 * df['bb_std_20']
    df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_middle_20']
    
    # Average True Range (ATR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    # Volume indicators
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Add target columns for different models
    
    # Stock selection model target (profitable trade in next 60 minutes/1 day)
    # For daily data, we'll use next day's return
    df['forward_return_1d'] = df['close'].shift(-1) / df['close'] - 1
    df['target'] = (df['forward_return_1d'] > 0.005).astype(int)  # 0.5% threshold
    
    # Market regime model target
    df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    df['regime'] = np.where(df['volatility_20'] > df['volatility_20'].rolling(60).mean(), 
                           'high_volatility', 'normal')
    
    # Entry timing model target
    df['entry_signal'] = ((df['close'] > df['ma_20']) & 
                         (df['rsi_14'] > 30) & (df['rsi_14'] < 70)).astype(int)
    
    # Peak detection model target
    df['local_peak'] = ((df['close'] > df['close'].shift(1)) & 
                       (df['close'] > df['close'].shift(-1))).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def create_features_table_if_needed(conn):
    """
    Create the features_data table if it doesn't exist.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    try:
        # Check if features_data table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'features_data'
            )
        """)
        
        if not cursor.fetchone()[0]:
            # Create features_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features_data (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value FLOAT,
                    feature_group TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create hypertable
            cursor.execute("""
                SELECT create_hypertable('features_data', 'time', if_not_exists => TRUE)
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_data_symbol ON features_data (symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_data_name ON features_data (feature_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_data_group ON features_data (feature_group)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_data_time_symbol ON features_data (time, symbol)
            """)
            
            logger.info("Created features_data table and indexes")
    except Exception as e:
        logger.error(f"Error creating features_data table: {str(e)}")
    finally:
        cursor.close()

def insert_market_data(conn, df):
    """
    Insert market data into the database.
    
    Args:
        conn: Database connection
        df: DataFrame with market data
    """
    cursor = conn.cursor()
    
    # Prepare data for insertion
    data = [
        (
            row.time, row.symbol, row.open, row.high, row.low, row.close, 
            row.volume, row.vwap, row.num_trades, row.source, 
            json.dumps({
                'technical_indicators': {
                    'ma_5': float(row.ma_5) if not pd.isna(row.ma_5) else None,
                    'ma_10': float(row.ma_10) if not pd.isna(row.ma_10) else None,
                    'ma_20': float(row.ma_20) if not pd.isna(row.ma_20) else None,
                    'rsi_14': float(row.rsi_14) if not pd.isna(row.rsi_14) else None,
                    'macd': float(row.macd) if not pd.isna(row.macd) else None,
                    'atr_14': float(row.atr_14) if not pd.isna(row.atr_14) else None
                },
                'targets': {
                    'target': int(row.target) if not pd.isna(row.target) else None,
                    'regime': row.regime if not pd.isna(row.regime) else None,
                    'entry_signal': int(row.entry_signal) if not pd.isna(row.entry_signal) else None,
                    'local_peak': int(row.local_peak) if not pd.isna(row.local_peak) else None
                }
            }),  # metadata as JSON
            row.data_category, row.interval
        )
        for _, row in df.iterrows()
    ]
    
    # Insert data
    try:
        # First try with ON CONFLICT
        try:
            execute_values(
                cursor,
                """
                INSERT INTO market_data 
                (time, symbol, open, high, low, close, volume, vwap, num_trades, source, metadata, data_category, interval)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                metadata = EXCLUDED.metadata
                """,
                data
            )
        except Exception as e:
            # If ON CONFLICT fails, try without it
            logger.warning(f"ON CONFLICT clause failed: {str(e)}, trying without it")
            execute_values(
                cursor,
                """
                INSERT INTO market_data 
                (time, symbol, open, high, low, close, volume, vwap, num_trades, source, metadata, data_category, interval)
                VALUES %s
                """,
                data
            )
        
        logger.info(f"Inserted {len(data)} rows for {df.symbol.iloc[0]}")
    except Exception as e:
        logger.error(f"Error inserting data: {str(e)}")
    finally:
        cursor.close()

def insert_features_data(conn, df):
    """
    Insert features data into the features_data table.
    
    Args:
        conn: Database connection
        df: DataFrame with features
    """
    cursor = conn.cursor()
    
    # Prepare feature data for insertion
    feature_data = []
    
    # Technical indicator columns to insert as features
    feature_columns = [
        'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_200',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_middle_20', 'bb_upper_20', 'bb_lower_20', 'bb_width_20',
        'atr_14', 'volume_ma_20', 'volume_ratio',
        'return_5d', 'return_10d', 'return_20d'
    ]
    
    for _, row in df.iterrows():
        for feature_name in feature_columns:
            if feature_name in df.columns and not pd.isna(row[feature_name]):
                feature_group = 'price_action' if 'return' in feature_name else \
                              'moving_average' if 'ma_' in feature_name else \
                              'oscillator' if 'rsi' in feature_name or 'macd' in feature_name else \
                              'volatility' if 'bb_' in feature_name or 'atr' in feature_name else \
                              'volume' if 'volume' in feature_name else 'other'
                
                feature_data.append((
                    row.time, row.symbol, feature_name, float(row[feature_name]), 
                    feature_group, None
                ))
    
    # Insert feature data
    try:
        if feature_data:
            # Try with ON CONFLICT first
            try:
                execute_values(
                    cursor,
                    """
                    INSERT INTO features_data 
                    (time, symbol, feature_name, feature_value, feature_group, metadata)
                    VALUES %s
                    ON CONFLICT (time, symbol, feature_name) DO UPDATE SET
                    feature_value = EXCLUDED.feature_value,
                    feature_group = EXCLUDED.feature_group
                    """,
                    feature_data
                )
            except Exception as e:
                # If ON CONFLICT fails, try without it
                logger.warning(f"ON CONFLICT clause failed for features: {str(e)}, trying without it")
                execute_values(
                    cursor,
                    """
                    INSERT INTO features_data 
                    (time, symbol, feature_name, feature_value, feature_group, metadata)
                    VALUES %s
                    """,
                    feature_data
                )
                
            logger.info(f"Inserted {len(feature_data)} features for {df.symbol.iloc[0]}")
    except Exception as e:
        logger.error(f"Error inserting feature data: {str(e)}")
    finally:
        cursor.close()

def add_columns_if_needed(conn):
    """
    Add required columns to market_data table if they don't exist.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    try:
        # Check if record_type column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'market_data' AND column_name = 'record_type'
        """)
        
        if cursor.fetchone() is None:
            # Add column
            cursor.execute("ALTER TABLE market_data ADD COLUMN record_type TEXT")
            logger.info("Added record_type column to market_data table")
            
            # Update existing rows
            cursor.execute("UPDATE market_data SET record_type = 'OHLCV'")
            logger.info("Updated existing rows with record_type = 'OHLCV'")
            
            # Create index
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_record_type ON market_data (record_type)")
            logger.info("Created index on record_type column")
        
        # Check if data_category column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'market_data' AND column_name = 'data_category'
        """)
        
        if cursor.fetchone() is None:
            # Add column
            cursor.execute("ALTER TABLE market_data ADD COLUMN data_category TEXT")
            logger.info("Added data_category column to market_data table")
            
            # Update existing rows
            cursor.execute("UPDATE market_data SET data_category = 'OHLCV'")
            logger.info("Updated existing rows with data_category = 'OHLCV'")
            
            # Create index
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_data_category ON market_data (data_category)")
            logger.info("Created index on data_category column")
        
        # Check if interval column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'market_data' AND column_name = 'interval'
        """)
        
        if cursor.fetchone() is None:
            # Add column
            cursor.execute("ALTER TABLE market_data ADD COLUMN interval TEXT")
            logger.info("Added interval column to market_data table")
            
            # Update existing rows
            cursor.execute("UPDATE market_data SET interval = '1d'")
            logger.info("Updated existing rows with interval = '1d'")
            
            # Create index
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_interval ON market_data (interval)")
            logger.info("Created index on interval column")
            
        # Add unique constraint on time and symbol if it doesn't exist
        cursor.execute("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'market_data' AND constraint_type = 'UNIQUE'
        """)
        
        if cursor.fetchone() is None:
            try:
                cursor.execute("""
                    ALTER TABLE market_data ADD CONSTRAINT market_data_time_symbol_key UNIQUE (time, symbol)
                """)
                logger.info("Added unique constraint on time and symbol")
            except Exception as e:
                logger.warning(f"Could not add unique constraint: {str(e)}")
    except Exception as e:
        logger.error(f"Error adding columns: {str(e)}")
    finally:
        cursor.close()

def main():
    """Main function to populate the database."""
    logger.info("Starting database population")
    
    # Connect to the database
    conn = connect_to_db()
    
    # Add required columns if needed
    add_columns_if_needed(conn)
    
    # Create features table if needed
    create_features_table_if_needed(conn)
    
    # Define parameters
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Generate and insert data for each symbol
    for symbol in symbols:
        logger.info(f"Generating data for {symbol}")
        
        # Set initial price based on symbol
        if symbol == 'SPY':
            initial_price = 400.0
        elif symbol == 'QQQ':
            initial_price = 350.0
        elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
            initial_price = np.random.uniform(150.0, 300.0)
        else:
            initial_price = np.random.uniform(50.0, 150.0)
        
        # Generate price data
        df = create_synthetic_price_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_price=initial_price,
            volatility=0.015 if symbol in ['SPY', 'QQQ'] else 0.025
        )
        
        # Add technical indicators and target columns
        df = add_technical_indicators(df)
        
        # Insert market data
        insert_market_data(conn, df)
        
        # Insert features data
        insert_features_data(conn, df)
    
    # Close connection
    conn.close()
    logger.info("Database population completed")

if __name__ == "__main__":
    main()
