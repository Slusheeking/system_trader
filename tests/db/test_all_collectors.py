#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Collector Test Script
----------------------------------
Tests all data collectors with all their endpoints, processes the data,
and verifies storage in both TimescaleDB and Redis cache.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, List, Set
import json

# Import system modules
from config.collector_config import CollectorConfig, get_collector_config
from data.collectors.factory import CollectorFactory
from data.collectors.schema import StandardRecord, RecordType
from data.database.timeseries_db import TimeseriesDBClient
from data.database.redis_client import RedisClient
from data.processors.data_cleaner import DataCleaner
from data.processors.data_validator import DataValidator
from data.processors.data_cache import DataCache
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('test_all_collectors', log_level=logging.INFO)

# Test symbols - using popular stocks with good data availability
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Test data types for each collector
COLLECTOR_DATA_TYPES = {
    'polygon': ['bars', 'trades', 'quotes', 'snapshot'],
    'alpaca': ['bars', 'trades', 'quotes', 'account', 'positions', 'orders'],
    'yahoo': ['quotes', 'historical', 'options'],
    'unusual_whales': ['flow', 'news']
}

def setup_api_keys():
    """
    Check and setup API keys from environment variables and api_credentials.yaml.
    
    Returns:
        Dict of API keys status
    """
    # Try to load API keys from api_credentials.yaml first
    try:
        from utils.config_loader import ConfigLoader
        credentials_path = 'config/api_credentials.yaml'
        if os.path.exists(credentials_path):
            credentials = ConfigLoader.load_yaml(credentials_path)
            
            # Set environment variables from credentials file if they don't exist
            if 'Polygon' in credentials and 'API_Key' in credentials['Polygon']:
                polygon_key = credentials['Polygon']['API_Key']
                os.environ['POLYGON_API_KEY'] = polygon_key
                logger.info(f"Using Polygon API key from credentials file: {polygon_key[:4]}...{polygon_key[-4:]}")
                
            if 'Alpaca' in credentials:
                if 'API_Key' in credentials['Alpaca'] and not os.environ.get('ALPACA_API_KEY'):
                    os.environ['ALPACA_API_KEY'] = credentials['Alpaca']['API_Key']
                if 'Secret_Key' in credentials['Alpaca'] and not os.environ.get('ALPACA_API_SECRET'):
                    os.environ['ALPACA_API_SECRET'] = credentials['Alpaca']['Secret_Key']
                    
            if 'Yahoo' in credentials and 'API_Key' in credentials['Yahoo'] and not os.environ.get('YAHOO_API_KEY'):
                os.environ['YAHOO_API_KEY'] = credentials['Yahoo']['API_Key']
                    
            if 'Unusual_Whales' in credentials and 'API_Token' in credentials['Unusual_Whales'] and not os.environ.get('UNUSUAL_WHALES_API_KEY'):
                os.environ['UNUSUAL_WHALES_API_KEY'] = credentials['Unusual_Whales']['API_Token']
                
            logger.info("Loaded API credentials from config/api_credentials.yaml")
    except Exception as e:
        logger.warning(f"Could not load API credentials from file: {str(e)}")
    
    # Get API keys from environment variables (which may have been set from the credentials file)
    api_keys = {
        'polygon': os.environ.get('POLYGON_API_KEY', ''),
        'alpaca': {
            'key': os.environ.get('ALPACA_API_KEY', ''),
            'secret': os.environ.get('ALPACA_API_SECRET', '')
        },
        'yahoo': os.environ.get('YAHOO_API_KEY', ''),
        'unusual_whales': os.environ.get('UNUSUAL_WHALES_API_KEY', '')
    }
    
    # Check API keys
    api_status = {}
    for api, key in api_keys.items():
        if api == 'alpaca':
            status = bool(key['key'] and key['secret'])
        else:
            status = bool(key)
        
        api_status[api] = status
        if status:
            logger.info(f"✓ {api.capitalize()} API key found")
        else:
            logger.warning(f"✗ {api.capitalize()} API key missing")
    
    return api_status

def collect_data(collector_name: str, symbols: List[str], data_types: List[str], start_date: datetime, end_date: datetime):
    """
    Collect data from a specific API for all specified data types.
    
    Args:
        collector_name: Name of the collector to use
        symbols: List of symbols to collect data for
        data_types: List of data types to collect
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        Dict mapping data types to lists of StandardRecord objects
    """
    logger.info(f"Collecting data from {collector_name} for {len(symbols)} symbols and {len(data_types)} data types")
    
    # Get collector configuration
    config = get_collector_config(collector_name)
    
    # Add symbols to config
    config_dict = config.config if hasattr(config, 'config') else {}
    config_dict['symbols'] = symbols
    
    # Enable only the specified data types
    if 'data_types' in config_dict:
        for data_type in config_dict['data_types']:
            if data_type in data_types:
                config_dict['data_types'][data_type]['enabled'] = True
            else:
                config_dict['data_types'][data_type]['enabled'] = False
    
    config = CollectorConfig(config_dict)
    
    # Create collector instance
    collector = CollectorFactory.create(collector_name, config)
    
    all_records_by_type = {data_type: [] for data_type in data_types}
    
    for symbol in symbols:
        try:
            logger.info(f"Collecting {collector_name} data for {symbol}")
            
            # Set symbol if the collector has that method
            if hasattr(collector, 'set_symbol'):
                collector.set_symbol(symbol)
            
            # For each data type, collect data
            for data_type in data_types:
                try:
                    logger.info(f"Collecting {data_type} data for {symbol}")
                    
                    # Set the data type if the collector supports it
                    if hasattr(collector, 'set_data_type'):
                        collector.set_data_type(data_type)
                    
                    # Collect data
                    if hasattr(collector, 'collect_data_type'):
                        records = collector.collect_data_type(data_type, start_date, end_date)
                    else:
                        # Fall back to standard collect method
                        records = collector.collect(start_date, end_date)
                    
                    logger.info(f"Collected {len(records)} {data_type} records for {symbol}")
                    all_records_by_type[data_type].extend(records)
                    
                except Exception as e:
                    logger.error(f"Error collecting {data_type} data for {symbol}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error collecting {collector_name} data for {symbol}: {str(e)}")
    
    return all_records_by_type

def process_data(records: List[StandardRecord]):
    """
    Process collected data using data processors.
    
    Args:
        records: List of StandardRecord objects
        
    Returns:
        Processed DataFrame
    """
    if not records:
        logger.warning("No records to process")
        return pd.DataFrame()
    
    # Convert records to DataFrame
    data = []
    for record in records:
        record_dict = record.model_dump()
        # Convert extended_data to a string to avoid unhashable type error
        if 'extended_data' in record_dict and isinstance(record_dict['extended_data'], dict):
            record_dict['extended_data'] = str(record_dict['extended_data'])
        data.append(record_dict)
    
    df = pd.DataFrame(data)
    
    # Explicitly convert OHLC columns to numeric types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'price', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Processing {len(df)} records")
    
    # Create data processors
    cleaner = DataCleaner()
    validator = DataValidator()
    
    # Process data
    try:
        # Validate data
        validation_results = validator.validate(df)
        logger.info(f"Validated data: {len(df)} records passed validation")
        
        # Clean data
        df = cleaner.clean(df)
        logger.info(f"Cleaned data: {len(df)} records after cleaning")
        
        # Remove duplicates
        df = cleaner.remove_duplicates(df)
        logger.info(f"Removed duplicates: {len(df)} unique records")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def store_in_database(df: pd.DataFrame, db_client: TimeseriesDBClient, data_type: str):
    """
    Store processed data in the TimescaleDB database.
    
    Args:
        df: Processed DataFrame
        db_client: Database client
        data_type: Type of data being stored
        
    Returns:
        Number of records stored
    """
    if df.empty:
        logger.warning(f"No {data_type} data to store")
        return 0
    
    logger.info(f"Storing {len(df)} {data_type} records in database")
    
    try:
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Determine record type and store in appropriate table
        if 'record_type' in df.columns:
            # Group by record type
            aggregate_records = [r for r in records if r.get('record_type') == 'aggregate']
            bar_records = [r for r in records if r.get('record_type') == 'bar']
            trade_records = [r for r in records if r.get('record_type') == 'trade']
            quote_records = [r for r in records if r.get('record_type') == 'quote']
            
            # Store market data (bars and aggregates)
            market_data = []
            
            # Process bar records
            for record in bar_records:
                market_data.append({
                    'time': record.get('timestamp'),
                    'symbol': record.get('symbol'),
                    'open': record.get('open'),
                    'high': record.get('high'),
                    'low': record.get('low'),
                    'close': record.get('close'),
                    'volume': record.get('volume'),
                    'vwap': record.get('vwap'),
                    'num_trades': record.get('num_trades'),
                    'source': record.get('source'),
                    'metadata': record.get('extended_data')
                })
            
            # Process aggregate records
            for record in aggregate_records:
                # Parse transactions from extended_data
                num_trades = None
                extended_data = record.get('extended_data')
                
                # Handle extended_data properly for JSON serialization
                if isinstance(extended_data, str) and 'transactions' in extended_data:
                    # Try to extract transactions from the string representation
                    import re
                    match = re.search(r"'transactions':\s*(\d+)", extended_data)
                    if match:
                        num_trades = int(match.group(1))
                    
                    # Convert string representation to proper JSON
                    try:
                        # Replace single quotes with double quotes for JSON compatibility
                        extended_data = extended_data.replace("'", '"')
                        # Handle None values which aren't valid JSON
                        extended_data = extended_data.replace("None", "null")
                    except Exception as e:
                        logger.warning(f"Error formatting extended_data: {str(e)}")
                        extended_data = "{}"
                
                market_data.append({
                    'time': record.get('timestamp'),
                    'symbol': record.get('symbol'),
                    'open': record.get('open'),
                    'high': record.get('high'),
                    'low': record.get('low'),
                    'close': record.get('close'),
                    'volume': record.get('volume'),
                    'vwap': record.get('vwap'),
                    'num_trades': num_trades,
                    'source': record.get('source'),
                    'metadata': extended_data
                })
            
            # Insert market data
            if market_data:
                count = db_client.insert_market_data(market_data)
                logger.info(f"Inserted {count} records into market_data table")
            
            # Store trade data
            if trade_records:
                # Convert to trade_data format
                trade_data = []
                for record in trade_records:
                    trade_data.append({
                        'time': record.get('timestamp'),
                        'order_id': record.get('trade_id', ''),
                        'trade_id': record.get('trade_id'),
                        'symbol': record.get('symbol'),
                        'side': record.get('side', 'buy'),
                        'quantity': record.get('volume', 0),
                        'price': record.get('price', 0),
                        'source': record.get('source'),
                        'metadata': record.get('extended_data')
                    })
                
                count = db_client.insert_trade_data(trade_data)
                logger.info(f"Inserted {count} records into trade_data table")
            
            # Store quote data
            if quote_records:
                # For now, quotes are stored as analytics data
                analytics_data = []
                for record in quote_records:
                    analytics_data.append({
                        'time': record.get('timestamp'),
                        'metric_name': 'quote',
                        'metric_value': (record.get('bid_price', 0) + record.get('ask_price', 0)) / 2 if record.get('bid_price') and record.get('ask_price') else None,
                        'symbol': record.get('symbol'),
                        'dimension': 'price',
                        'metadata': record.get('extended_data')
                    })
                
                count = db_client.insert_analytics_data(analytics_data)
                logger.info(f"Inserted {count} quote records into analytics_data table")
            
            return len(market_data) + len(trade_records) + len(quote_records)
        else:
            # Assume all records are market data
            market_data = []
            for record in records:
                market_data.append({
                    'time': record.get('timestamp', datetime.now()),
                    'symbol': record.get('symbol', ''),
                    'open': record.get('open'),
                    'high': record.get('high'),
                    'low': record.get('low'),
                    'close': record.get('close'),
                    'volume': record.get('volume'),
                    'vwap': record.get('vwap'),
                    'num_trades': record.get('num_trades'),
                    'source': record.get('source', 'unknown'),
                    'metadata': record.get('extended_data')
                })
            
            count = db_client.insert_market_data(market_data)
            logger.info(f"Inserted {count} records into market_data table")
            return count
    
    except Exception as e:
        logger.error(f"Error storing {data_type} data: {str(e)}")
        return 0

def store_in_redis(df: pd.DataFrame, redis_client: RedisClient, collector_name: str, data_type: str):
    """
    Store processed data in Redis cache.
    
    Args:
        df: Processed DataFrame
        redis_client: Redis client
        collector_name: Name of the collector
        data_type: Type of data being stored
        
    Returns:
        Number of records stored
    """
    if df.empty:
        logger.warning(f"No {data_type} data to cache in Redis")
        return 0
    
    logger.info(f"Caching {len(df)} {data_type} records in Redis")
    
    try:
        # Group data by symbol
        symbols = df['symbol'].unique()
        count = 0
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]
            
            # Create cache key
            cache_key = f"{collector_name}:{data_type}:{symbol}"
            
            # Convert to JSON
            json_data = symbol_data.to_json(orient='records', date_format='iso')
            
            # Store in Redis with 1 hour expiration
            redis_client.set(cache_key, json_data, 3600)
            count += len(symbol_data)
            
            logger.info(f"Cached {len(symbol_data)} {data_type} records for {symbol} in Redis")
        
        return count
    
    except Exception as e:
        logger.error(f"Error caching {data_type} data in Redis: {str(e)}")
        return 0

def review_database_data(db_client: TimeseriesDBClient, symbols: List[str]):
    """
    Review the actual data stored in the database to ensure it's correct.
    
    Args:
        db_client: Database client
        symbols: List of symbols to check
        
    Returns:
        Boolean indicating if data looks valid
    """
    logger.info("Reviewing database data quality")
    
    try:
        for symbol in symbols:
            # Get the latest market data for the symbol
            query = f"""
            SELECT * FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY time DESC
            LIMIT 5
            """
            result = db_client.execute_query(query)
            
            if not result:
                logger.warning(f"No market data found for {symbol}")
                continue
            
            # Check if the data has the expected fields
            logger.info(f"Reviewing market data for {symbol}:")
            for i, row in enumerate(result):
                # Check for required fields
                required_fields = ['time', 'symbol', 'open', 'high', 'low', 'close']
                missing_fields = [field for field in required_fields if field not in row or row[field] is None]
                
                if missing_fields:
                    logger.warning(f"Row {i+1} is missing required fields: {missing_fields}")
                    continue
                
                # Check if OHLC values make sense
                if not (row['low'] <= row['open'] <= row['high'] and 
                        row['low'] <= row['close'] <= row['high']):
                    logger.warning(f"Row {i+1} has inconsistent OHLC values: open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}")
                
                # Check if volume is positive
                if 'volume' in row and row['volume'] is not None and row['volume'] <= 0:
                    logger.warning(f"Row {i+1} has invalid volume: {row['volume']}")
                
                # Log the data for review
                logger.info(f"  Row {i+1}: time={row['time']}, open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}, volume={row.get('volume')}")
            
            # Check trade data if available
            trade_query = f"""
            SELECT * FROM trade_data 
            WHERE symbol = '{symbol}'
            ORDER BY time DESC
            LIMIT 5
            """
            trade_result = db_client.execute_query(trade_query)
            
            if trade_result:
                logger.info(f"Reviewing trade data for {symbol}:")
                for i, row in enumerate(trade_result):
                    logger.info(f"  Trade {i+1}: time={row['time']}, price={row['price']}, quantity={row['quantity']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error reviewing database data: {str(e)}")
        return False

def verify_database_storage(db_client: TimeseriesDBClient, collector_name: str, symbols: List[str], data_types: List[str]):
    """
    Verify that data was correctly stored in the database.
    
    Args:
        db_client: Database client
        collector_name: Name of the collector
        symbols: List of symbols
        data_types: List of data types
        
    Returns:
        Boolean indicating success
    """
    logger.info(f"Verifying database storage for {collector_name}")
    
    try:
        # Check market data table
        market_data_types = ['bars', 'aggregate', 'snapshot']
        trade_data_types = ['trades']
        quote_data_types = ['quotes']
        
        for symbol in symbols:
            # Check if we have market data
            if any(dt in data_types for dt in market_data_types):
                query = f"""
                SELECT COUNT(*) FROM market_data 
                WHERE symbol = '{symbol}'
                """
                result = db_client.execute_query(query)
                count = result[0]['count'] if result else 0
                
                if count > 0:
                    logger.info(f"Found {count} market data records for {symbol} from {collector_name}")
                else:
                    logger.warning(f"No market data found for {symbol} from {collector_name}")
            
            # Check if we have trade data
            if any(dt in data_types for dt in trade_data_types):
                query = f"""
                SELECT COUNT(*) FROM trade_data 
                WHERE symbol = '{symbol}'
                """
                result = db_client.execute_query(query)
                count = result[0]['count'] if result else 0
                
                if count > 0:
                    logger.info(f"Found {count} trade data records for {symbol} from {collector_name}")
                else:
                    logger.warning(f"No trade data found for {symbol} from {collector_name}")
            
            # Check if we have quote data (in analytics table)
            if any(dt in data_types for dt in quote_data_types):
                query = f"""
                SELECT COUNT(*) FROM analytics_data 
                WHERE symbol = '{symbol}' AND metric_name = 'quote'
                """
                result = db_client.execute_query(query)
                count = result[0]['count'] if result else 0
                
                if count > 0:
                    logger.info(f"Found {count} quote data records for {symbol} from {collector_name}")
                else:
                    logger.warning(f"No quote data found for {symbol} from {collector_name}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error verifying database storage: {str(e)}")
        return False

def verify_redis_storage(redis_client: RedisClient, collector_name: str, symbols: List[str], data_types: List[str]):
    """
    Verify that data was correctly cached in Redis.
    
    Args:
        redis_client: Redis client
        collector_name: Name of the collector
        symbols: List of symbols
        data_types: List of data types
        
    Returns:
        Boolean indicating success
    """
    logger.info(f"Verifying Redis cache for {collector_name}")
    
    try:
        success = True
        
        for symbol in symbols:
            for data_type in data_types:
                # Create cache key
                cache_key = f"{collector_name}:{data_type}:{symbol}"
                
                # Check if key exists
                exists = redis_client.exists(cache_key)
                
                if exists:
                    # Get data and check if it exists
                    cached_data = redis_client.get(cache_key)
                    if cached_data:
                        logger.info(f"Found cached {data_type} records for {symbol} from {collector_name}")
                    else:
                        logger.warning(f"Invalid data in Redis for {cache_key}")
                        success = False
                else:
                    logger.warning(f"No cached data found for {cache_key}")
                    success = False
        
        return success
    
    except Exception as e:
        logger.error(f"Error verifying Redis cache: {str(e)}")
        return False

def test_collector(collector_name: str, symbols: List[str], days: int, db_client: TimeseriesDBClient, redis_client: RedisClient):
    """
    Test a specific collector with all its data types.
    
    Args:
        collector_name: Name of the collector to test
        symbols: List of symbols to collect data for
        days: Number of days of historical data to collect
        db_client: Database client
        redis_client: Redis client
        
    Returns:
        Dict with test results
    """
    logger.info(f"Testing {collector_name} collector")
    
    # Setup date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get data types for this collector
    data_types = COLLECTOR_DATA_TYPES.get(collector_name, [])
    
    # Collect data for all data types
    all_records_by_type = collect_data(collector_name, symbols, data_types, start_date, end_date)
    
    results = {
        'collector': collector_name,
        'symbols': symbols,
        'data_types': {},
        'database_verified': False,
        'redis_verified': False
    }
    
    # Process and store each data type
    for data_type, records in all_records_by_type.items():
        if not records:
            logger.error(f"No {data_type} data collected from {collector_name}")
            results['data_types'][data_type] = {
                'collected': 0,
                'processed': 0,
                'stored_db': 0,
                'stored_redis': 0
            }
            continue
        
        # Process data
        df = process_data(records)
        
        if df.empty:
            logger.error(f"Failed to process {data_type} data from {collector_name}")
            results['data_types'][data_type] = {
                'collected': len(records),
                'processed': 0,
                'stored_db': 0,
                'stored_redis': 0
            }
            continue
        
        # Store in database
        db_count = store_in_database(df, db_client, data_type)
        
        # Store in Redis
        redis_count = store_in_redis(df, redis_client, collector_name, data_type)
        
        results['data_types'][data_type] = {
            'collected': len(records),
            'processed': len(df),
            'stored_db': db_count,
            'stored_redis': redis_count
        }
    
    # Verify storage
    results['database_verified'] = verify_database_storage(db_client, collector_name, symbols, data_types)
    results['redis_verified'] = verify_redis_storage(redis_client, collector_name, symbols, data_types)
    
    # Review the actual data in the database
    results['data_quality_verified'] = review_database_data(db_client, symbols)
    
    logger.info(f"Completed testing {collector_name} collector")
    return results

def main():
    """Main function to run the collector tests."""
    parser = argparse.ArgumentParser(description='Comprehensive Collector Test')
    parser.add_argument('--collectors', type=str, default='all',
                      help='Comma-separated list of collectors to test (default: all)')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL',
                      help='Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)')
    parser.add_argument('--days', type=int, default=1,
                      help='Number of days of historical data to collect (default: 1)')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Setup API keys
    api_status = setup_api_keys()
    
    # Create database client
    db_client = TimeseriesDBClient()
    
    # Create Redis client
    redis_client = RedisClient()
    
    # Determine which collectors to test
    collectors_to_test = []
    if args.collectors.lower() == 'all':
        collectors_to_test = [api for api, status in api_status.items() if status]
    else:
        for collector in args.collectors.split(','):
            collector = collector.strip().lower()
            if api_status.get(collector, False):
                collectors_to_test.append(collector)
            else:
                logger.error(f"Cannot test {collector} collector: API key missing")
    
    if not collectors_to_test:
        logger.error("No collectors available to test. Please set API keys as environment variables.")
        return 1
    
    # Test each collector
    results = {}
    for collector in collectors_to_test:
        results[collector] = test_collector(collector, symbols, args.days, db_client, redis_client)
    
    # Print summary
    logger.info("Collector Testing Summary:")
    for collector, result in results.items():
        status = "✓ Success" if all(dt['stored_db'] > 0 for dt in result['data_types'].values()) else "✗ Failed"
        logger.info(f"{collector}: {status}")
        
        for data_type, counts in result['data_types'].items():
            logger.info(f"  {data_type}: Collected {counts['collected']}, Processed {counts['processed']}, Stored DB {counts['stored_db']}, Stored Redis {counts['stored_redis']}")
        
        logger.info(f"  Database Verification: {'✓ Success' if result['database_verified'] else '✗ Failed'}")
        logger.info(f"  Redis Verification: {'✓ Success' if result['redis_verified'] else '✗ Failed'}")
        logger.info(f"  Data Quality Verification: {'✓ Success' if result['data_quality_verified'] else '✗ Failed'}")
    
    # Overall success if at least one collector succeeded
    return 0 if any(all(dt['stored_db'] > 0 for dt in result['data_types'].values()) for result in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
