#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test API Data Collector Script
----------------------------
Tests each API data collector, processes the data, and stores it in the database.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, List

# Import system modules
from config.collector_config import CollectorConfig, get_collector_config
from data.collectors.factory import CollectorFactory
from data.collectors.schema import StandardRecord
from data.database.timeseries_db import TimeseriesDBClient
from data.processors.data_cleaner import DataCleaner
from data.processors.data_validator import DataValidator
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('test_api_data_collector', log_level=logging.INFO)

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

def collect_data(collector_name: str, symbols: List[str], start_date: datetime, end_date: datetime):
    """
    Collect data from a specific API.
    
    Args:
        collector_name: Name of the collector to use
        symbols: List of symbols to collect data for
        start_date: Start date for data collection
        end_date: End date for data collection
        
    Returns:
        List of StandardRecord objects
    """
    logger.info(f"Collecting data from {collector_name} for {len(symbols)} symbols")
    
    # Get collector configuration
    config = get_collector_config(collector_name)
    
    # Add symbols to config
    config_dict = config.config if hasattr(config, 'config') else {}
    config_dict['symbols'] = symbols
    config = CollectorConfig(config_dict)
    
    all_records = []
    
    for symbol in symbols:
        try:
            # Create collector instance with symbol in config
            collector = CollectorFactory.create(collector_name, config)
            
            # Collect data
            logger.info(f"Collecting {collector_name} data for {symbol}")
            
            # Handle different collector interfaces
            if hasattr(collector, 'set_symbol'):
                # For collectors with separate set_symbol method (like PolygonCollector)
                collector.set_symbol(symbol)
                records = collector.collect(start_date, end_date)
            else:
                # For collectors with the old interface
                records = collector.collect(symbol, start_date, end_date)
            
            logger.info(f"Collected {len(records)} records for {symbol}")
            all_records.extend(records)
            
        except Exception as e:
            logger.error(f"Error collecting {collector_name} data for {symbol}: {str(e)}")
    
    return all_records

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
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap']
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

def store_data(df: pd.DataFrame, db_client: TimeseriesDBClient):
    """
    Store processed data in the database.
    
    Args:
        df: Processed DataFrame
        db_client: Database client
        
    Returns:
        Number of records stored
    """
    if df.empty:
        logger.warning("No data to store")
        return 0
    
    logger.info(f"Storing {len(df)} records in database")
    
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
        logger.error(f"Error storing data: {str(e)}")
        return 0

def test_api(api_name: str, symbols: List[str], days: int, db_client: TimeseriesDBClient):
    """
    Test a specific API by collecting, processing, and storing data.
    
    Args:
        api_name: Name of the API to test
        symbols: List of symbols to collect data for
        days: Number of days of historical data to collect
        db_client: Database client
        
    Returns:
        Success status
    """
    logger.info(f"Testing {api_name} API")
    
    # Setup date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Collect data
    records = collect_data(api_name, symbols, start_date, end_date)
    
    if not records:
        logger.error(f"No data collected from {api_name}")
        return False
    
    # Process data
    df = process_data(records)
    
    if df.empty:
        logger.error(f"Failed to process {api_name} data")
        return False
    
    # Store data
    count = store_data(df, db_client)
    
    if count == 0:
        logger.error(f"Failed to store {api_name} data")
        return False
    
    logger.info(f"Successfully tested {api_name} API: collected, processed, and stored {count} records")
    return True

def main():
    """Main function to run the API data collector."""
    parser = argparse.ArgumentParser(description='Test API Data Collector')
    parser.add_argument('--api', type=str, choices=['polygon', 'yahoo', 'alpaca', 'unusual_whales', 'all'],
                      default='all', help='API to test (default: all)')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL',
                      help='Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)')
    parser.add_argument('--days', type=int, default=7,
                      help='Number of days of historical data to collect (default: 7)')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Setup API keys
    api_status = setup_api_keys()
    
    # Create database client
    db_client = TimeseriesDBClient()
    
    # Determine which APIs to test
    apis_to_test = []
    if args.api == 'all':
        apis_to_test = [api for api, status in api_status.items() if status]
    else:
        if api_status.get(args.api, False):
            apis_to_test = [args.api]
        else:
            logger.error(f"Cannot test {args.api} API: API key missing")
            return 1
    
    if not apis_to_test:
        logger.error("No APIs available to test. Please set API keys as environment variables.")
        return 1
    
    # Test each API
    results = {}
    for api in apis_to_test:
        results[api] = test_api(api, symbols, args.days, db_client)
    
    # Print summary
    logger.info("API Testing Summary:")
    for api, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{api}: {status}")
    
    # Overall success if at least one API succeeded
    return 0 if any(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
