#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time Data Provider
----------------------
Provides access to real-time market data from Redis cache for ML models.
Handles both WebSocket streaming data and historical data retrieval.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Set, Tuple
import pandas as pd
import time
import json
from enum import Enum

from data.collectors.schema import RecordType
from data.database.redis_client import get_redis_client
from data.database.timeseries_db import get_timescale_client
from data.processors.data_cache import get_data_cache
from data.processors.data_cleaner import get_data_cleaner
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('realtime_data_provider', category='data')

# Initialize clients
redis_client = get_redis_client()
timescale_client = get_timescale_client()
data_cache = get_data_cache()
data_cleaner = get_data_cleaner()


class DataSource(Enum):
    """Enumeration of data sources."""
    POLYGON = "polygon"
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    UNUSUAL_WHALES = "unusual_whales"
    REDDIT = "reddit"
    ANY = "any"  # Try all sources


class RealtimeDataProvider:
    """
    Provider for real-time market data from Redis cache and database.
    Supports both WebSocket streaming data and historical/REST API data.
    """

    @staticmethod
    def get_latest_data(symbol: str, record_type: RecordType, 
                      source: DataSource = DataSource.ANY) -> Optional[Dict[str, Any]]:
        """
        Get the latest data for a symbol and record type.
        
        Args:
            symbol: Symbol to get data for
            record_type: Type of record to get
            source: Data source to use (default: try all sources)
            
        Returns:
            Latest data or None if not available
        """
        symbol = symbol.upper()
        type_str = record_type.value
        
        # Try the specified source first
        if source != DataSource.ANY:
            data = RealtimeDataProvider._get_latest_from_source(symbol, type_str, source.value)
            if data:
                return data
        
        # Try all sources if specified source doesn't have data or ANY is specified
        for src in DataSource:
            if src != DataSource.ANY:
                data = RealtimeDataProvider._get_latest_from_source(symbol, type_str, src.value)
                if data:
                    return data
        
        logger.warning(f"No latest {type_str} data found for {symbol} in any source")
        return None
    
    @staticmethod
    def _get_latest_from_source(symbol: str, type_str: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest data from a specific source.
        
        Args:
            symbol: Symbol to get data for
            type_str: Record type string
            source: Source name
            
        Returns:
            Latest data or None if not available
        """
        # Try WebSocket data first (newest)
        key = f"ws:{type_str}:{symbol}"
        latest_ws = data_cache.get_latest_websocket_data(symbol, type_str, 1)
        
        if latest_ws and len(latest_ws) > 0:
            logger.debug(f"Retrieved latest {type_str} data for {symbol} from WebSocket cache")
            return latest_ws[0]
        
        # Then try latest data from REST API cache
        key = f"{source}:latest:{type_str}:{symbol}"
        data = redis_client.get(key)
        
        if data:
            logger.debug(f"Retrieved latest {type_str} data for {symbol} from {source} Redis cache")
            return data
        
        # Finally, try database
        if type_str == 'bars' or type_str == 'trades' or type_str == 'quotes':
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)
                
                # Query the database for the latest record
                if type_str == 'bars':
                    df = timescale_client.query_market_data(symbol, start_time, end_time)
                elif type_str == 'trades':
                    df = timescale_client.query_market_data(symbol, start_time, end_time)
                elif type_str == 'quotes':
                    df = timescale_client.query_market_data(symbol, start_time, end_time)
                
                if not df.empty:
                    # Get the latest row
                    latest = df.iloc[-1].to_dict()
                    logger.debug(f"Retrieved latest {type_str} data for {symbol} from database")
                    return latest
            except Exception as e:
                logger.error(f"Error querying database for {symbol} {type_str}: {str(e)}")
        
        return None
    
    @staticmethod
    def get_recent_data(symbol: str, record_type: RecordType, 
                      limit: int = 100, source: DataSource = DataSource.ANY) -> List[Dict[str, Any]]:
        """
        Get recent data for a symbol and record type.
        
        Args:
            symbol: Symbol to get data for
            record_type: Type of record to get
            limit: Maximum number of records to return
            source: Data source to use (default: try all sources)
            
        Returns:
            List of recent data records
        """
        symbol = symbol.upper()
        type_str = record_type.value
        
        # Try to get WebSocket data first (it's realtime)
        start_time = time.time() - 86400  # Last 24 hours
        end_time = time.time()
        
        websocket_data = data_cache.get_websocket_data_range(symbol, type_str, start_time, end_time)
        
        if websocket_data and len(websocket_data) >= limit:
            logger.debug(f"Retrieved {len(websocket_data)} recent {type_str} records for {symbol} from WebSocket cache")
            # Return the most recent 'limit' items
            return websocket_data[-limit:]
        
        # Try the specified source if we don't have enough from WebSocket
        if source != DataSource.ANY:
            key = f"{source.value}:websocket:{type_str}:{symbol}:recent"
            data = redis_client.lrange(key, 0, limit - 1)
            if data and len(data) > 0:
                logger.debug(f"Retrieved {len(data)} recent {type_str} records for {symbol} from {source.value} Redis cache")
                # If we got some data from WebSocket but not enough, combine them
                if websocket_data:
                    combined = websocket_data + data[:limit - len(websocket_data)]
                    return combined[:limit]
                return data
        
        # If we still don't have enough, try all sources
        if source == DataSource.ANY:
            for src in DataSource:
                if src != DataSource.ANY:
                    key = f"{src.value}:websocket:{type_str}:{symbol}:recent"
                    data = redis_client.lrange(key, 0, limit - 1)
                    if data and len(data) > 0:
                        logger.debug(f"Retrieved {len(data)} recent {type_str} records for {symbol} from {src.value} Redis cache")
                        # If we got some data from WebSocket but not enough, combine them
                        if websocket_data:
                            combined = websocket_data + data[:limit - len(websocket_data)]
                            return combined[:limit]
                        return data
        
        # If still not enough, query the database
        if record_type in [RecordType.BAR, RecordType.TRADE, RecordType.QUOTE]:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=7)  # Look back further
                
                # Query the database
                if record_type == RecordType.BAR:
                    df = timescale_client.query_market_data(symbol, start_time, end_time, limit=limit)
                elif record_type == RecordType.TRADE:
                    df = timescale_client.query_market_data(symbol, start_time, end_time, limit=limit)
                elif record_type == RecordType.QUOTE:
                    df = timescale_client.query_market_data(symbol, start_time, end_time, limit=limit)
                
                if not df.empty:
                    # Convert to list of dicts
                    db_data = df.to_dict('records')
                    logger.debug(f"Retrieved {len(db_data)} recent {type_str} records for {symbol} from database")
                    
                    # Combine with any WebSocket data
                    if websocket_data:
                        # Ensure no duplicates by timestamp if present
                        if 'timestamp' in db_data[0] and 'timestamp' in websocket_data[0]:
                            ws_timestamps = {item['timestamp'] for item in websocket_data}
                            filtered_db = [item for item in db_data if item['timestamp'] not in ws_timestamps]
                            combined = websocket_data + filtered_db
                        else:
                            combined = websocket_data + db_data
                        
                        return combined[:limit]
                    
                    return db_data[:limit]
            except Exception as e:
                logger.error(f"Error querying database for {symbol} {type_str}: {str(e)}")
        
        # Return whatever WebSocket data we have, even if less than limit
        if websocket_data:
            return websocket_data
        
        logger.warning(f"No recent {type_str} data found for {symbol}")
        return []
    
    @staticmethod
    def get_recent_dataframe(symbol: str, record_type: RecordType, 
                           limit: int = 100, source: DataSource = DataSource.ANY) -> pd.DataFrame:
        """
        Get recent data as a DataFrame for a symbol and record type.
        
        Args:
            symbol: Symbol to get data for
            record_type: Type of record to get
            limit: Maximum number of records to return
            source: Data source to use (default: try all sources)
            
        Returns:
            DataFrame with recent data
        """
        # Get recent data
        data = RealtimeDataProvider.get_recent_data(symbol, record_type, limit, source)
        
        if not data:
            logger.warning(f"No recent {record_type.value} data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if present
            if 'timestamp' in df.columns:
                # Handle both string and numeric timestamps
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Clean the data if needed
            if len(df) > 0:
                df = data_cleaner.clean(df)
            
            return df
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_latest_ohlcv(symbol: str, source: DataSource = DataSource.ANY) -> Optional[Dict[str, Any]]:
        """
        Get the latest OHLCV data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            source: Data source to use (default: try all sources)
            
        Returns:
            Dictionary with OHLCV data or None if not available
        """
        # Try to get aggregate data first
        data = RealtimeDataProvider.get_latest_data(symbol, RecordType.AGGREGATE, source)
        
        if not data:
            # Try to get bar data
            data = RealtimeDataProvider.get_latest_data(symbol, RecordType.BAR, source)
            
        if not data:
            logger.warning(f"No OHLCV data found for {symbol}")
            return None
            
        # Extract OHLCV data
        ohlcv = {
            'symbol': symbol,
            'timestamp': data.get('timestamp'),
            'open': data.get('open'),
            'high': data.get('high'),
            'low': data.get('low'),
            'close': data.get('close'),
            'volume': data.get('volume')
        }
        
        return ohlcv
    
    @staticmethod
    def get_latest_price(symbol: str, source: DataSource = DataSource.ANY) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            source: Data source to use (default: try all sources)
            
        Returns:
            Latest price or None if not available
        """
        # Try to get trade data first
        data = RealtimeDataProvider.get_latest_data(symbol, RecordType.TRADE, source)
        
        if data and 'price' in data:
            return float(data['price'])
            
        # Try to get quote data
        data = RealtimeDataProvider.get_latest_data(symbol, RecordType.QUOTE, source)
        
        if data and 'bid_price' in data and 'ask_price' in data:
            # Use mid price
            return (float(data['bid_price']) + float(data['ask_price'])) / 2
        elif data and 'bid' in data and 'ask' in data:
            # Some providers use different field names
            return (float(data['bid']) + float(data['ask'])) / 2
            
        # Try to get aggregate/bar data
        data = RealtimeDataProvider.get_latest_data(symbol, RecordType.AGGREGATE, source)
        
        if not data:
            data = RealtimeDataProvider.get_latest_data(symbol, RecordType.BAR, source)
            
        if data and 'close' in data:
            return float(data['close'])
            
        logger.warning(f"No price data found for {symbol}")
        return None
    
    @staticmethod
    def get_ohlcv_dataframe(symbol: str, limit: int = 100, 
                          source: DataSource = DataSource.ANY) -> pd.DataFrame:
        """
        Get OHLCV data as a DataFrame for a symbol.
        
        Args:
            symbol: Symbol to get data for
            limit: Maximum number of records to return
            source: Data source to use (default: try all sources)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try to get aggregate data first
        df = RealtimeDataProvider.get_recent_dataframe(symbol, RecordType.AGGREGATE, limit, source)
        
        if df.empty:
            # Try to get bar data
            df = RealtimeDataProvider.get_recent_dataframe(symbol, RecordType.BAR, limit, source)
            
        if df.empty:
            logger.warning(f"No OHLCV data found for {symbol}")
            return pd.DataFrame()
            
        # Select OHLCV columns
        ohlcv_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in ohlcv_columns if col in df.columns]
        
        if len(available_columns) < 5:  # Need at least timestamp + OHLC
            logger.warning(f"Insufficient OHLCV data for {symbol}")
            return pd.DataFrame()
            
        return df[available_columns]
    
    @staticmethod
    def get_multi_symbol_data(symbols: List[str], record_type: RecordType, 
                            source: DataSource = DataSource.ANY) -> Dict[str, Dict[str, Any]]:
        """
        Get the latest data for multiple symbols.
        
        Args:
            symbols: List of symbols to get data for
            record_type: Type of record to get
            source: Data source to use (default: try all sources)
            
        Returns:
            Dictionary mapping symbols to their latest data
        """
        result = {}
        
        for symbol in symbols:
            data = RealtimeDataProvider.get_latest_data(symbol, record_type, source)
            if data:
                result[symbol] = data
                
        return result
    
    @staticmethod
    def get_multi_symbol_prices(symbols: List[str], 
                              source: DataSource = DataSource.ANY) -> Dict[str, float]:
        """
        Get the latest prices for multiple symbols.
        
        Args:
            symbols: List of symbols to get prices for
            source: Data source to use (default: try all sources)
            
        Returns:
            Dictionary mapping symbols to their latest prices
        """
        result = {}
        
        for symbol in symbols:
            price = RealtimeDataProvider.get_latest_price(symbol, source)
            if price is not None:
                result[symbol] = price
                
        return result
    
    @staticmethod
    def get_active_symbols(record_type: Optional[RecordType] = None) -> Set[str]:
        """
        Get all symbols that have real-time data available.
        
        Args:
            record_type: Optional record type to filter by
            
        Returns:
            Set of active symbols
        """
        # Get active symbols from WebSocket cache
        type_str = record_type.value if record_type else None
        ws_symbols = data_cache.get_all_active_symbols(type_str)
        
        # Get active symbols from other Redis keys
        pattern = "polygon:latest:*"
        keys = redis_client.keys(pattern)
        
        redis_symbols = set()
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 4:
                # Skip non-symbol keys
                if parts[2] in ['market', 'status', 'config']:
                    continue
                redis_symbols.add(parts[3])
        
        # Combine sets
        return ws_symbols.union(redis_symbols)
    
    @staticmethod
    def get_options_flow_data(symbol: str, limit: int = 30) -> pd.DataFrame:
        """
        Get options flow data from Unusual Whales for a symbol.
        
        Args:
            symbol: Symbol to get options flow data for
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with options flow data
        """
        symbol = symbol.upper()
        
        # Get from Redis
        key = f"unusual_whales:flow:{symbol}:recent"
        data = redis_client.lrange(key, 0, limit - 1)
        
        if not data:
            logger.warning(f"No options flow data found for {symbol}")
            return pd.DataFrame()
        
        logger.debug(f"Retrieved {len(data)} options flow records for {symbol} from Redis")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)
        
        return df
    
    @staticmethod
    def get_sentiment_data(symbol: str, sources: List[str] = None, 
                         days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment data for a symbol.
        
        Args:
            symbol: Symbol to get sentiment for
            sources: List of sentiment sources to include (None for all)
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment data
        """
        symbol = symbol.upper()
        
        # Default sources
        if sources is None:
            sources = ['news', 'social']
        
        # Get sentiment summary from database
        try:
            sentiment = timescale_client.get_sentiment_summary(symbol, days)
            return sentiment
        except Exception as e:
            logger.error(f"Error getting sentiment data for {symbol}: {str(e)}")
        
        # Fallback to Redis cache
        result = {
            'symbol': symbol,
            'period_days': days,
            'combined': {'avg_sentiment': 0, 'mention_count': 0},
            'sources': {}
        }
        
        # Check each source
        if 'news' in sources:
            news_key = f"sentiment:news:{symbol}:summary"
            news_data = redis_client.get(news_key)
            if news_data:
                result['sources']['news'] = news_data
        
        if 'social' in sources:
            social_key = f"sentiment:social:{symbol}:summary"
            social_data = redis_client.get(social_key)
            if social_data:
                result['sources']['social'] = social_data
        
        # If we have data from any source, calculate combined
        if result['sources']:
            total_sentiment = 0
            total_mentions = 0
            
            for source, data in result['sources'].items():
                total_sentiment += data.get('avg_sentiment', 0) * data.get('mention_count', 0)
                total_mentions += data.get('mention_count', 0)
            
            if total_mentions > 0:
                result['combined']['avg_sentiment'] = total_sentiment / total_mentions
                result['combined']['mention_count'] = total_mentions
        
        return result
    
    @staticmethod
    def get_combined_market_data(symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get combined market data for a symbol, merging WebSocket and database data.
        This is particularly useful for ML models that need the latest data plus history.
        
        Args:
            symbol: Symbol to get data for
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with combined market data
        """
        # Get real-time data from WebSocket
        ws_data = RealtimeDataProvider.get_recent_dataframe(symbol, RecordType.BAR, limit=limit//2)
        
        # Get historical data from database
        try:
            end_time = datetime.now() - timedelta(minutes=15)  # Avoid overlap with WebSocket data
            start_time = end_time - timedelta(days=30)  # Get enough history
            
            # Query the database
            db_data = timescale_client.query_market_data(symbol, start_time, end_time, limit=limit)
            
            # Merge data
            if not ws_data.empty and not db_data.empty:
                # Ensure consistent column names
                db_cols = set(db_data.columns)
                ws_cols = set(ws_data.columns)
                
                # Find common columns
                common_cols = list(db_cols.intersection(ws_cols))
                
                # Merge on common columns
                merged = data_cleaner.merge_websocket_and_rest_data(
                    ws_data[common_cols].to_dict('records'), 
                    db_data[common_cols], 
                    symbol
                )
                
                return merged.iloc[-limit:]
            elif not ws_data.empty:
                return ws_data.iloc[-limit:]
            elif not db_data.empty:
                return db_data.iloc[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting combined market data for {symbol}: {str(e)}")
            # Return whatever WebSocket data we have
            if not ws_data.empty:
                return ws_data
        
        logger.warning(f"No combined market data found for {symbol}")
        return pd.DataFrame()


# Module-level instance for convenience
_provider = RealtimeDataProvider()


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    import time
    
    parser = argparse.ArgumentParser(description='Realtime Data Provider')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to query')
    parser.add_argument('--type', type=str, default='bar', choices=['trade', 'quote', 'bar', 'aggregate'],
                      help='Data type to query')
    parser.add_argument('--limit', type=int, default=10, help='Number of records to return')
    parser.add_argument('--price', action='store_true', help='Get latest price')
    parser.add_argument('--ohlcv', action='store_true', help='Get OHLCV data')
    parser.add_argument('--flow', action='store_true', help='Get options flow data')
    parser.add_argument('--source', type=str, default='any', 
                      choices=['any', 'polygon', 'alpaca', 'yahoo', 'unusual_whales', 'reddit'],
                      help='Data source to use')
    
    args = parser.parse_args()
    
    # Convert args to enum values
    record_type_map = {
        'trade': RecordType.TRADE,
        'quote': RecordType.QUOTE,
        'bar': RecordType.BAR,
        'aggregate': RecordType.AGGREGATE
    }
    
    source_map = {
        'any': DataSource.ANY,
        'polygon': DataSource.POLYGON,
        'alpaca': DataSource.ALPACA,
        'yahoo': DataSource.YAHOO,
        'unusual_whales': DataSource.UNUSUAL_WHALES,
        'reddit': DataSource.REDDIT
    }
    
    record_type = record_type_map[args.type]
    source = source_map[args.source]
    
    # Get the requested data
    if args.price:
        price = RealtimeDataProvider.get_latest_price(args.symbol, source)
        print(f"Latest price for {args.symbol}: {price}")
    
    elif args.ohlcv:
        ohlcv = RealtimeDataProvider.get_ohlcv_dataframe(args.symbol, args.limit, source)
        print(f"OHLCV data for {args.symbol}:")
        print(ohlcv)
    
    elif args.flow:
        flow = RealtimeDataProvider.get_options_flow_data(args.symbol, args.limit)
        print(f"Options flow data for {args.symbol}:")
        print(flow)
    
    else:
        # Get recent data
        data = RealtimeDataProvider.get_recent_dataframe(args.symbol, record_type, args.limit, source)
        print(f"Recent {args.type} data for {args.symbol}:")
        print(data)
        
        # Get latest data
        latest = RealtimeDataProvider.get_latest_data(args.symbol, record_type, source)
        print(f"\nLatest {args.type} data for {args.symbol}:")
        print(json.dumps(latest, indent=2, default=str))
        
        # Get active symbols
        active = RealtimeDataProvider.get_active_symbols()
        print(f"\nActive symbols ({len(active)}):")
        print(", ".join(sorted(list(active))[:10]) + "..." if len(active) > 10 else "")
