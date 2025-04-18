#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unusual Whales Data Collector Module
----------------------------------
This module handles data collection from the Unusual Whales API,
providing options flow data for enhanced market regime detection.
"""

import requests
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.logging import setup_logger
from config.collector_config import CollectorConfig
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType
from data.database.redis_client import get_redis_client
from data.database.timeseries_db import get_timescale_client

# Setup logging
logger = setup_logger('unusual_whales_collector', category='data')

# Initialize shared database clients
redis_client = get_redis_client()
db_client = get_timescale_client()


class UnusualWhalesCollector(BaseCollector):
    """
    Unusual Whales API data collector for options flow data.

    This collector fetches options flow data from the Unusual Whales REST API
    and returns it as a list of StandardRecord instances.
    """

    def __init__(self, config: CollectorConfig = None):
        """
        Initialize the Unusual Whales collector.

        Args:
            config: Optional CollectorConfig instance; loaded if None
        """
        if config is None:
            config = CollectorConfig.load('unusual_whales')
        super().__init__(config)
        
        # Base URL and headers from config
        self.base_url: str = getattr(config, 'endpoint', 'https://api.unusualwhales.com').rstrip('/')
        self.headers: Dict[str, str] = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Configure data retention
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 3600)  # 1 hour by default
        
        # These will be set in collect()
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        
        logger.info(f"Initialized Unusual Whales Collector using API endpoint: {self.base_url}")

    def collect(self, start: datetime, end: datetime, page_token: Optional[str] = None) -> List[StandardRecord]:
        """
        Collect options flow data between start and end datetimes.

        Args:
            start: Start datetime
            end: End datetime
            page_token: Optional starting page token

        Returns:
            List of StandardRecord objects
        """
        # Store for use in page requests
        self._start = start
        self._end = end
        return super().collect(start, end, page_token)

    def _authenticate(self) -> None:
        """
        No-op for Unusual Whales since we use static API key in headers.
        """
        # Authentication handled via header; nothing to do here.
        return

    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Request a single page of options flow data.

        Args:
            page_token: Optional pagination token from previous page

        Returns:
            Parsed JSON response
        """
        if self._start is None or self._end is None:
            raise CollectorError("Start and end must be set before requesting data")

        params: Dict[str, Any] = {
            'start_date': self._start.strftime('%Y-%m-%d'),
            'end_date': self._end.strftime('%Y-%m-%d')
        }
        if page_token:
            params['page_token'] = page_token

        try:
            url = f"{self.base_url}/options/flow"
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.config.timeout
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                reset = response.headers.get('X-RateLimit-Reset')
                retry_after = response.headers.get('Retry-After')
                error_msg = f"Rate limit exceeded. Reset at {reset}"
                if retry_after:
                    error_msg += f", retry after {retry_after} seconds"
                raise CollectorError(error_msg)
                
            # Handle other errors    
            if not response.ok:
                raise CollectorError(f"HTTP {response.status_code}: {response.text}")
                
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise CollectorError(f"Request error: {str(e)}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {str(e)}")
            raise CollectorError(f"Failed to decode JSON response: {str(e)}") from e

    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw JSON page into StandardRecord list and return next page token.

        Args:
            raw: Raw JSON response from the API

        Returns:
            Tuple of (records list, next_page_token or None)
        """
        try:
            data = raw.get('data', [])
            next_token = raw.get('next_page', None)

            records: List[StandardRecord] = []
            for item in data:
                # Parse timestamp (ISO 8601 string)
                ts_str = item.get('timestamp')
                try:
                    # Handle possible Z suffix
                    if ts_str:
                        if ts_str.endswith('Z'):
                            ts = datetime.fromisoformat(ts_str.rstrip('Z')).replace(tzinfo=timezone.utc)
                        else:
                            ts = datetime.fromisoformat(ts_str)
                    else:
                        ts = datetime.now(timezone.utc)
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp '{ts_str}': {str(e)}")
                    ts = datetime.now(timezone.utc)

                # Extract and convert premium to Decimal
                premium = item.get('premium')
                if premium is not None:
                    try:
                        premium = Decimal(str(premium))
                    except (ValueError, TypeError):
                        premium = None

                # Extract and convert volume to int
                volume = item.get('volume')
                if volume is not None:
                    try:
                        volume = int(volume)
                    except (ValueError, TypeError):
                        volume = None
                
                # Extract contract type (call/put)
                contract_type = item.get('contract_type', '')
                # Extract strike price
                strike_price = item.get('strike')
                if strike_price is not None:
                    try:
                        strike_price = Decimal(str(strike_price))
                    except (ValueError, TypeError):
                        strike_price = None

                # Build standard record
                record = StandardRecord(
                    symbol=item.get('symbol', ''),
                    timestamp=ts,
                    record_type=RecordType.TRADE,
                    source='unusual_whales',
                    price=premium,
                    volume=volume,
                    extended_data={
                        **item,
                        'contract_type': contract_type,
                        'strike_price': strike_price,
                        'options_flow': True
                    }
                )
                records.append(record)
                
                # Cache the record in Redis
                self._cache_record(record)

            # Store records in the database
            self._store_records_in_db(records)

            return records, next_token
            
        except Exception as e:
            logger.error(f"Error parsing Unusual Whales data: {str(e)}")
            raise CollectorError(f"Failed to parse Unusual Whales data: {str(e)}")

    def _cache_record(self, record: StandardRecord) -> None:
        """
        Cache a record in Redis.
        
        Args:
            record: StandardRecord to cache
        """
        try:
            # Create cache key based on record symbol and timestamp
            symbol = record.symbol
            record_hash = hash(f"{record.timestamp.isoformat()}:{record.extended_data.get('strike_price')}:{record.extended_data.get('contract_type')}")
            cache_key = f"unusual_whales:flow:{symbol}:{record_hash}"
            
            # Convert record to dictionary
            record_dict = record.model_dump()
            
            # Store in Redis
            redis_client.set(cache_key, record_dict, self.cache_ttl)
            
            # Also store in a list of recent records for this symbol
            recent_key = f"unusual_whales:flow:{symbol}:recent"
            redis_client.lpush(recent_key, record_dict)
            redis_client.ltrim(recent_key, 0, 99)  # Keep only the 100 most recent records
            redis_client.expire(recent_key, self.cache_ttl)
            
            # Store in a global recent options flow list
            global_recent_key = "unusual_whales:flow:recent"
            redis_client.lpush(global_recent_key, record_dict)
            redis_client.ltrim(global_recent_key, 0, 499)  # Keep the 500 most recent records
            redis_client.expire(global_recent_key, self.cache_ttl)
            
        except Exception as e:
            logger.error(f"Error caching record in Redis: {str(e)}")

    def _store_records_in_db(self, records: List[StandardRecord]) -> None:
        """
        Store records in the database.
        
        Args:
            records: List of StandardRecord to store
        """
        if not records:
            return
            
        try:
            # Convert records to database format for analytics_data table
            analytics_records = []
            
            for record in records:
                record_dict = record.model_dump()
                extended_data = record_dict.get('extended_data', {})
                
                # Determine if this is a call or put
                contract_type = extended_data.get('contract_type', '')
                is_call = contract_type.lower() == 'call'
                is_put = contract_type.lower() == 'put'
                
                # Calculate sentiment value (premium * volume * directional factor)
                sentiment_factor = 1 if is_call else -1 if is_put else 0
                premium = float(record_dict.get('price', 0) or 0)
                volume = record_dict.get('volume', 0) or 0
                sentiment_value = premium * volume * sentiment_factor
                
                # Store as analytics data
                db_record = {
                    'time': record_dict.get('timestamp'),
                    'metric_name': 'options_flow',
                    'metric_value': premium,  # Store premium as the main metric
                    'symbol': record_dict.get('symbol'),
                    'dimension': contract_type,  # Use contract type as dimension
                    'metadata': {
                        'volume': volume,
                        'strike_price': extended_data.get('strike_price'),
                        'contract_type': contract_type,
                        'expiration': extended_data.get('expiration'),
                        'sentiment_value': sentiment_value,
                        'raw': extended_data
                    }
                }
                analytics_records.append(db_record)
            
            # Bulk insert to database
            if analytics_records:
                db_client.insert_analytics_data(analytics_records)
                logger.debug(f"Inserted {len(analytics_records)} options flow records into analytics_data")
                
        except Exception as e:
            logger.error(f"Error storing records in database: {str(e)}")

    def get_recent_options_flow(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent options flow data from cache.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of recent options flow data
        """
        try:
            if symbol:
                # Get symbol-specific recent data
                key = f"unusual_whales:flow:{symbol}:recent"
            else:
                # Get global recent data
                key = "unusual_whales:flow:recent"
                
            data = redis_client.lrange(key, 0, limit - 1)
            return data or []
        except Exception as e:
            logger.error(f"Error getting recent options flow data: {str(e)}")
            return []
    
    def get_options_sentiment(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate options sentiment for the given symbols.
        
        Args:
            symbols: List of symbols to get sentiment for
            
        Returns:
            Dictionary of symbol to sentiment metrics
        """
        result = {}
        
        for symbol in symbols:
            try:
                # Get recent options flow for this symbol
                flow_data = self.get_recent_options_flow(symbol)
                
                if not flow_data:
                    result[symbol] = {
                        'call_volume': 0,
                        'put_volume': 0,
                        'call_premium': 0.0,
                        'put_premium': 0.0,
                        'pc_ratio': 1.0,  # Neutral by default
                        'sentiment_score': 0.0,  # Neutral by default
                        'flow_count': 0
                    }
                    continue
                
                # Calculate aggregates
                call_volume = 0
                put_volume = 0
                call_premium = 0.0
                put_premium = 0.0
                
                for item in flow_data:
                    contract_type = item.get('extended_data', {}).get('contract_type', '').lower()
                    volume = item.get('volume', 0) or 0
                    premium = float(item.get('price', 0) or 0)
                    
                    if contract_type == 'call':
                        call_volume += volume
                        call_premium += premium * volume
                    elif contract_type == 'put':
                        put_volume += volume
                        put_premium += premium * volume
                
                # Calculate put/call ratio (by volume)
                pc_ratio = put_volume / call_volume if call_volume > 0 else float('inf')
                
                # Calculate sentiment score
                # -1.0 = very bearish, 0 = neutral, 1.0 = very bullish
                total_volume = call_volume + put_volume
                if total_volume > 0:
                    sentiment_score = (call_volume - put_volume) / total_volume
                else:
                    sentiment_score = 0.0
                
                result[symbol] = {
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'call_premium': call_premium,
                    'put_premium': put_premium,
                    'pc_ratio': pc_ratio,
                    'sentiment_score': sentiment_score,
                    'flow_count': len(flow_data)
                }
                
            except Exception as e:
                logger.error(f"Error calculating options sentiment for {symbol}: {str(e)}")
                result[symbol] = {
                    'error': str(e),
                    'sentiment_score': 0.0  # Neutral on error
                }
        
        return result


# Factory function for creating an instance
def create_unusual_whales_collector(config: Optional[CollectorConfig] = None) -> UnusualWhalesCollector:
    """Create an instance of the Unusual Whales collector."""
    return UnusualWhalesCollector(config)


if __name__ == "__main__":
    import logging
    from datetime import timedelta
    
    # Set logging level for testing
    logger.setLevel(logging.INFO)
    
    # Create collector
    collector = UnusualWhalesCollector()
    
    # Define date range for collection
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)  # Last 24 hours
    
    # Collect data
    records = collector.collect(start, end)
    print(f"Collected {len(records)} options flow records")
    
    # Get sentiment for some symbols
    sentiment = collector.get_options_sentiment(['AAPL', 'TSLA', 'MSFT', 'AMZN', 'NVDA'])
    for symbol, data in sentiment.items():
        print(f"{symbol}: sentiment_score={data['sentiment_score']:.2f}, pc_ratio={data['pc_ratio']:.2f}")
