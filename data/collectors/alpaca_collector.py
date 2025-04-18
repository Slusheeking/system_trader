#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Markets REST API Collector for historical market data.
This collector integrates with Alpaca's REST API to fetch historical
bars, trades, quotes, and other market data.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
import yaml
import time

from utils.logging import setup_logger
from config.collector_config import CollectorConfig
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType
from data.database.redis_client import get_redis_client
from data.database.timeseries_db import get_timescale_client
from data.processors.data_cache import get_data_cache
from data.collectors.api_key_manager import get_api_key_manager, APIKeyAuthenticationError, APIKeyRateLimitError, retry_on_rate_limit

# Setup logging
logger = setup_logger('alpaca_collector', category='data')

# Initialize shared clients
redis_client = get_redis_client()
db_client = get_timescale_client()
data_cache = get_data_cache()
api_key_manager = get_api_key_manager()


class AlpacaRESTClient:
    """Client for interacting with Alpaca REST API"""
    
    # API endpoints
    BASE_URL_PAPER = "https://paper-api.alpaca.markets"
    BASE_URL_LIVE = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"
    API_VERSION = "v2"
    
    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True):
        """Initialize the REST client with credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.BASE_URL_PAPER if paper_trading else self.BASE_URL_LIVE
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with authentication headers"""
        session = requests.Session()
        session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        })
        return session
    
    def _get_endpoint_url(self, endpoint: str, data_api: bool = False) -> str:
        """Construct the full URL for a given endpoint"""
        base = self.DATA_URL if data_api else self.base_url
        return f"{base}/{self.API_VERSION}/{endpoint.lstrip('/')}"
    
    def _handle_response(self, response: requests.Response) -> Union[dict, List[dict]]:
        """Handle API response and errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                # Rate limit exceeded
                retry_after = response.headers.get('Retry-After', '60')
                error_msg = f"Rate limit exceeded: {e}, retry after {retry_after}s"
                logger.warning(error_msg)
                raise APIKeyRateLimitError(error_msg, retry_after=int(retry_after))
            elif response.status_code in (401, 403):
                # Authentication or authorization error
                error_msg = f"Authentication error: {e}"
                logger.error(error_msg)
                raise APIKeyAuthenticationError(error_msg)
            else:
                error_msg = f"HTTP error: {e}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data}"
                    except:
                        error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            raise

    # Account endpoints
    @retry_on_rate_limit(max_retries=3)
    def get_account(self) -> dict:
        """Get account information"""
        url = self._get_endpoint_url('/account')
        response = self.session.get(url)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_positions(self) -> List[dict]:
        """Get all open positions"""
        url = self._get_endpoint_url('/positions')
        response = self.session.get(url)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_position(self, symbol: str) -> dict:
        """Get position for a specific symbol"""
        url = self._get_endpoint_url(f'/positions/{symbol}')
        response = self.session.get(url)
        return self._handle_response(response)
    
    # Order endpoints
    @retry_on_rate_limit(max_retries=3)
    def submit_order(self, 
                     symbol: str, 
                     qty: Optional[float] = None, 
                     notional: Optional[float] = None,
                     side: str = "buy", 
                     type: str = "market", 
                     time_in_force: str = "day",
                     limit_price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     client_order_id: Optional[str] = None) -> dict:
        """Submit a new order"""
        url = self._get_endpoint_url('/orders')
        
        # Basic order parameters
        order_data = {
            "symbol": symbol,
            "side": side,
            "type": type,
            "time_in_force": time_in_force
        }
        
        # Add either quantity or notional amount (dollar-based)
        if qty is not None:
            order_data["qty"] = str(qty)
        elif notional is not None:
            order_data["notional"] = str(notional)
        else:
            raise ValueError("Either qty or notional must be specified")
            
        # Add optional parameters if provided
        if limit_price is not None and type in ["limit", "stop_limit"]:
            order_data["limit_price"] = str(limit_price)
        if stop_price is not None and type in ["stop", "stop_limit"]:
            order_data["stop_price"] = str(stop_price)
        if client_order_id is not None:
            order_data["client_order_id"] = client_order_id
            
        response = self.session.post(url, json=order_data)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_orders(self, status: str = "open", limit: int = 100) -> List[dict]:
        """Get orders with the given status"""
        url = self._get_endpoint_url('/orders')
        params = {
            "status": status,
            "limit": limit
        }
        response = self.session.get(url, params=params)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_order(self, order_id: str) -> dict:
        """Get an order by ID"""
        url = self._get_endpoint_url(f'/orders/{order_id}')
        response = self.session.get(url)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def cancel_order(self, order_id: str) -> None:
        """Cancel an order by ID"""
        url = self._get_endpoint_url(f'/orders/{order_id}')
        response = self.session.delete(url)
        if response.status_code != 204:  # 204 No Content is success for DELETE
            return self._handle_response(response)
        return None
    
    # Market Data endpoints
    @retry_on_rate_limit(max_retries=3)
    def get_bars(self, 
                 symbol: str, 
                 timeframe: str = "1Day", 
                 start: Optional[str] = None, 
                 end: Optional[str] = None, 
                 limit: int = 100,
                 adjustment: str = 'raw') -> dict:
        """
        Get historical bars/candles
        timeframe options: 1Min, 5Min, 15Min, 1Hour, 1Day
        adjustment options: raw, split, dividend, all
        """
        url = self._get_endpoint_url(f'/stocks/{symbol}/bars', data_api=True)
        params = {
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": adjustment
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        response = self.session.get(url, params=params)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_trades(self, 
                  symbol: str, 
                  start: Optional[str] = None, 
                  end: Optional[str] = None, 
                  limit: int = 100) -> dict:
        """Get historical trades"""
        url = self._get_endpoint_url(f'/stocks/{symbol}/trades', data_api=True)
        params = {
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        response = self.session.get(url, params=params)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_quotes(self, 
                  symbol: str, 
                  start: Optional[str] = None, 
                  end: Optional[str] = None, 
                  limit: int = 100) -> dict:
        """Get historical quotes (bid/ask)"""
        url = self._get_endpoint_url(f'/stocks/{symbol}/quotes', data_api=True)
        params = {
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        response = self.session.get(url, params=params)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_assets(self, status: str = "active", asset_class: Optional[str] = None) -> List[dict]:
        """Get list of assets"""
        url = self._get_endpoint_url('/assets')
        params = {
            "status": status
        }
        if asset_class:
            params["asset_class"] = asset_class
        
        response = self.session.get(url, params=params)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_asset(self, symbol: str) -> dict:
        """Get information for a specific asset"""
        url = self._get_endpoint_url(f'/assets/{symbol}')
        response = self.session.get(url)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_clock(self) -> dict:
        """Get current market clock"""
        url = self._get_endpoint_url('/clock')
        response = self.session.get(url)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_calendar(self, start: Optional[str] = None, end: Optional[str] = None) -> List[dict]:
        """Get market calendar"""
        url = self._get_endpoint_url('/calendar')
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        response = self.session.get(url, params=params)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_snapshot(self, symbol: str) -> dict:
        """Get latest snapshot (bars, trades, quotes) for a symbol"""
        url = self._get_endpoint_url(f'/stocks/{symbol}/snapshot', data_api=True)
        response = self.session.get(url)
        return self._handle_response(response)
    
    @retry_on_rate_limit(max_retries=3)
    def get_multi_snapshot(self, symbols: List[str]) -> dict:
        """Get latest snapshot for multiple symbols"""
        symbols_str = ",".join(symbols)
        url = self._get_endpoint_url(f'/stocks/snapshots', data_api=True)
        params = {"symbols": symbols_str}
        response = self.session.get(url, params=params)
        return self._handle_response(response)


class AlpacaCollector(BaseCollector):
    """Collector for Alpaca Markets historical data, integrating with the system architecture."""
    
    def __init__(self, config: CollectorConfig = None):
        """
        Initialize Alpaca collector.
        
        Args:
            config: Optional CollectorConfig instance; loaded if None
        """
        if config is None:
            config = CollectorConfig.load('alpaca')
        super().__init__(config)
        
        # Extract configuration
        self.paper_trading = getattr(config, 'paper_trading', True)
        self.max_retries = getattr(config, 'max_retries', 3)
        self.retry_delay = getattr(config, 'retry_delay_seconds', 2)
        
        # Initialize REST client with API key manager
        try:
            credentials = api_key_manager.get_api_key('alpaca', validate=True)
            self.api_key = credentials['api_key']
            self.api_secret = credentials['api_secret']
            self.logger.info(f"Initialized Alpaca Collector with API key from manager: {self.api_key[:4]}...{self.api_key[-4:]}")
        except APIKeyAuthenticationError as e:
            self.logger.error(f"Failed to authenticate Alpaca API key: {str(e)}")
            # Fallback to config if available
            self.api_key = getattr(config, 'api_key', None)
            self.api_secret = getattr(config, 'api_secret', None)
            
            if not self.api_key or not self.api_secret:
                raise CollectorError("No valid API key available for Alpaca") from e
            
            self.logger.warning(f"Using fallback API key from config: {self.api_key[:4]}...{self.api_key[-4:]}")
        
        # Create the REST client
        self.client = AlpacaRESTClient(self.api_key, self.api_secret, self.paper_trading)
        
        # Type of data to fetch (bars, trades, quotes)
        self.data_type = getattr(config, 'data_type', 'bars')
        self.timeframe = getattr(config, 'timeframe', '1Min')
        self.adjustments = getattr(config, 'adjustments', 'raw')
        
        # Cache TTL (in seconds)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 3600)  # 1 hour by default
        
        # Batch processing settings
        self.batch_inserts = getattr(config, 'batch_inserts', True)
        self.batch_size = getattr(config, 'batch_size', 100)
        
        # Data cleaning
        self.clean_data = getattr(config, 'clean_data', True)
        self.validate_data = getattr(config, 'validate_data', True)
    
    @retry_on_rate_limit(max_retries=3)
    def _authenticate(self) -> None:
        """
        Authenticate with Alpaca API.
        Required by BaseCollector.
        """
        try:
            # Test authentication by getting account info
            account = self.client.get_account()
            logger.info(f"Authenticated with Alpaca API as account: {account.get('id')}")
        except APIKeyAuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            # Notify the API key manager of the authentication failure
            api_key_manager.handle_authentication_failure('alpaca', 
                {'api_key': self.api_key, 'api_secret': self.api_secret})
            raise CollectorError(f"Failed to authenticate with Alpaca API: {str(e)}")
        except APIKeyRateLimitError as e:
            logger.warning(f"Rate limit hit during authentication: {str(e)}")
            # Notify the API key manager of the rate limit
            api_key_manager.handle_rate_limit('alpaca', retry_after=e.retry_after)
            raise
        except Exception as e:
            logger.error(f"Authentication failed with unexpected error: {str(e)}")
            raise CollectorError(f"Failed to authenticate with Alpaca API: {str(e)}")
    
    @retry_on_rate_limit(max_retries=3)
    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Request a page of data from Alpaca.
        
        Args:
            page_token: Optional token for pagination
            
        Returns:
            Raw data from API
        """
        try:
            # Parse the page token for request parameters
            if page_token:
                params = json.loads(page_token)
                symbol = params.get('symbol')
                start = params.get('start')
                end = params.get('end')
                limit = params.get('limit', 1000)
            else:
                # No page token, use default parameters
                raise CollectorError("No page token provided, cannot determine what data to request")
            
            # Make appropriate API call based on data type
            if self.data_type == 'bars':
                data = self.client.get_bars(symbol, timeframe=self.timeframe, start=start, end=end, limit=limit, adjustment=self.adjustments)
                next_page_token = None
                
                # Check for pagination in the response
                if 'next_page_token' in data:
                    next_page_token = json.dumps({
                        'symbol': symbol,
                        'start': data.get('next_page_token'),
                        'end': end,
                        'limit': limit
                    })
                
                return data, next_page_token
                
            elif self.data_type == 'trades':
                data = self.client.get_trades(symbol, start=start, end=end, limit=limit)
                next_page_token = None
                
                # Check for pagination in the response
                if 'next_page_token' in data:
                    next_page_token = json.dumps({
                        'symbol': symbol,
                        'start': data.get('next_page_token'),
                        'end': end,
                        'limit': limit
                    })
                
                return data, next_page_token
                
            elif self.data_type == 'quotes':
                data = self.client.get_quotes(symbol, start=start, end=end, limit=limit)
                next_page_token = None
                
                # Check for pagination in the response
                if 'next_page_token' in data:
                    next_page_token = json.dumps({
                        'symbol': symbol,
                        'start': data.get('next_page_token'),
                        'end': end,
                        'limit': limit
                    })
                
                return data, next_page_token
            
            else:
                raise CollectorError(f"Unsupported data type: {self.data_type}")
                
        except APIKeyAuthenticationError as e:
            logger.error(f"Authentication error requesting data: {str(e)}")
            api_key_manager.handle_authentication_failure('alpaca', 
                {'api_key': self.api_key, 'api_secret': self.api_secret})
            raise CollectorError(f"Authentication failed: {str(e)}")
        except APIKeyRateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            api_key_manager.handle_rate_limit('alpaca', retry_after=e.retry_after)
            raise
        except Exception as e:
            logger.error(f"Error requesting data from Alpaca API: {str(e)}")
            raise CollectorError(f"Failed to request data from Alpaca API: {str(e)}")
    
    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw Alpaca data into standard records.
        
        Args:
            raw: Raw data from Alpaca API
            
        Returns:
            Tuple of (list of StandardRecord, next page token)
        """
        try:
            data, next_page_token = raw
            symbol = data.get('symbol', 'UNKNOWN')
            records = []
            
            # Parse based on data type
            if self.data_type == 'bars':
                bars = data.get('bars', [])
                for bar in bars:
                    timestamp_str = bar.get('t')
                    if timestamp_str:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Use current time if not available
                        timestamp = datetime.now(timezone.utc)
                    
                    # Create standard record
                    record = StandardRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        record_type=RecordType.BAR,
                        source='alpaca',
                        open=Decimal(str(bar.get('o', 0))),
                        high=Decimal(str(bar.get('h', 0))),
                        low=Decimal(str(bar.get('l', 0))),
                        close=Decimal(str(bar.get('c', 0))),
                        volume=int(bar.get('v', 0)),
                        vwap=Decimal(str(bar.get('vw', 0))) if bar.get('vw') else None,
                        extended_data={
                            'trade_count': bar.get('n'),
                            'timeframe': self.timeframe,
                            'adjustment': self.adjustments,
                            'raw': bar
                        }
                    )
                    records.append(record)
            
            elif self.data_type == 'trades':
                trades = data.get('trades', [])
                for trade in trades:
                    timestamp_str = trade.get('t')
                    if timestamp_str:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Use current time if not available
                        timestamp = datetime.now(timezone.utc)
                    
                    # Create standard record
                    record = StandardRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        record_type=RecordType.TRADE,
                        source='alpaca',
                        price=Decimal(str(trade.get('p', 0))),
                        volume=int(trade.get('s', 0)),
                        exchange=trade.get('x', ''),
                        trade_id=trade.get('i', ''),
                        tape=trade.get('z', ''),
                        conditions=trade.get('c', []),
                        extended_data={
                            'raw': trade
                        }
                    )
                    records.append(record)
            
            elif self.data_type == 'quotes':
                quotes = data.get('quotes', [])
                for quote in quotes:
                    timestamp_str = quote.get('t')
                    if timestamp_str:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Use current time if not available
                        timestamp = datetime.now(timezone.utc)
                    
                    # Create standard record
                    record = StandardRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        record_type=RecordType.QUOTE,
                        source='alpaca',
                        bid_price=Decimal(str(quote.get('bp', 0))),
                        bid_size=int(quote.get('bs', 0)),
                        ask_price=Decimal(str(quote.get('ap', 0))),
                        ask_size=int(quote.get('as', 0)),
                        exchange=quote.get('x', ''),
                        conditions=quote.get('c', []),
                        extended_data={
                            'raw': quote
                        }
                    )
                    records.append(record)
            
            return records, next_page_token
        
        except Exception as e:
            logger.error(f"Error parsing Alpaca data: {str(e)}")
            raise CollectorError(f"Failed to parse Alpaca data: {str(e)}")
    
    def _cache_records(self, records: List[StandardRecord]) -> None:
        """
        Cache records in Redis using data_cache.
        
        Args:
            records: List of StandardRecord to cache
        """
        try:
            for record in records:
                # Determine data type from record type
                if record.record_type == RecordType.BAR:
                    data_type = 'bars'
                elif record.record_type == RecordType.TRADE:
                    data_type = 'trades'
                elif record.record_type == RecordType.QUOTE:
                    data_type = 'quotes'
                else:
                    data_type = 'other'
                
                # Convert record to dictionary and store in data cache
                record_dict = record.model_dump()
                data_cache.add_historical_data(record.symbol, data_type, record_dict)
                
                # For backwards compatibility, also store in traditional Redis format
                # Create cache key based on record type and symbol
                record_type = record.record_type.value
                symbol = record.symbol
                cache_key = f"alpaca:{record_type}:{symbol}:{record.timestamp.isoformat()}"
                
                # Store in Redis
                redis_client.set(cache_key, record_dict, self.cache_ttl)
                
                # Also store in a list of recent records for this symbol and type
                list_key = f"alpaca:{record_type}:{symbol}:recent"
                redis_client.lpush(list_key, record_dict)
                redis_client.ltrim(list_key, 0, 99)  # Keep only the 100 most recent records
                redis_client.expire(list_key, self.cache_ttl)
                
                # Store latest record for ML model access
                latest_key = f"alpaca:latest:{record_type}:{symbol}"
                redis_client.set(latest_key, record_dict, self.cache_ttl * 2)
                
        except Exception as e:
            logger.error(f"Error caching records in Redis: {str(e)}")
    
    def _store_records_in_db(self, records: List[StandardRecord]) -> None:
        """
        Store records in the database.
        
        Args:
            records: List of StandardRecord to store
        """
        try:
            # Group records by type for bulk inserts
            market_data = []
            trade_data = []
            analytics_data = []
            
            for record in records:
                record_dict = record.model_dump()
                
                if record.record_type == RecordType.BAR or record.record_type == RecordType.AGGREGATE:
                    # Store as market data
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'symbol': record_dict.get('symbol'),
                        'open': record_dict.get('open'),
                        'high': record_dict.get('high'),
                        'low': record_dict.get('low'),
                        'close': record_dict.get('close'),
                        'volume': record_dict.get('volume'),
                        'vwap': record_dict.get('vwap'),
                        'num_trades': record_dict.get('extended_data', {}).get('trade_count'),
                        'source': record_dict.get('source'),
                        'data_type': 'bars',
                        'record_type': record.record_type.value,
                        'metadata': record_dict.get('extended_data')
                    }
                    market_data.append(db_record)
                
                elif record.record_type == RecordType.TRADE:
                    # Store as trade data
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'order_id': record_dict.get('trade_id', ''),
                        'trade_id': record_dict.get('trade_id'),
                        'symbol': record_dict.get('symbol'),
                        'side': 'unknown',  # Alpaca doesn't provide trade side in market data
                        'quantity': record_dict.get('volume', 0),
                        'price': record_dict.get('price', 0),
                        'exchange': record_dict.get('exchange'),
                        'source': record_dict.get('source'),
                        'data_type': 'trades',
                        'record_type': 'trade',
                        'metadata': record_dict.get('extended_data')
                    }
                    trade_data.append(db_record)
                
                elif record.record_type == RecordType.QUOTE:
                    # Store as analytics data
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'metric_name': 'quote',
                        'metric_value': (float(record_dict.get('bid_price', 0)) + float(record_dict.get('ask_price', 0))) / 2,
                        'symbol': record_dict.get('symbol'),
                        'dimension': 'price',
                        'metadata': {
                            'bid_price': float(record_dict.get('bid_price', 0)),
                            'bid_size': record_dict.get('bid_size', 0),
                            'ask_price': float(record_dict.get('ask_price', 0)),
                            'ask_size': record_dict.get('ask_size', 0),
                            'exchange': record_dict.get('exchange'),
                            'source': record_dict.get('source'),
                            'data_type': 'quotes',
                            'record_type': 'quote',
                            **record_dict.get('extended_data', {})
                        }
                    }
                    analytics_data.append(db_record)
            
            # Bulk insert to database
            if market_data:
                db_client.insert_market_data(market_data)
                logger.debug(f"Inserted {len(market_data)} market data records")
                
            if trade_data:
                db_client.insert_trade_data(trade_data)
                logger.debug(f"Inserted {len(trade_data)} trade data records")
                
            if analytics_data:
                db_client.insert_analytics_data(analytics_data)
                logger.debug(f"Inserted {len(analytics_data)} analytics data records")
        
        except Exception as e:
            logger.error(f"Error storing records in database: {str(e)}")
    
    @retry_on_rate_limit(max_retries=3)
    def collect_historical(self, symbol: str, start: datetime, end: datetime, data_type: Optional[str] = None) -> List[StandardRecord]:
        """
        Collect historical data for a symbol in a date range.
        
        Args:
            symbol: Symbol to collect data for
            start: Start datetime
            end: End datetime
            data_type: Optional data type override ('bars', 'trades', 'quotes')
            
        Returns:
            List of StandardRecord objects
        """
        # Save original data type to restore later
        original_data_type = self.data_type
        
        # Override data type if specified
        if data_type:
            self.data_type = data_type
        
        try:
            # Format dates for Alpaca API (ISO format)
            start_str = start.isoformat()
            end_str = end.isoformat()
            
            # Create initial page token
            page_token = json.dumps({
                'symbol': symbol,
                'start': start_str,
                'end': end_str,
                'limit': 1000
            })
            
            # Collect all records using the standard collect method
            records = self.collect(start, end, page_token=page_token)
            
            # Cache records in Redis
            self._cache_records(records)
            
            # Store records in database
            self._store_records_in_db(records)
            
            return records
            
        finally:
            # Restore original data type
            self.data_type = original_data_type
    
    @retry_on_rate_limit(max_retries=3)
    def get_snapshots(self, symbols: List[str]) -> Dict[str, StandardRecord]:
        """
        Get latest snapshots for multiple symbols.
        
        Args:
            symbols: List of symbols to get snapshots for
            
        Returns:
            Dictionary of symbol to StandardRecord
        """
        try:
            # Request snapshots from Alpaca
            snapshots = self.client.get_multi_snapshot(symbols)
            records = {}
            
            # Process each snapshot
            for symbol, snapshot in snapshots.items():
                # Extract the latest bar
                bar = snapshot.get('minuteBar') or snapshot.get('dailyBar')
                if bar:
                    timestamp_str = bar.get('t')
                    if timestamp_str:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Use current time if not available
                        timestamp = datetime.now(timezone.utc)
                    
                    # Create standard record for bar
                    bar_record = StandardRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        record_type=RecordType.BAR,
                        source='alpaca',
                        open=Decimal(str(bar.get('o', 0))),
                        high=Decimal(str(bar.get('h', 0))),
                        low=Decimal(str(bar.get('l', 0))),
                        close=Decimal(str(bar.get('c', 0))),
                        volume=int(bar.get('v', 0)),
                        vwap=Decimal(str(bar.get('vw', 0))) if bar.get('vw') else None,
                        extended_data={
                            'snapshot': True,
                            'trade_count': bar.get('n'),
                            'raw': bar
                        }
                    )
                    records[symbol] = bar_record
                    
                    # Cache the record in data_cache
                    data_cache.add_historical_data(symbol, 'bars', bar_record.model_dump())
                    
                    # Cache the record for backward compatibility
                    cache_key = f"alpaca:snapshot:bar:{symbol}"
                    redis_client.set(cache_key, bar_record.model_dump(), self.cache_ttl)
                
                # Extract the latest trade
                trade = snapshot.get('latestTrade')
                if trade:
                    timestamp_str = trade.get('t')
                    if timestamp_str:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Use current time if not available
                        timestamp = datetime.now(timezone.utc)
                    
                    # Create standard record for trade
                    trade_record = StandardRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        record_type=RecordType.TRADE,
                        source='alpaca',
                        price=Decimal(str(trade.get('p', 0))),
                        volume=int(trade.get('s', 0)),
                        exchange=trade.get('x', ''),
                        trade_id=trade.get('i', ''),
                        tape=trade.get('z', ''),
                        conditions=trade.get('c', []),
                        extended_data={
                            'snapshot': True,
                            'raw': trade
                        }
                    )
                    
                    # Cache in data_cache
                    data_cache.add_historical_data(symbol, 'trades', trade_record.model_dump())
                    
                    # Cache for backward compatibility
                    cache_key = f"alpaca:snapshot:trade:{symbol}"
                    redis_client.set(cache_key, trade_record.model_dump(), self.cache_ttl)
                
                # Extract the latest quote
                quote = snapshot.get('latestQuote')
                if quote:
                    timestamp_str = quote.get('t')
                    if timestamp_str:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Use current time if not available
                        timestamp = datetime.now(timezone.utc)
                    
                    # Create standard record for quote
                    quote_record = StandardRecord(
                        symbol=symbol,
                        timestamp=timestamp,
                        record_type=RecordType.QUOTE,
                        source='alpaca',
                        bid_price=Decimal(str(quote.get('bp', 0))),
                        bid_size=int(quote.get('bs', 0)),
                        ask_price=Decimal(str(quote.get('ap', 0))),
                        ask_size=int(quote.get('as', 0)),
                        exchange=quote.get('x', ''),
                        conditions=quote.get('c', []),
                        extended_data={
                            'snapshot': True,
                            'raw': quote
                        }
                    )
                    
                    # Cache in data_cache
                    data_cache.add_historical_data(symbol, 'quotes', quote_record.model_dump())
                    
                    # Cache for backward compatibility
                    cache_key = f"alpaca:snapshot:quote:{symbol}"
                    redis_client.set(cache_key, quote_record.model_dump(), self.cache_ttl)
            
            return records
            
        except APIKeyAuthenticationError as e:
            logger.error(f"Authentication error getting snapshots: {str(e)}")
            api_key_manager.handle_authentication_failure('alpaca', 
                {'api_key': self.api_key, 'api_secret': self.api_secret})
            return {}
        except APIKeyRateLimitError as e:
            logger.warning(f"Rate limit exceeded getting snapshots: {str(e)}")
            api_key_manager.handle_rate_limit('alpaca', retry_after=e.retry_after)
            raise
        except Exception as e:
            logger.error(f"Error getting snapshots: {str(e)}")
            return {}
    
    @retry_on_rate_limit(max_retries=3)
    def check_market_status(self) -> Dict[str, Any]:
        """
        Check if the market is open.
        
        Returns:
            Dictionary with market status
        """
        try:
            clock = self.client.get_clock()
            
            # Cache the result using data_cache
            data_cache.set_market_status('alpaca', clock)
            
            # Cache for backward compatibility
            redis_client.set("alpaca:market:clock", clock, 300)  # Cache for 5 minutes
            
            return clock
        except APIKeyAuthenticationError as e:
            logger.error(f"Authentication error checking market status: {str(e)}")
            api_key_manager.handle_authentication_failure('alpaca', 
                {'api_key': self.api_key, 'api_secret': self.api_secret})
            # Try to get from cache if available
            cached = data_cache.get_market_status('alpaca')
            if cached:
                return cached
            return {"is_open": False, "error": str(e)}
        except APIKeyRateLimitError as e:
            logger.warning(f"Rate limit exceeded checking market status: {str(e)}")
            api_key_manager.handle_rate_limit('alpaca', retry_after=e.retry_after)
            # Try to get from cache if available
            cached = data_cache.get_market_status('alpaca')
            if cached:
                return cached
            return {"is_open": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            # Try to get from cache if available
            cached = data_cache.get_market_status('alpaca')
            if cached:
                return cached
            cached = redis_client.get("alpaca:market:clock")
            if cached:
                return cached
            return {"is_open": False, "error": str(e)}


# Factory function for creating an instance
def create_alpaca_collector(config: Optional[CollectorConfig] = None) -> AlpacaCollector:
    """Create an instance of the Alpaca collector."""
    return AlpacaCollector(config)


# For backwards compatibility
if __name__ == "__main__":
    # Configure logging
    logger.setLevel(logging.INFO)
    
    # Create collector
    collector = AlpacaCollector()
    
    # Test authentication
    collector._authenticate()
    
    # Check if market is open
    clock = collector.check_market_status()
    print(f"Market is {'open' if clock.get('is_open') else 'closed'}")
    
    # Get recent bars for Apple
    start = datetime.now(timezone.utc) - timedelta(days=5)
    end = datetime.now(timezone.utc)
    bars = collector.collect_historical("AAPL", start, end, "bars")
    print(f"Collected {len(bars)} AAPL bars")
    
    # Get snapshots for a few symbols
    snapshots = collector.get_snapshots(["AAPL", "MSFT", "AMZN"])
    for symbol, record in snapshots.items():
        print(f"{symbol}: {record.close}")
