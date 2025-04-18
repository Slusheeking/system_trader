#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yahoo Finance Data Collector Module
----------------------------------
This module handles data collection from Yahoo Finance API,
including historical data via standardized BaseCollector interface.
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

import requests
import pandas as pd
import yfinance as yf

from utils.logging import setup_logger
from config.collector_config import CollectorConfig
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType
from data.database.timeseries_db import get_timescale_client
from data.database.redis_client import get_redis_client
from data.processors.data_cache import get_data_cache
from data.collectors.api_key_manager import get_api_key_manager, APIKeyAuthenticationError, APIKeyRateLimitError, retry_on_rate_limit

# Setup logging
logger = setup_logger('yahoo_collector', category='data')

# Initialize shared clients
db_client = get_timescale_client()
redis_client = get_redis_client()
data_cache = get_data_cache()
api_key_manager = get_api_key_manager()


class YahooCollector(BaseCollector):
    """Yahoo Finance data collector as a BaseCollector subclass."""

    def __init__(
        self,
        config: CollectorConfig = None,
        symbol: str = None,
        interval: str = "1d",
        include_prepost: bool = False,
    ):
        """
        Initialize the YahooCollector.

        Args:
            config: Optional CollectorConfig with endpoint and settings; loaded if None
            symbol: Ticker symbol to collect; if None, uses first symbol from config
            interval: Data interval ('1d', '1wk', '1mo', '1m', etc.)
            include_prepost: Whether to include pre- and post-market data
        """
        if config is None:
            config = CollectorConfig.load('yahoo')
        super().__init__(config)
        
        # Set symbol, falling back to config if not explicitly provided
        self.symbol = symbol or getattr(config, 'symbol', None)
        if not self.symbol and hasattr(config, 'symbols') and config.symbols:
            self.symbol = config.symbols[0]
        if not self.symbol:
            self.symbol = 'SPY'  # Default to SPY as fallback
            
        self.interval = interval or getattr(config, 'interval', '1d')
        self.include_prepost = include_prepost or getattr(config, 'include_prepost', False)
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        
        # Set endpoint, falling back to a default if not in config
        self.endpoint = getattr(config, 'endpoint', 'https://query1.finance.yahoo.com')
        
        # Cache TTL (in seconds)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 3600)  # 1 hour default
        
        # RapidAPI settings (for premium features)
        self.use_rapidapi = getattr(config, 'use_rapidapi', False)
        
        # Setup API key for RapidAPI if enabled (Yahoo Finance on RapidAPI requires authentication)
        if self.use_rapidapi:
            try:
                credentials = api_key_manager.get_api_key('yahoo', validate=False)
                self.api_key = credentials.get('api_key')
                self.api_host = credentials.get('api_host', 'apidojo-yahoo-finance-v1.p.rapidapi.com')
                logger.info(f"Using RapidAPI for Yahoo Finance with API key: {self.api_key[:4]}...{self.api_key[-4:] if self.api_key else ''}")
            except Exception as e:
                logger.warning(f"Failed to get Yahoo Finance API key, falling back to public API: {e}")
                self.use_rapidapi = False
                self.api_key = None
                self.api_host = None
        else:
            self.api_key = None
            self.api_host = None
            
        # Data cleaning and validation settings
        self.clean_data = getattr(config, 'clean_data', True)
        self.validate_data = getattr(config, 'validate_data', True)
        
        # Retry settings
        self.max_retries = getattr(config, 'max_retries', 3)
        self.retry_delay = getattr(config, 'retry_delay_seconds', 2)
        
        logger.info(f"Initialized Yahoo collector for symbol: {self.symbol}, interval: {self.interval}, RapidAPI: {self.use_rapidapi}")

    def _authenticate(self) -> None:
        """No authentication required for Yahoo Finance public API, but we validate RapidAPI keys if used."""
        if not self.use_rapidapi:
            # No authentication needed for public API
            return
            
        try:
            # Validate RapidAPI key with a simple request
            url = f"https://{self.api_host}/market/v2/get-summary"
            headers = {
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': self.api_host
            }
            params = {'region': 'US'}
            
            response = requests.get(url, headers=headers, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            # If successful, we're authenticated
            logger.info("Successfully authenticated with RapidAPI Yahoo Finance")
            
            # Register the API key as valid
            api_key_manager.register_api_key_success('yahoo', {'api_key': self.api_key, 'api_host': self.api_host})
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (401, 403):
                logger.error(f"RapidAPI authentication failed: {e}")
                # Notify the key manager of the failure
                api_key_manager.handle_authentication_failure('yahoo', {'api_key': self.api_key, 'api_host': self.api_host})
                # Fall back to public API
                self.use_rapidapi = False
                logger.warning("Falling back to public Yahoo Finance API")
            elif e.response.status_code == 429:
                logger.warning(f"RapidAPI rate limit reached: {e}")
                retry_after = e.response.headers.get('Retry-After', '60')
                api_key_manager.handle_rate_limit('yahoo', retry_after=int(retry_after))
                # Fall back to public API
                self.use_rapidapi = False
                logger.warning("Falling back to public Yahoo Finance API")
            else:
                logger.error(f"HTTP error during RapidAPI authentication: {e}")
                # Fall back to public API
                self.use_rapidapi = False
                logger.warning("Falling back to public Yahoo Finance API")
        except Exception as e:
            logger.error(f"Error during RapidAPI authentication: {e}")
            # Fall back to public API
            self.use_rapidapi = False
            logger.warning("Falling back to public Yahoo Finance API")

    @retry_on_rate_limit(max_retries=3)
    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Request a single page of raw data from Yahoo Finance.

        Args:
            page_token: Not used for Yahoo (no pagination).

        Returns:
            Raw JSON response from Yahoo Finance chart API.
        """
        if self._start is None or self._end is None:
            raise CollectorError("Start and end times must be set before requesting data")

        # Check data cache first
        cache_key = f"yahoo:{self.symbol}:{self.interval}:{int(self._start.timestamp())}:{int(self._end.timestamp())}"
        cached_data = data_cache.get_cached_data(cache_key)
        if cached_data:
            logger.debug(f"Using cached Yahoo data for {self.symbol}")
            return cached_data

        # Create request parameters
        params = {
            "period1": int(self._start.timestamp()),
            "period2": int(self._end.timestamp()),
            "interval": self.interval,
            "includePrePost": str(self.include_prepost).lower(),
            "events": "div,split,capitalGains"  # Include important events
        }
        
        try:
            # Determine which API to use (RapidAPI or public)
            if self.use_rapidapi:
                url = f"https://{self.api_host}/market/get-charts"
                headers = {
                    'X-RapidAPI-Key': self.api_key,
                    'X-RapidAPI-Host': self.api_host
                }
                params.update({
                    'symbol': self.symbol,
                    'region': 'US'
                })
                
                logger.debug(f"Requesting data from Yahoo Finance via RapidAPI for {self.symbol}")
                response = requests.get(url, headers=headers, params=params, timeout=self.config.timeout)
            else:
                # Use public API
                url = f"{self.endpoint}/v8/finance/chart/{self.symbol}"
                
                logger.debug(f"Requesting data from Yahoo Finance public API for {self.symbol}")
                response = requests.get(url, params=params, timeout=self.config.timeout)
            
            # Handle errors
            response.raise_for_status()
            data = response.json()
            
            # Cache the result with data_cache
            data_cache.set_cached_data(cache_key, data, self.cache_ttl)
            
            # Also cache in Redis for backward compatibility
            redis_key = f"yahoo:data:{self.symbol}:{self.interval}:{params['period1']}:{params['period2']}"
            redis_client.set(redis_key, data, self.cache_ttl)
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if self.use_rapidapi and e.response.status_code == 429:
                # Rate limit exceeded for RapidAPI
                logger.warning(f"RapidAPI rate limit exceeded: {e}")
                retry_after = e.response.headers.get('Retry-After', '60')
                api_key_manager.handle_rate_limit('yahoo', retry_after=int(retry_after))
                raise APIKeyRateLimitError(f"Yahoo Finance RapidAPI rate limit exceeded: {e}", 
                                          retry_after=int(retry_after))
            elif self.use_rapidapi and e.response.status_code in (401, 403):
                # Authentication failed for RapidAPI
                logger.error(f"RapidAPI authentication failed: {e}")
                api_key_manager.handle_authentication_failure('yahoo', 
                                                           {'api_key': self.api_key, 'api_host': self.api_host})
                # Fall back to public API
                self.use_rapidapi = False
                logger.warning("Falling back to public Yahoo Finance API - retrying request")
                return self._request_page(page_token)
            else:
                logger.error(f"Request failed for Yahoo Finance API: {e}")
                raise CollectorError(f"HTTP request failed for Yahoo Finance: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for Yahoo Finance API: {e}")
            raise CollectorError(f"HTTP request failed for Yahoo Finance: {e}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Yahoo Finance API response: {e}")
            raise CollectorError(f"Invalid JSON response from Yahoo Finance: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in Yahoo Finance data request: {e}")
            raise CollectorError(f"Error in Yahoo Finance data request: {e}")

    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw JSON data into StandardRecord instances.

        Args:
            raw: Raw JSON data from Yahoo API.

        Returns:
            A tuple of (records list, next_page_token). Next page token is always None.
        """
        try:
            chart = raw.get("chart", {})
            error = chart.get("error")
            if error:
                raise CollectorError(f"Yahoo API error: {error}")

            results = chart.get("result")
            if not results or len(results) == 0:
                return [], None

            data = results[0]
            timestamps = data.get("timestamp", [])
            indicators = data.get("indicators", {}).get("quote", [])
            if not indicators or len(indicators) == 0:
                return [], None

            # Extract base quotes
            quote = indicators[0]
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])
            
            # Check for adjusted close
            adjclose = data.get("indicators", {}).get("adjclose", [])
            adj_closes = adjclose[0].get("adjclose", []) if adjclose else []
            
            # Extract events if available
            events = data.get("events", {})
            dividends = events.get("dividends", {})
            splits = events.get("splits", {})

            records: List[StandardRecord] = []
            for idx, ts in enumerate(timestamps):
                # Convert timestamp to datetime with timezone
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                
                # Some intervals may have None entries
                open_p = opens[idx] if idx < len(opens) else None
                high_p = highs[idx] if idx < len(highs) else None
                low_p = lows[idx] if idx < len(lows) else None
                close_p = closes[idx] if idx < len(closes) else None
                adj_close = adj_closes[idx] if idx < len(adj_closes) else None
                vol = volumes[idx] if idx < len(volumes) else None
                
                # Skip records with all None values
                if all(x is None for x in [open_p, high_p, low_p, close_p, vol]):
                    continue
                
                # Create extended data with events
                extended_data = {
                    'interval': self.interval,
                    'include_prepost': self.include_prepost,
                    'adj_close': adj_close
                }
                
                # Add dividend or split info if available for this timestamp
                str_timestamp = str(ts)
                if str_timestamp in dividends:
                    div_info = dividends[str_timestamp]
                    extended_data['dividend'] = div_info.get('amount')
                
                if str_timestamp in splits:
                    split_info = splits[str_timestamp]
                    extended_data['split'] = f"{split_info.get('numerator', 1)}/{split_info.get('denominator', 1)}"
                
                # Convert to Decimal for precision
                try:
                    open_val = Decimal(str(open_p)) if open_p is not None else None
                    high_val = Decimal(str(high_p)) if high_p is not None else None
                    low_val = Decimal(str(low_p)) if low_p is not None else None
                    close_val = Decimal(str(close_p)) if close_p is not None else None
                except (ValueError, TypeError):
                    # If Decimal conversion fails, use original values
                    open_val = open_p
                    high_val = high_p
                    low_val = low_p
                    close_val = close_p

                record = StandardRecord(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    record_type=RecordType.BAR,
                    source="yahoo",
                    open=open_val,
                    high=high_val,
                    low=low_val,
                    close=close_val,
                    volume=vol,
                    extended_data=extended_data
                )
                
                # Clean data if enabled
                if self.clean_data and hasattr(data_cache, 'get_data_cleaner'):
                    cleaner = data_cache.get_data_cleaner()
                    record_dict = record.model_dump()
                    clean_data = cleaner.clean_historical_data(record_dict, 'bars', self.symbol)
                    
                    # Update record with cleaned data
                    for key, value in clean_data.items():
                        if hasattr(record, key):
                            setattr(record, key, value)
                
                # Validate data if enabled
                if self.validate_data and hasattr(data_cache, 'get_data_cleaner'):
                    cleaner = data_cache.get_data_cleaner()
                    if not cleaner.validate_historical_data(record.model_dump(), 'bars'):
                        logger.warning(f"Skipping invalid data point for {self.symbol} at {timestamp}")
                        continue
                
                records.append(record)

            return records, None
        except Exception as e:
            logger.error(f"Error parsing Yahoo data: {str(e)}")
            raise CollectorError(f"Error parsing Yahoo data: {str(e)}")

    @retry_on_rate_limit(max_retries=3)
    def collect(self, start: datetime, end: datetime, page_token: Optional[str] = None) -> List[StandardRecord]:
        """
        Collect data between start and end datetime using BaseCollector flow.

        Args:
            start: Start datetime
            end: End datetime
            page_token: Not used for Yahoo (no pagination)

        Returns:
            List of StandardRecord instances
        """
        # Ensure timezone information
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
            
        self._start = start
        self._end = end
        records = super().collect(start, end, page_token)
        
        # Store collected data in TimescaleDB
        self._store_records_in_db(records)
        
        # Store in data cache
        self._cache_records(records)
        
        return records

    def _store_records_in_db(self, records: List[StandardRecord]) -> None:
        """
        Store records in the database.
        
        Args:
            records: List of StandardRecord to store
        """
        if not records:
            return
            
        try:
            # Convert StandardRecord objects to dictionaries for database storage
            db_records = []
            for record in records:
                if record.record_type == RecordType.BAR:
                    extended_data = record.extended_data if record.extended_data else {}
                    
                    db_record = {
                        'time': record.timestamp,
                        'symbol': record.symbol,
                        'open': record.open,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close,
                        'volume': record.volume,
                        'source': 'yahoo',
                        'data_type': 'bars',
                        'record_type': 'bar',
                        'metadata': {
                            'interval': self.interval,
                            'include_prepost': self.include_prepost,
                            'adj_close': extended_data.get('adj_close'),
                            'dividend': extended_data.get('dividend'),
                            'split': extended_data.get('split')
                        }
                    }
                    db_records.append(db_record)
            
            # Write to database
            if db_records:
                inserted_count = db_client.insert_market_data(db_records)
                logger.info(f"Stored {inserted_count} Yahoo Finance records in TimescaleDB for {self.symbol}")
        except Exception as e:
            logger.error(f"Failed to store Yahoo Finance data in database: {str(e)}")
    
    def _cache_records(self, records: List[StandardRecord]) -> None:
        """
        Cache records in data_cache and Redis.
        
        Args:
            records: List of StandardRecord to cache
        """
        if not records:
            return
            
        try:
            # Create a pandas DataFrame for easier manipulation
            df_data = []
            for record in records:
                row = {
                    'timestamp': record.timestamp,
                    'open': float(record.open) if record.open is not None else None,
                    'high': float(record.high) if record.high is not None else None,
                    'low': float(record.low) if record.low is not None else None,
                    'close': float(record.close) if record.close is not None else None,
                    'volume': int(record.volume) if record.volume is not None else None,
                    'adj_close': float(record.extended_data.get('adj_close')) if record.extended_data and record.extended_data.get('adj_close') is not None else None
                }
                df_data.append(row)
            
            # Skip if no data
            if not df_data:
                return
                
            # Create DataFrame
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Store in data_cache
            for record in records:
                data_cache.add_historical_data(self.symbol, 'bars', record.model_dump())
            
            # Cache the dataframe in Redis for backward compatibility
            cache_key = f"yahoo:dataframe:{self.symbol}:{self.interval}"
            redis_client.cache_dataframe(cache_key, df, self.cache_ttl)
            
            # Cache the latest data
            if not df.empty:
                latest = df.iloc[-1].to_dict()
                latest_key = f"yahoo:latest:{self.symbol}"
                latest_data = {
                    'symbol': self.symbol,
                    'timestamp': df.index[-1].isoformat(),
                    'price': latest.get('close'),
                    'change': latest.get('close') - latest.get('open') if latest.get('close') is not None and latest.get('open') is not None else None,
                    'change_pct': ((latest.get('close') / latest.get('open')) - 1) * 100 if latest.get('close') is not None and latest.get('open') is not None and latest.get('open') != 0 else None,
                    'volume': latest.get('volume'),
                    'data': latest
                }
                
                # Cache in data_cache
                data_cache.set_market_price(self.symbol, latest_data)
                
                # Cache in Redis for backward compatibility
                redis_client.set(latest_key, latest_data, self.cache_ttl * 2)
            
            logger.debug(f"Cached {len(df)} records for {self.symbol} in data cache and Redis")
        except Exception as e:
            logger.error(f"Failed to cache Yahoo Finance data: {str(e)}")

    @retry_on_rate_limit(max_retries=3)
    def get_historical_data(self, start: datetime, end: datetime, as_df: bool = False) -> Union[List[StandardRecord], pd.DataFrame]:
        """
        Get historical data for the configured symbol.
        
        Args:
            start: Start datetime
            end: End datetime
            as_df: Return as pandas DataFrame if True, else as StandardRecord list
            
        Returns:
            List of StandardRecord or pandas DataFrame
        """
        records = self.collect(start, end)
        
        if as_df and records:
            # Convert to DataFrame
            df_data = []
            for record in records:
                row = {
                    'timestamp': record.timestamp,
                    'open': float(record.open) if record.open is not None else None,
                    'high': float(record.high) if record.high is not None else None,
                    'low': float(record.low) if record.low is not None else None,
                    'close': float(record.close) if record.close is not None else None,
                    'volume': int(record.volume) if record.volume is not None else None,
                    'adj_close': float(record.extended_data.get('adj_close')) if record.extended_data and record.extended_data.get('adj_close') is not None else None
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            return df
        
        return records
    
    @retry_on_rate_limit(max_retries=3)
    def get_latest_price(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get the latest price data for the configured symbol.
        
        Args:
            use_cache: Use cached data if available
            
        Returns:
            Dictionary with latest price information
        """
        if use_cache:
            # Try to get from data_cache
            latest_data = data_cache.get_market_price(self.symbol)
            if latest_data:
                return latest_data
                
            # Try Redis cache for backward compatibility
            latest_key = f"yahoo:latest:{self.symbol}"
            cached_data = redis_client.get(latest_key)
            if cached_data:
                return cached_data
        
        # Get from API if not in cache or cache not requested
        end = datetime.now(timezone.utc)
        if self.interval == '1d':
            # For daily data, get last 5 days
            start = end - timedelta(days=5)
        elif self.interval == '1h' or self.interval == '60m':
            # For hourly data, get last 24 hours
            start = end - timedelta(hours=24)
        else:
            # For other intervals, get last day
            start = end - timedelta(days=1)
        
        records = self.collect(start, end)
        
        if not records:
            return {
                'symbol': self.symbol,
                'error': 'No data available',
                'timestamp': end.isoformat()
            }
        
        # Get the most recent record
        latest_record = max(records, key=lambda r: r.timestamp)
        
        latest_data = {
            'symbol': self.symbol,
            'timestamp': latest_record.timestamp.isoformat(),
            'price': float(latest_record.close) if latest_record.close is not None else None,
            'open': float(latest_record.open) if latest_record.open is not None else None,
            'high': float(latest_record.high) if latest_record.high is not None else None,
            'low': float(latest_record.low) if latest_record.low is not None else None,
            'volume': int(latest_record.volume) if latest_record.volume is not None else None,
            'interval': self.interval
        }
        
        # Cache the result
        data_cache.set_market_price(self.symbol, latest_data)
        
        return latest_data
    
    @retry_on_rate_limit(max_retries=3)
    def get_intraday_data(self, days: int = 1) -> pd.DataFrame:
        """
        Get intraday price data.
        
        Args:
            days: Number of days of intraday data to retrieve
            
        Returns:
            DataFrame with intraday data
        """
        # Save current interval
        original_interval = self.interval
        
        try:
            # Switch to 1-minute data
            self.interval = '1m'
            
            # Get data for specified days
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            
            # Get data as DataFrame
            df = self.get_historical_data(start, end, as_df=True)
            return df
        finally:
            # Restore original interval
            self.interval = original_interval
    
    @retry_on_rate_limit(max_retries=3)
    def get_daily_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get daily price data.
        
        Args:
            days: Number of days of daily data to retrieve
            
        Returns:
            DataFrame with daily data
        """
        # Save current interval
        original_interval = self.interval
        
        try:
            # Switch to daily data
            self.interval = '1d'
            
            # Get data for specified days
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            
            # Get data as DataFrame
            df = self.get_historical_data(start, end, as_df=True)
            return df
        finally:
            # Restore original interval
            self.interval = original_interval
    
    @retry_on_rate_limit(max_retries=3)
    def fetch_stock_info(self) -> Dict[str, Any]:
        """
        Fetch company information for the configured symbol.
        
        Returns:
            Dictionary with company information
        """
        try:
            # Check data_cache first
            cache_key = f"yahoo:info:{self.symbol}"
            cached_info = data_cache.get_cached_data(cache_key)
            if cached_info:
                return cached_info
                
            # Try Redis cache for backward compatibility
            redis_cached_info = redis_client.get(cache_key)
            if redis_cached_info:
                return redis_cached_info
            
            # Use RapidAPI if enabled, otherwise use yfinance
            if self.use_rapidapi:
                try:
                    url = f"https://{self.api_host}/stock/v2/get-profile"
                    headers = {
                        'X-RapidAPI-Key': self.api_key,
                        'X-RapidAPI-Host': self.api_host
                    }
                    params = {
                        'symbol': self.symbol,
                        'region': 'US'
                    }
                    
                    response = requests.get(url, headers=headers, params=params, timeout=self.config.timeout)
                    response.raise_for_status()
                    info = response.json()
                except Exception as e:
                    logger.warning(f"RapidAPI request for stock info failed: {e}. Falling back to yfinance.")
                    # Fall back to yfinance
                    ticker = yf.Ticker(self.symbol)
                    info = ticker.info
            else:
                # Use yfinance directly
                ticker = yf.Ticker(self.symbol)
                info = ticker.info
            
            # Cache the result in data_cache
            data_cache.set_cached_data(cache_key, info, self.cache_ttl * 24)  # Cache for a longer period
            
            # Cache in Redis for backward compatibility
            redis_client.set(cache_key, info, self.cache_ttl * 24)
            
            return info
        except Exception as e:
            logger.error(f"Failed to fetch stock info for {self.symbol}: {str(e)}")
            
            # Try to get from cache if available
            cache_key = f"yahoo:info:{self.symbol}"
            cached_info = data_cache.get_cached_data(cache_key)
            if cached_info:
                return cached_info
                
            redis_cached_info = redis_client.get(cache_key)
            if redis_cached_info:
                return redis_cached_info
                
            return {'symbol': self.symbol, 'error': str(e)}


# Factory function for creating an instance
def create_yahoo_collector(config: Optional[CollectorConfig] = None, symbol: Optional[str] = None) -> YahooCollector:
    """Create an instance of the Yahoo collector."""
    return YahooCollector(config, symbol)


if __name__ == "__main__":
    import logging
    
    # Set logging level for testing
    logger.setLevel(logging.INFO)
    
    # Create collector
    collector = YahooCollector(symbol="AAPL")
    
    # Get historical data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)
    
    data = collector.get_historical_data(start_date, end_date, as_df=True)
    print(f"Collected {len(data)} records for AAPL")
    
    # Print latest price
    latest = collector.get_latest_price()
    print(f"Latest AAPL price: ${latest['price']}")
    
    # Get stock info
    info = collector.fetch_stock_info()
    print(f"Company name: {info.get('shortName', 'Unknown')}")
    print(f"Industry: {info.get('industry', 'Unknown')}")
    print(f"Market cap: ${info.get('marketCap', 0):,}")
