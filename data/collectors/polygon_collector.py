#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polygon.io Historical Data Collector using BaseCollector.
Supports market data, news, and sentiment analysis.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from polygon import RESTClient

from utils.logging import setup_logger
from config.collector_config import CollectorConfig
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType
from data.database.timeseries_db import TimeseriesDBClient, get_timescale_client
from data.database.redis_client import get_redis_client
from nlp.sentiment_analyzer import SentimentAnalyzer
from data.collectors.api_key_manager import get_api_key_manager, APIKeyAuthenticationError, APIKeyRateLimitError, retry_on_rate_limit

# Setup logging
logger = setup_logger('polygon_collector', category='data')

# Initialize shared clients
db_client = get_timescale_client()
redis_client = get_redis_client()
api_key_manager = get_api_key_manager()


class PolygonCollector(BaseCollector):
    """
    Collector for Polygon.io data including market data, news, and sentiment.
    
    This collector supports:
    - Aggregated bar data (OHLCV)
    - News articles with sentiment analysis
    - Company details and financials
    """

    def __init__(self, config: CollectorConfig = None):
        """
        Initialize Polygon collector.
        
        Args:
            config: Optional CollectorConfig instance; loaded if None
        """
        if config is None:
            config = CollectorConfig.load('polygon')
        super().__init__(config)
        
        # Default settings for market data
        self.multiplier = 1
        self.timespan = 'day'
        self.adjusted = True
        self.limit: int = getattr(config, 'limit', 50000)
        self.symbol = ""  # Initialize symbol with empty string
        
        # Settings for news collection
        self.news_limit = getattr(config, 'news_limit', 1000)
        self.include_content = getattr(config, 'include_content', True)
        self.news_sort = getattr(config, 'news_sort', 'published_utc')
        self.news_order = getattr(config, 'news_order', 'desc')
        
        # Collection mode (market_data, news, or both)
        self.collection_mode = getattr(config, 'collection_mode', 'market_data')
        
        # Store base URL for reference
        if hasattr(config, 'base_url'):
            self.base_url = str(config.base_url).rstrip('/')
        else:
            self.base_url = "https://api.polygon.io/v2"  # Default base URL
            
        # Retry settings
        self.max_retries = getattr(config, 'max_retries', 3)
        self.retry_delay = getattr(config, 'retry_delay_seconds', 2)
        
        # Initialize REST client with API key from API key manager
        try:
            credentials = api_key_manager.get_api_key('polygon', validate=True)
            api_key = credentials['api_key']
            self.logger.info(f"Initializing Polygon client with API key: {api_key[:4]}...{api_key[-4:]}")
            self.client = RESTClient(api_key=api_key)
        except APIKeyAuthenticationError as e:
            self.logger.error(f"Failed to authenticate Polygon API key: {str(e)}")
            # Fallback to config if available
            if hasattr(config, 'api_key') and config.api_key:
                api_key = config.api_key
                self.logger.warning(f"Using fallback API key from config: {api_key[:4]}...{api_key[-4:]}")
                self.client = RESTClient(api_key=api_key)
            else:
                raise CollectorError("No valid API key available for Polygon") from e
            
        # Initialize sentiment analyzer
        use_finbert = getattr(config, 'use_finbert', False)
        model_path = getattr(config, 'finbert_path', None)
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert, model_path)
        
        # Cache TTL (in seconds)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 3600)  # 1 hour

    def _authenticate(self) -> None:
        """
        Polygon RESTClient uses API key; no additional auth step required.
        """
        # No-op for RESTClient; API key provided in client initialization
        pass

    def set_symbol(self, symbol: str) -> None:
        """
        Set the symbol for data collection.
        
        Args:
            symbol: Ticker symbol to collect data for
        """
        self.symbol = symbol.upper()
        
    @retry_on_rate_limit(max_retries=3)
    def collect(self, start: datetime, end: datetime) -> List[StandardRecord]:
        """
        Collect data for the current symbol and date range.
        
        This method overrides the BaseCollector's collect method but maintains
        the same signature to ensure proper inheritance.
        
        Args:
            start: Start date for data collection
            end: End date for data collection
            
        Returns:
            List of StandardRecord objects
        
        Raises:
            CollectorError: If symbol is not set before calling collect
        """
        # Validate that symbol is set
        if not self.symbol:
            raise CollectorError("Symbol must be set before calling collect. Use set_symbol() method.")
            
        # Store formatted date strings for API requests
        self._start_str = start.strftime('%Y-%m-%d')
        self._end_str = end.strftime('%Y-%m-%d')
        
        # Collect data based on mode
        if self.collection_mode == 'market_data':
            return self._collect_market_data(start, end)
        elif self.collection_mode == 'news':
            return self._collect_news(start, end)
        elif self.collection_mode == 'both':
            market_records = self._collect_market_data(start, end)
            news_records = self._collect_news(start, end)
            return market_records + news_records
        else:
            raise CollectorError(f"Invalid collection mode: {self.collection_mode}")
    
    def _collect_market_data(self, start: datetime, end: datetime) -> List[StandardRecord]:
        """
        Collect market data for the current symbol and date range.
        
        Args:
            start: Start date for data collection
            end: End date for data collection
            
        Returns:
            List of StandardRecord objects
        """
        # Delegate paging, retries, and error handling to BaseCollector
        records = super().collect(start=start, end=end)
        
        # Persist collected data to TimescaleDB
        if records:
            try:
                # Convert StandardRecord objects to dictionaries for database storage
                db_records = []
                for record in records:
                    record_dict = record.model_dump()
                    # Convert to the format expected by insert_market_data
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'symbol': record_dict.get('symbol'),
                        'open': record_dict.get('open'),
                        'high': record_dict.get('high'),
                        'low': record_dict.get('low'),
                        'close': record_dict.get('close'),
                        'volume': record_dict.get('volume'),
                        'vwap': record_dict.get('vwap'),
                        'num_trades': record_dict.get('extended_data', {}).get('transactions'),
                        'source': record_dict.get('source'),
                        'data_type': 'price',
                        'record_type': 'aggregate',
                        'metadata': record_dict.get('extended_data')
                    }
                    db_records.append(db_record)
                
                count = db_client.insert_market_data(db_records)
                self.logger.info(f"Successfully persisted {count} {self.symbol} market data records to TimescaleDB")
                
                # Cache the most recent record for each symbol
                if records:
                    latest_record = max(records, key=lambda r: r.timestamp)
                    latest_key = f"polygon:latest:aggregate:{self.symbol}"
                    redis_client.set(latest_key, latest_record.model_dump(), self.cache_ttl)
                
            except Exception as e:
                self.logger.error(f"Failed to persist {self.symbol} market data to TimescaleDB: {str(e)}")
        
        return records
    
    def _collect_news(self, start: datetime, end: datetime) -> List[StandardRecord]:
        """
        Collect news data for the current symbol and date range.
        
        Args:
            start: Start date for data collection
            end: End date for data collection
            
        Returns:
            List of StandardRecord objects
        """
        self.logger.info(f"Collecting news for {self.symbol} from {start} to {end}")
        
        # Set collection mode to news for the parent collect method
        original_mode = self.collection_mode
        self.collection_mode = 'news'
        
        try:
            # Delegate paging, retries, and error handling to BaseCollector
            records = super().collect(start=start, end=end)
            
            # Persist collected news data to TimescaleDB
            if records:
                self._store_news_in_db(records)
            
            return records
        finally:
            # Restore original collection mode
            self.collection_mode = original_mode
    
    @retry_on_rate_limit(max_retries=3)
    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Request a single page of data from Polygon.io.
        
        Args:
            page_token: Optional pagination token
            
        Returns:
            Raw response from Polygon API
        """
        try:
            if self.collection_mode == 'market_data':
                # Request market data
                return self.client.get_aggs(
                    ticker=self.symbol,
                    multiplier=self.multiplier,
                    timespan=self.timespan,
                    from_=self._start_str,
                    to=self._end_str,
                    limit=self.limit,
                    adjusted=self.adjusted
                )
            elif self.collection_mode == 'news':
                # Request news data
                # Parse page token for pagination if available
                if page_token:
                    # Use page token as offset
                    offset = int(page_token)
                else:
                    offset = 0
                
                # Convert dates to ISO format for news API
                start_iso = datetime.fromisoformat(self._start_str).isoformat()
                end_iso = datetime.fromisoformat(self._end_str).isoformat()
                
                # Request news data
                return self.client.get_ticker_news(
                    ticker=self.symbol,
                    published_utc_gte=start_iso,
                    published_utc_lte=end_iso,
                    limit=self.news_limit,
                    order=self.news_order,
                    sort=self.news_sort,
                    offset=offset
                )
            else:
                raise CollectorError(f"Invalid collection mode: {self.collection_mode}")
        except Exception as e:
            # Check for rate limit errors
            if any(err_msg in str(e).lower() for err_msg in ['rate limit', '429', 'too many requests']):
                self.logger.warning(f"Rate limit exceeded for Polygon API: {str(e)}")
                api_key_manager.handle_rate_limit('polygon', retry_after=60)  # Default 1 minute retry
                raise APIKeyRateLimitError(f"Polygon API rate limit exceeded: {str(e)}")
            raise CollectorError(f"Error requesting page {page_token}: {e}") from e

    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw Polygon response into StandardRecord instances.
        
        Args:
            raw: Raw response from Polygon API
            
        Returns:
            Tuple of (records list, next_page_token)
        """
        self.logger.debug(f"Parsing response: {raw}")
        
        if self.collection_mode == 'market_data':
            return self._parse_market_data(raw)
        elif self.collection_mode == 'news':
            return self._parse_news(raw)
        else:
            raise CollectorError(f"Invalid collection mode: {self.collection_mode}")
    
    def _parse_market_data(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw market data response into StandardRecord instances.
        
        Args:
            raw: Raw response from Polygon API
            
        Returns:
            Tuple of (records list, next_page_token)
        """
        # Handle different response formats
        if hasattr(raw, 'results'):
            # Response is an object with results attribute
            data = raw.results
        elif isinstance(raw, dict) and 'results' in raw:
            # Response is a dictionary with results key
            data = raw['results']
        else:
            # Assume raw is the data itself
            data = raw
            
        # Ensure data is a list
        if not isinstance(data, list):
            self.logger.warning(f"Expected list of results, got {type(data)}")
            data = []
            
        # Extract next page token if available
        if hasattr(raw, 'next_page_token'):
            next_page = raw.next_page_token
        elif isinstance(raw, dict) and 'next_page_token' in raw:
            next_page = raw['next_page_token']
        elif isinstance(raw, dict) and 'next_page' in raw:
            next_page = raw['next_page']
        else:
            next_page = None
            
        records: List[StandardRecord] = []

        for item in data:
            try:
                # Check if item is an Agg object or a dictionary
                if hasattr(item, 'timestamp'):
                    # It's an Agg object
                    timestamp_ms = item.timestamp
                    open_val = item.open
                    high_val = item.high
                    low_val = item.low
                    close_val = item.close
                    volume_val = item.volume
                    vwap_val = item.vwap
                    
                    # Create extended data from other attributes
                    extended_data = {}
                    if hasattr(item, 'transactions'):
                        extended_data['transactions'] = item.transactions
                    if hasattr(item, 'otc'):
                        extended_data['otc'] = item.otc
                elif isinstance(item, dict):
                    # It's a dictionary
                    timestamp_ms = item.get('t') or item.get('timestamp')
                    open_val = item.get('o') or item.get('open')
                    high_val = item.get('h') or item.get('high')
                    low_val = item.get('l') or item.get('low')
                    close_val = item.get('c') or item.get('close')
                    volume_val = item.get('v') or item.get('volume')
                    vwap_val = item.get('vw') or item.get('vwap')
                    
                    # Create extended data from other keys
                    extended_data = {k: v for k, v in item.items()
                                    if k not in {'o', 'h', 'l', 'c', 'v', 'vw', 't',
                                                'open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp'}}
                else:
                    # Skip items that are neither objects with attributes nor dictionaries
                    self.logger.warning(f"Skipping item with unexpected type: {type(item)}")
                    continue
                
                # Check if timestamp_ms is valid before conversion
                if timestamp_ms is None:
                    self.logger.warning(f"Skipping item with missing timestamp: {item}")
                    continue
                    
                # Convert timestamp from milliseconds to datetime
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                
                # Create StandardRecord
                record = StandardRecord(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    record_type=RecordType.AGGREGATE,
                    source='polygon',
                    open=Decimal(str(open_val or 0)),
                    high=Decimal(str(high_val or 0)),
                    low=Decimal(str(low_val or 0)),
                    close=Decimal(str(close_val or 0)),
                    volume=volume_val or 0,
                    vwap=Decimal(str(vwap_val or 0)),
                    extended_data=extended_data
                )
                records.append(record)
                self.logger.debug(f"Parsed market data record: {record}")
            except Exception as e:
                self.logger.error(f"Error parsing market data item {item}: {str(e)}")
                
        self.logger.info(f"Parsed {len(records)} market data records")
        return records, next_page
    
    def _parse_news(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw news response into StandardRecord instances.
        
        Args:
            raw: Raw response from Polygon API
            
        Returns:
            Tuple of (records list, next_page_token)
        """
        # Handle different response formats
        if hasattr(raw, 'results'):
            # Response is an object with results attribute
            data = raw.results
        elif isinstance(raw, dict) and 'results' in raw:
            # Response is a dictionary with results key
            data = raw['results']
        else:
            # Assume raw is the data itself
            data = raw
            
        # Ensure data is a list
        if not isinstance(data, list):
            self.logger.warning(f"Expected list of news results, got {type(data)}")
            data = []
            
        # Extract pagination info
        count = getattr(raw, 'count', 0) if hasattr(raw, 'count') else raw.get('count', 0) if isinstance(raw, dict) else 0
        next_url = getattr(raw, 'next_url', None) if hasattr(raw, 'next_url') else raw.get('next_url', None) if isinstance(raw, dict) else None
        
        # Extract offset from next_url if available
        next_page = None
        if next_url and 'offset=' in next_url:
            try:
                offset_part = next_url.split('offset=')[1]
                next_page = offset_part.split('&')[0]
            except Exception:
                next_page = None
            
        records: List[StandardRecord] = []

        for item in data:
            try:
                # Extract news data
                if hasattr(item, 'id'):
                    # It's a News object
                    news_id = item.id
                    title = item.title
                    author = item.author
                    published_utc = item.published_utc
                    article_url = item.article_url
                    tickers = item.tickers if hasattr(item, 'tickers') else []
                    keywords = item.keywords if hasattr(item, 'keywords') else []
                    description = item.description if hasattr(item, 'description') else None
                    content = item.content if hasattr(item, 'content') and self.include_content else None
                    source = item.publisher.name if hasattr(item, 'publisher') and hasattr(item.publisher, 'name') else None
                    
                    # Create extended data
                    extended_data = {
                        'news_id': news_id,
                        'title': title,
                        'author': author,
                        'url': article_url,
                        'source': source,
                        'keywords': keywords,
                        'description': description,
                        'content': content,
                        'tickers': tickers
                    }
                elif isinstance(item, dict):
                    # It's a dictionary
                    news_id = item.get('id')
                    title = item.get('title')
                    author = item.get('author')
                    published_utc = item.get('published_utc')
                    article_url = item.get('article_url')
                    tickers = item.get('tickers', [])
                    keywords = item.get('keywords', [])
                    description = item.get('description')
                    content = item.get('content') if self.include_content else None
                    source = item.get('publisher', {}).get('name') if isinstance(item.get('publisher'), dict) else None
                    
                    # Create extended data
                    extended_data = {
                        'news_id': news_id,
                        'title': title,
                        'author': author,
                        'url': article_url,
                        'source': source,
                        'keywords': keywords,
                        'description': description,
                        'content': content,
                        'tickers': tickers
                    }
                else:
                    # Skip items that are neither objects with attributes nor dictionaries
                    self.logger.warning(f"Skipping news item with unexpected type: {type(item)}")
                    continue
                
                # Skip if no tickers or symbol not in tickers
                if not tickers or (self.symbol and self.symbol not in tickers):
                    continue
                
                # Parse published_utc
                if isinstance(published_utc, str):
                    try:
                        timestamp = datetime.fromisoformat(published_utc.replace('Z', '+00:00'))
                    except ValueError:
                        # Try alternative format
                        timestamp = datetime.strptime(published_utc, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                else:
                    self.logger.warning(f"Skipping news item with invalid published_utc: {published_utc}")
                    continue
                
                # Analyze sentiment
                text_to_analyze = f"{title} {description or ''}"
                sentiment = self.sentiment_analyzer.analyze(text_to_analyze)
                
                # Create records for each ticker
                for ticker in tickers:
                    # Skip if not the requested symbol and symbol is set
                    if self.symbol and ticker != self.symbol:
                        continue
                    
                    record = StandardRecord(
                        symbol=ticker,
                        timestamp=timestamp,
                        record_type=RecordType.NEWS,
                        source='polygon',
                        sentiment_score=sentiment['score'],
                        sentiment_magnitude=sentiment['magnitude'],
                        extended_data={
                            **extended_data,
                            'sentiment': sentiment
                        }
                    )
                    records.append(record)
                    self.logger.debug(f"Parsed news record for {ticker}: {title}")
            except Exception as e:
                self.logger.error(f"Error parsing news item: {str(e)}")
                
        self.logger.info(f"Parsed {len(records)} news records")
        return records, next_page
    
    def _store_news_in_db(self, records: List[StandardRecord]) -> None:
        """
        Store news records in database and cache.
        
        Args:
            records: List of StandardRecord objects
        """
        try:
            # Group records by ticker
            ticker_data = {}
            
            for record in records:
                symbol = record.symbol
                if symbol not in ticker_data:
                    ticker_data[symbol] = []
                ticker_data[symbol].append(record)
            
            # Store in TimescaleDB
            for symbol, ticker_records in ticker_data.items():
                # Convert records to DB format
                db_records = []
                
                for record in ticker_records:
                    record_dict = record.model_dump()
                    extended_data = record_dict.get('extended_data', {})
                    
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'symbol': record_dict.get('symbol'),
                        'news_id': extended_data.get('news_id'),
                        'title': extended_data.get('title'),
                        'author': extended_data.get('author'),
                        'source': extended_data.get('source'),
                        'url': extended_data.get('url'),
                        'sentiment_score': record_dict.get('sentiment_score'),
                        'sentiment_magnitude': record_dict.get('sentiment_magnitude'),
                        'metadata': extended_data
                    }
                    
                    db_records.append(db_record)
                
                # Insert into news_data table
                try:
                    count = db_client.insert_news_data(db_records)
                    self.logger.info(f"Stored {count} news records for {symbol} in TimescaleDB")
                except Exception as e:
                    self.logger.error(f"Error storing news data in TimescaleDB: {str(e)}")
                
                # Cache aggregate sentiment data in Redis
                self._cache_news_sentiment(symbol, ticker_records)
        
        except Exception as e:
            self.logger.error(f"Error in _store_news_in_db: {str(e)}")
    
    def _cache_news_sentiment(self, symbol: str, records: List[StandardRecord]) -> None:
        """
        Cache aggregate news sentiment data in Redis.
        
        Args:
            symbol: Ticker symbol
            records: List of StandardRecord objects for this symbol
        """
        try:
            # Calculate aggregate sentiment metrics
            sentiment_scores = [r.sentiment_score for r in records if hasattr(r, 'sentiment_score')]
            
            if not sentiment_scores:
                return
            
            # Calculate statistics
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Calculate mention count
            mention_count = len(records)
            
            # Number of bullish vs bearish mentions
            bullish_count = sum(1 for r in records if hasattr(r, 'sentiment_score') and r.sentiment_score > 0.05)
            bearish_count = sum(1 for r in records if hasattr(r, 'sentiment_score') and r.sentiment_score < -0.05)
            neutral_count = mention_count - bullish_count - bearish_count
            
            # Create cache data
            cache_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'avg_sentiment': avg_sentiment,
                'mention_count': mention_count,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'source': 'polygon_news'
            }
            
            # Store in Redis
            cache_key = f"news:sentiment:{symbol}"
            redis_client.set(cache_key, cache_data, self.cache_ttl)
            
            # Also store in time series list
            timeseries_key = f"news:sentiment:{symbol}:history"
            redis_client.lpush(timeseries_key, cache_data)
            redis_client.ltrim(timeseries_key, 0, 99)  # Keep last 100 entries
            redis_client.expire(timeseries_key, self.cache_ttl * 24)  # 24x longer expiry for history
            
            self.logger.debug(f"Cached news sentiment data for {symbol} in Redis")
        
        except Exception as e:
            self.logger.error(f"Error caching news sentiment data: {str(e)}")


# Helper function to create a collector instance
def create_polygon_collector(config: Optional[CollectorConfig] = None) -> PolygonCollector:
    """Create a new instance of the PolygonCollector."""
    return PolygonCollector(config)


# For testing and individual usage
if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser(description="Collect data from Polygon.io")
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to collect data for')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    parser.add_argument('--mode', type=str, choices=['market_data', 'news', 'both'], default='market_data',
                       help='Type of data to collect')
    
    args = parser.parse_args()
    
    collector = PolygonCollector()
    collector.collection_mode = args.mode
    collector.set_symbol(args.symbol)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"Collecting {args.mode} data for {args.symbol} from {start_date.date()} to {end_date.date()}")
    records = collector.collect(start_date, end_date)
    
    print(f"Collected {len(records)} records")
