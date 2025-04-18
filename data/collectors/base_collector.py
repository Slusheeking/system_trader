from abc import ABC, abstractmethod
from datetime import datetime
import logging
import time
from typing import Any, List, Optional, Tuple, Dict, Union

from config.collector_config import CollectorConfig
from data.collectors.schema import StandardRecord
from utils.logging import setup_logger
from data.database.redis_client import get_redis_client
from data.database.timeseries_db import get_timescale_client


class CollectorError(Exception):
    """Custom exception for collector errors."""
    pass


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    def __init__(self, config: CollectorConfig):
        """
        Initialize the collector with configuration.
        
        Args:
            config: CollectorConfig instance with settings for this collector
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, category='data')
        
        # Retry settings from config or defaults
        self.retry_attempts: int = getattr(config, 'retry_attempts', 3)
        self.backoff_factor: float = getattr(config, 'backoff_factor', 1.0)
        
        # Cache TTL (in seconds)
        self.cache_ttl: int = getattr(config, 'cache_ttl_seconds', 3600)  # 1 hour default
        
        # Initialize shared database clients
        self.redis_client = get_redis_client()
        self.db_client = get_timescale_client()
        
        # Status tracking
        self.status = "initialized"

    @abstractmethod
    def _authenticate(self) -> None:
        """Perform any setup or authentication required for the API client."""
        pass

    @abstractmethod
    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Request a single page of raw data from the data source.
        
        Args:
            page_token: Optional token for pagination
            
        Returns:
            Raw data from the API
        """
        pass

    @abstractmethod
    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw page data into standard records and return next page token.
        
        Args:
            raw: Raw data from _request_page
            
        Returns:
            Tuple of (list of StandardRecord, next page token or None)
        """
        pass

    def collect(self, start: datetime, end: datetime, page_token: Optional[str] = None) -> List[StandardRecord]:
        """
        Collect data between start and end datetime, handling retries, backoff, and pagination.
        
        Args:
            start: Start datetime
            end: End datetime
            page_token: Optional starting page token for pagination
            
        Returns:
            List of StandardRecord instances
        """
        self.logger.info(f"Starting collection from {start} to {end}")
        start_time = time.time()
        pages_fetched = 0
        self.status = "collecting"

        try:
            # Authenticate or set up the client
            self._authenticate()

            all_records: List[StandardRecord] = []
            current_page_token: Optional[str] = page_token

            while True:
                attempt = 0
                while attempt < self.retry_attempts:
                    try:
                        raw_page = self._request_page(current_page_token)
                        break
                    except Exception as e:
                        attempt += 1
                        self.logger.warning(
                            f"Attempt {attempt}/{self.retry_attempts} failed for page {current_page_token}: {e}"
                        )
                        if attempt >= self.retry_attempts:
                            self.status = "error"
                            raise CollectorError(
                                f"Failed to fetch page {current_page_token} after {self.retry_attempts} attempts"
                            ) from e
                        sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                        time.sleep(sleep_time)

                # Parse the retrieved page
                try:
                    records, next_page_token = self._parse(raw_page)
                except Exception as e:
                    self.status = "error"
                    raise CollectorError(f"Error parsing page {current_page_token}: {e}") from e

                all_records.extend(records)
                pages_fetched += 1
                self.logger.debug(f"Fetched page {pages_fetched} with {len(records)} records")

                if not next_page_token:
                    break
                current_page_token = next_page_token

            elapsed = time.time() - start_time
            self.logger.info(
                f"Completed collection: {len(all_records)} records in {pages_fetched} pages in {elapsed:.2f}s"
            )
            self.status = "completed"
            return all_records

        except CollectorError:
            self.status = "error"
            raise
        except Exception as e:
            self.status = "error"
            raise CollectorError(f"Unexpected error during collection: {e}") from e
    
    def _cache_record(self, record: StandardRecord, ttl: Optional[int] = None) -> bool:
        """
        Cache a record in Redis.
        
        Args:
            record: StandardRecord to cache
            ttl: Optional TTL override in seconds
            
        Returns:
            Boolean indicating success
        """
        try:
            # Use provided TTL or default from config
            cache_ttl = ttl if ttl is not None else self.cache_ttl
            
            # Create cache key based on record type and symbol
            record_type = record.record_type.value
            symbol = record.symbol
            source = record.source
            timestamp = record.timestamp.isoformat()
            cache_key = f"{source}:{record_type}:{symbol}:{timestamp}"
            
            # Convert record to dictionary
            record_dict = record.model_dump()
            
            # Store in Redis
            self.redis_client.set(cache_key, record_dict, cache_ttl)
            
            # Also store as latest record for this symbol and type
            latest_key = f"{source}:latest:{record_type}:{symbol}"
            self.redis_client.set(latest_key, record_dict, cache_ttl * 2)  # Keep latest data longer
            
            return True
        except Exception as e:
            self.logger.error(f"Error caching record in Redis: {str(e)}")
            return False
    
    def _store_records_in_db(self, records: List[StandardRecord]) -> int:
        """
        Store records in the TimescaleDB.
        
        Args:
            records: List of StandardRecord to store
            
        Returns:
            Number of records stored
        """
        if not records:
            return 0
            
        try:
            # Group records by type for bulk inserts
            market_data = []
            trade_data = []
            analytics_data = []
            social_data = []
            
            for record in records:
                record_dict = record.model_dump()
                
                if record.record_type in [record.record_type.BAR, record.record_type.AGGREGATE]:
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
                        'source': record_dict.get('source'),
                        'metadata': record_dict.get('extended_data')
                    }
                    market_data.append(db_record)
                
                elif record.record_type == record.record_type.TRADE:
                    # Store as trade data
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'order_id': record_dict.get('trade_id', ''),
                        'trade_id': record_dict.get('trade_id'),
                        'symbol': record_dict.get('symbol'),
                        'side': record_dict.get('extended_data', {}).get('side', 'unknown'),
                        'quantity': record_dict.get('volume', 0),
                        'price': record_dict.get('price', 0),
                        'source': record_dict.get('source'),
                        'metadata': record_dict.get('extended_data')
                    }
                    trade_data.append(db_record)
                
                elif record.record_type == record.record_type.QUOTE:
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
                            'source': record_dict.get('source'),
                            **record_dict.get('extended_data', {})
                        }
                    }
                    analytics_data.append(db_record)
                
                elif record.record_type == record.record_type.SOCIAL:
                    # Store as social data
                    extended_data = record_dict.get('extended_data', {})
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'symbol': record_dict.get('symbol'),
                        'source': record_dict.get('source'),
                        'platform': extended_data.get('platform', record_dict.get('source')),
                        'subreddit': extended_data.get('subreddit'),
                        'post_id': extended_data.get('id') or extended_data.get('post_id'),
                        'parent_id': extended_data.get('parent_id'),
                        'author': extended_data.get('author'),
                        'content_type': extended_data.get('post_type', 'unknown'),
                        'sentiment_score': record_dict.get('sentiment_score'),
                        'sentiment_magnitude': record_dict.get('sentiment_magnitude'),
                        'score': extended_data.get('score', 0),
                        'metadata': extended_data
                    }
                    social_data.append(db_record)
            
            # Bulk insert to database
            total_inserted = 0
            
            if market_data:
                count = self.db_client.insert_market_data(market_data)
                total_inserted += count
                self.logger.debug(f"Inserted {count} market data records")
                
            if trade_data:
                count = self.db_client.insert_trade_data(trade_data)
                total_inserted += count
                self.logger.debug(f"Inserted {count} trade data records")
                
            if analytics_data:
                count = self.db_client.insert_analytics_data(analytics_data)
                total_inserted += count
                self.logger.debug(f"Inserted {count} analytics data records")
                
            if social_data:
                count = self.db_client.insert_social_data(social_data)
                total_inserted += count
                self.logger.debug(f"Inserted {count} social data records")
            
            return total_inserted
            
        except Exception as e:
            self.logger.error(f"Error storing records in database: {str(e)}")
            return 0
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get collector status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "status": self.status,
            "collector_type": self.__class__.__name__,
            "retry_config": {
                "retry_attempts": self.retry_attempts,
                "backoff_factor": self.backoff_factor
            },
            "cache_ttl": self.cache_ttl
        }
    
    def set_retry_config(self, attempts: int, backoff_factor: float) -> None:
        """
        Update retry configuration.
        
        Args:
            attempts: Number of retry attempts
            backoff_factor: Backoff factor for exponential backoff
        """
        self.retry_attempts = attempts
        self.backoff_factor = backoff_factor
        self.logger.info(f"Updated retry config: attempts={attempts}, backoff_factor={backoff_factor}")
    
    def set_cache_ttl(self, ttl: int) -> None:
        """
        Update cache TTL.
        
        Args:
            ttl: Cache TTL in seconds
        """
        self.cache_ttl = ttl
        self.logger.info(f"Updated cache TTL: {ttl} seconds")
