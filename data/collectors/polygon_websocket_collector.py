#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polygon.io Websocket Collector for real-time market data.
This collector connects to Polygon.io's websocket API to stream
trades, quotes, and aggregated data in real-time.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import websocket
from websocket import WebSocketApp

from utils.logging import setup_logger
from config.collector_config import CollectorConfig
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType
from data.database.redis_client import get_redis_client
from data.database.timeseries_db import get_timescale_client
from data.processors.data_cache import get_data_cache
from data.processors.data_cleaner import get_data_cleaner
from data.collectors.api_key_manager import get_api_key_manager, APIKeyAuthenticationError, APIKeyRateLimitError, retry_on_rate_limit

# Setup logging
logger = setup_logger('polygon_websocket_collector', category='data')

# Initialize shared clients
redis_client = get_redis_client()
db_client = get_timescale_client()
data_cache = get_data_cache()
data_cleaner = get_data_cleaner()
api_key_manager = get_api_key_manager()


class PolygonWebsocketCollector(BaseCollector):
    """Collector for Polygon.io real-time websocket data."""

    def __init__(self, config: CollectorConfig = None):
        """
        Initialize Polygon websocket collector.
        
        Args:
            config: Optional CollectorConfig instance; loaded if None
        """
        if config is None:
            config = CollectorConfig.load('polygon')
        super().__init__(config)
        
        # Extract websocket configuration
        try:
            # Get API key from the API key manager
            credentials = api_key_manager.get_api_key('polygon', validate=True)
            self.api_key = credentials['api_key']
        except APIKeyAuthenticationError as e:
            self.logger.error(f"API key authentication failed: {str(e)}")
            self.api_key = config.api_key  # Fallback to config
        
        self.websocket_url = getattr(config, 'websocket_url', "wss://socket.polygon.io/stocks")
        
        # Get websocket specific config
        websocket_config = getattr(config, 'websocket', {})
        self.enabled = websocket_config.get('enabled', True)
        self.channels = websocket_config.get('channels', ['T', 'Q', 'AM'])  # Trades, Quotes, Minute Aggregates
        self.reconnect_interval = websocket_config.get('reconnect_interval_seconds', 30)
        self.heartbeat_interval = websocket_config.get('heartbeat_interval_seconds', 30)
        self.max_retries = websocket_config.get('max_retries', 5)
        self.retry_delay = websocket_config.get('retry_delay_seconds', 2)
        
        # Subscriptions
        self.symbols = set()
        self.subscribed_channels = set()
        
        # Websocket connection
        self.ws: Optional[WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_connected = False
        self.should_reconnect = True
        self.connection_attempts = 0
        
        # Last heartbeat time
        self.last_heartbeat = time.time()
        
        # Heartbeat checker thread
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Cache TTL (in seconds)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 60)
        
        # Keep track of processed messages to avoid duplicates (can happen during reconnections)
        self.processed_messages = set()
        self.processed_messages_max_size = 10000  # Maximum size before clearing
        
        # Status tracking
        self.status = "initialized"
        
        # Data cleaner configuration
        self.clean_data = websocket_config.get('clean_data', True)
        self.validate_data = websocket_config.get('validate_data', True)
        
        # Data storage options
        self.store_trades = websocket_config.get('store_trades', True)  # Store trades in database
        self.store_quotes = websocket_config.get('store_quotes', False)  # Quotes are high volume, default to not storing
        self.batch_size = websocket_config.get('db_batch_size', 100)  # Batch size for db inserts
        self.batch_inserts = websocket_config.get('batch_inserts', True)  # Use batching for db inserts
        self.batch_queues = {  # Queues for batched inserts
            RecordType.TRADE: [],
            RecordType.QUOTE: [],
            RecordType.AGGREGATE: [],
            RecordType.BAR: []
        }
        self.last_batch_insert = time.time()
        self.batch_interval = websocket_config.get('batch_interval_seconds', 5)  # How often to insert batches
        
        # Performance tracking
        self.message_count = 0
        self.message_count_by_type = {
            'T': 0,  # Trades
            'Q': 0,  # Quotes
            'AM': 0,  # Minute aggregates
            'A': 0    # Second aggregates
        }
        self.start_time = None
        
        self.logger.info(f"Initialized Polygon Websocket Collector with API key: {self.api_key[:4]}...{self.api_key[-4:]}")

    def _authenticate(self) -> None:
        """
        Authenticate with Polygon.io websocket API.
        """
        if not self.ws or not self.is_connected:
            self.logger.error("Cannot authenticate: websocket not connected")
            return
        
        auth_message = {"action": "auth", "params": self.api_key}
        self.ws.send(json.dumps(auth_message))
        self.logger.info("Sent authentication message to Polygon websocket")

    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Not used for websocket collector, but required by BaseCollector.
        """
        raise NotImplementedError("Websocket collector does not support paged requests")

    def _parse(self, raw: Any) -> tuple[List[StandardRecord], Optional[str]]:
        """
        Not used for websocket collector, but required by BaseCollector.
        """
        raise NotImplementedError("Websocket collector does not support paged requests")

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        """
        Handle incoming websocket messages.
        
        Args:
            ws: WebSocketApp instance
            message: Message received from websocket
        """
        try:
            # Update last heartbeat time
            self.last_heartbeat = time.time()
            
            # Parse message
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                # Process each message in the list
                for msg in data:
                    self._process_message(msg)
            elif isinstance(data, dict):
                # Process single message
                self._process_message(data)
            else:
                self.logger.warning(f"Received unexpected message format: {message}")
        
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            
            # If we encountered an error during message processing, let's log the raw message for debugging
            try:
                truncated_message = message[:500] + "..." if len(message) > 500 else message
                self.logger.debug(f"Raw message that caused error: {truncated_message}")
            except Exception:
                pass  # If we can't log the message, just continue

    def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single message from the websocket.
        
        Args:
            message: Message dictionary
        """
        # Check if we've already processed this message (can happen during reconnects)
        # Create a hash of the message to check for duplicates
        try:
            message_hash = hash(json.dumps(message, sort_keys=True))
            if message_hash in self.processed_messages:
                return
            
            # Add to processed messages and maintain maximum size
            self.processed_messages.add(message_hash)
            if len(self.processed_messages) > self.processed_messages_max_size:
                # Clear half of the oldest entries if we hit the limit
                self.processed_messages = set(list(self.processed_messages)[self.processed_messages_max_size//2:])
        except (TypeError, ValueError):
            # If we can't hash the message, just continue
            pass
            
        # Check message type
        event_type = message.get('ev')
        
        # Update message count for performance tracking
        self.message_count += 1
        if event_type in self.message_count_by_type:
            self.message_count_by_type[event_type] += 1
        
        # Log periodic performance metrics
        if self.message_count % 10000 == 0 and self.start_time:
            elapsed = time.time() - self.start_time
            msgs_per_sec = self.message_count / elapsed if elapsed > 0 else 0
            self.logger.info(f"Processed {self.message_count} messages in {elapsed:.2f}s ({msgs_per_sec:.2f} msgs/sec)")
            self.logger.info(f"Message types: {self.message_count_by_type}")
        
        if event_type == 'status':
            # Status message
            status = message.get('status')
            message_text = message.get('message', '')
            
            # Update internal status
            self.status = status
            
            if status == 'connected':
                self.logger.info(f"Connected to Polygon websocket: {message_text}")
            elif status == 'auth_success':
                self.logger.info(f"Successfully authenticated with Polygon websocket: {message_text}")
                # Subscribe to channels after successful authentication
                self._subscribe_to_channels()
                # Reset connection attempts on successful auth
                self.connection_attempts = 0
            elif status == 'auth_failed':
                self.logger.error(f"Authentication failed: {message_text}")
                # Notify API key manager about auth failure
                try:
                    api_key_manager.validate_api_key('polygon', {'api_key': self.api_key})
                except Exception:
                    pass  # Already logged in the API key manager
            else:
                self.logger.info(f"Received status message: {status} - {message_text}")
        
        elif event_type == 'T':
            # Trade message
            self._process_trade(message)
        
        elif event_type == 'Q':
            # Quote message
            self._process_quote(message)
        
        elif event_type == 'AM':
            # Minute aggregate (bar) message
            self._process_aggregate(message)
        
        elif event_type == 'A':
            # Second aggregate message
            self._process_aggregate(message, is_second_agg=True)
        
        elif event_type == 'subscription':
            # Subscription confirmation
            sub_status = message.get('status')
            subs = message.get('subscriptions', [])
            self.logger.info(f"Subscription {sub_status}: {subs}")
        
        else:
            # Unknown message type
            self.logger.debug(f"Received unknown message type: {event_type}")
        
        # Check if we need to perform a batch insert
        current_time = time.time()
        if self.batch_inserts and (current_time - self.last_batch_insert) > self.batch_interval:
            self._perform_batch_inserts()
            self.last_batch_insert = current_time

    def _process_trade(self, message: Dict[str, Any]) -> None:
        """
        Process a trade message.
        
        Args:
            message: Trade message dictionary
        """
        try:
            # Extract trade data
            symbol = message.get('sym', '')
            price = message.get('p', 0)
            size = message.get('s', 0)
            timestamp_ms = message.get('t', 0)
            exchange = message.get('x', '')
            trade_id = message.get('i', '')
            conditions = message.get('c', [])
            
            # Convert timestamp to datetime
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            
            # Create StandardRecord
            record = StandardRecord(
                symbol=symbol,
                timestamp=timestamp,
                record_type=RecordType.TRADE,
                source='polygon',
                price=Decimal(str(price)),
                volume=size,
                exchange=exchange,
                trade_id=trade_id,
                conditions=conditions,
                extended_data=message
            )
            
            # Clean the data if enabled
            if self.clean_data:
                record_dict = record.model_dump()
                clean_data = data_cleaner.clean_websocket_data(record_dict, 'trades', symbol)
                
                # Update record with cleaned data
                for key, value in clean_data.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
            
            # Validate the data if enabled
            if self.validate_data:
                if not data_cleaner.validate_websocket_data(record.model_dump(), 'trades'):
                    self.logger.warning(f"Invalid trade data for {symbol}")
                    return
            
            # Store in Redis cache using enhanced DataCache
            data_cache.add_websocket_data(symbol, 'trades', record.model_dump())
            
            # Store in database if enabled
            if self.store_trades:
                if self.batch_inserts:
                    self.batch_queues[RecordType.TRADE].append(record)
                    if len(self.batch_queues[RecordType.TRADE]) >= self.batch_size:
                        self._insert_trade_batch()
                else:
                    self._store_record_in_db(record)
            
            self.logger.debug(f"Processed trade for {symbol}: price={price}, size={size}")
        
        except Exception as e:
            self.logger.error(f"Error processing trade message: {str(e)}")

    def _process_quote(self, message: Dict[str, Any]) -> None:
        """
        Process a quote message.
        
        Args:
            message: Quote message dictionary
        """
        try:
            # Extract quote data
            symbol = message.get('sym', '')
            bid_price = message.get('bp', 0)
            bid_size = message.get('bs', 0)
            ask_price = message.get('ap', 0)
            ask_size = message.get('as', 0)
            timestamp_ms = message.get('t', 0)
            exchange = message.get('x', '')
            conditions = message.get('c', [])
            
            # Convert timestamp to datetime
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            
            # Create StandardRecord
            record = StandardRecord(
                symbol=symbol,
                timestamp=timestamp,
                record_type=RecordType.QUOTE,
                source='polygon',
                bid_price=Decimal(str(bid_price)),
                bid_size=bid_size,
                ask_price=Decimal(str(ask_price)),
                ask_size=ask_size,
                exchange=exchange,
                conditions=conditions,
                extended_data=message
            )
            
            # Clean the data if enabled
            if self.clean_data:
                record_dict = record.model_dump()
                clean_data = data_cleaner.clean_websocket_data(record_dict, 'quotes', symbol)
                
                # Update record with cleaned data
                for key, value in clean_data.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
            
            # Validate the data if enabled
            if self.validate_data:
                if not data_cleaner.validate_websocket_data(record.model_dump(), 'quotes'):
                    self.logger.warning(f"Invalid quote data for {symbol}")
                    return
            
            # Store in Redis cache using enhanced DataCache
            data_cache.add_websocket_data(symbol, 'quotes', record.model_dump())
            
            # Store in database if enabled
            if self.store_quotes:
                if self.batch_inserts:
                    self.batch_queues[RecordType.QUOTE].append(record)
                    if len(self.batch_queues[RecordType.QUOTE]) >= self.batch_size:
                        self._insert_quote_batch()
                else:
                    self._store_record_in_db(record)
            
            self.logger.debug(f"Processed quote for {symbol}: bid={bid_price}x{bid_size}, ask={ask_price}x{ask_size}")
        
        except Exception as e:
            self.logger.error(f"Error processing quote message: {str(e)}")

    def _process_aggregate(self, message: Dict[str, Any], is_second_agg: bool = False) -> None:
        """
        Process an aggregate (bar) message.
        
        Args:
            message: Aggregate message dictionary
            is_second_agg: Whether this is a second (A) or minute (AM) aggregate
        """
        try:
            # Extract aggregate data
            symbol = message.get('sym', '')
            open_price = message.get('o', 0)
            high_price = message.get('h', 0)
            low_price = message.get('l', 0)
            close_price = message.get('c', 0)
            volume = message.get('v', 0)
            vwap = message.get('vw', 0)
            timestamp_ms = message.get('t', 0)
            num_items = message.get('n', 0)  # Number of items in the aggregate
            
            # Convert timestamp to datetime
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            
            # Create StandardRecord
            record = StandardRecord(
                symbol=symbol,
                timestamp=timestamp,
                record_type=RecordType.AGGREGATE,
                source='polygon',
                open=Decimal(str(open_price)),
                high=Decimal(str(high_price)),
                low=Decimal(str(low_price)),
                close=Decimal(str(close_price)),
                volume=volume,
                vwap=Decimal(str(vwap)) if vwap else None,
                extended_data={
                    **message,
                    'aggregate_type': 'second' if is_second_agg else 'minute',
                    'num_items': num_items
                }
            )
            
            # Clean the data if enabled
            if self.clean_data:
                record_dict = record.model_dump()
                clean_data = data_cleaner.clean_websocket_data(record_dict, 'bars', symbol)
                
                # Update record with cleaned data
                for key, value in clean_data.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
            
            # Validate the data if enabled
            if self.validate_data:
                if not data_cleaner.validate_websocket_data(record.model_dump(), 'bars'):
                    self.logger.warning(f"Invalid bar data for {symbol}")
                    return
            
            # Store in Redis cache using enhanced DataCache
            data_cache.add_websocket_data(symbol, 'bars', record.model_dump())
            
            # Store in database
            if self.batch_inserts:
                self.batch_queues[RecordType.AGGREGATE].append(record)
                if len(self.batch_queues[RecordType.AGGREGATE]) >= self.batch_size:
                    self._insert_aggregate_batch()
            else:
                self._store_record_in_db(record)
            
            self.logger.debug(f"Processed {'second' if is_second_agg else 'minute'} aggregate for {symbol}: open={open_price}, high={high_price}, low={low_price}, close={close_price}, volume={volume}")
        
        except Exception as e:
            self.logger.error(f"Error processing aggregate message: {str(e)}")

    def _perform_batch_inserts(self) -> None:
        """
        Insert all batched records into the database.
        """
        for record_type, queue in self.batch_queues.items():
            if queue:
                if record_type == RecordType.TRADE:
                    self._insert_trade_batch()
                elif record_type == RecordType.QUOTE:
                    self._insert_quote_batch()
                elif record_type in [RecordType.AGGREGATE, RecordType.BAR]:
                    self._insert_aggregate_batch()

    def _insert_trade_batch(self) -> None:
        """Insert a batch of trade records into the database."""
        if not self.batch_queues[RecordType.TRADE]:
            return
            
        try:
            # Convert records to database format
            db_records = []
            for record in self.batch_queues[RecordType.TRADE]:
                record_dict = record.model_dump()
                db_record = {
                    'time': record_dict.get('timestamp'),
                    'symbol': record_dict.get('symbol'),
                    'order_id': record_dict.get('trade_id', ''),
                    'trade_id': record_dict.get('trade_id'),
                    'side': 'unknown',  # Polygon doesn't provide trade side
                    'quantity': record_dict.get('volume', 0),
                    'price': record_dict.get('price', 0),
                    'exchange': record_dict.get('exchange'),
                    'source': record_dict.get('source'),
                    'metadata': record_dict.get('extended_data')
                }
                db_records.append(db_record)
            
            # Insert batch
            if db_records:
                db_client.insert_trade_data(db_records)
                self.logger.debug(f"Inserted {len(db_records)} trade records")
            
            # Clear queue
            self.batch_queues[RecordType.TRADE] = []
        except Exception as e:
            self.logger.error(f"Error inserting trade batch: {str(e)}")

    def _insert_quote_batch(self) -> None:
        """Insert a batch of quote records into the database."""
        if not self.batch_queues[RecordType.QUOTE]:
            return
            
        try:
            # Convert records to database format
            db_records = []
            for record in self.batch_queues[RecordType.QUOTE]:
                record_dict = record.model_dump()
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
                        **record_dict.get('extended_data', {})
                    }
                }
                db_records.append(db_record)
            
            # Insert batch
            if db_records:
                db_client.insert_analytics_data(db_records)
                self.logger.debug(f"Inserted {len(db_records)} quote records")
            
            # Clear queue
            self.batch_queues[RecordType.QUOTE] = []
        except Exception as e:
            self.logger.error(f"Error inserting quote batch: {str(e)}")

    def _insert_aggregate_batch(self) -> None:
        """Insert a batch of aggregate records into the database."""
        if not self.batch_queues[RecordType.AGGREGATE]:
            return
            
        try:
            # Convert records to database format
            db_records = []
            for record in self.batch_queues[RecordType.AGGREGATE]:
                record_dict = record.model_dump()
                db_record = {
                    'time': record_dict.get('timestamp'),
                    'symbol': record_dict.get('symbol'),
                    'open': record_dict.get('open'),
                    'high': record_dict.get('high'),
                    'low': record_dict.get('low'),
                    'close': record_dict.get('close'),
                    'volume': record_dict.get('volume'),
                    'vwap': record_dict.get('vwap'),
                    'num_trades': record_dict.get('extended_data', {}).get('n'),  # Number of trades in aggregate
                    'source': record_dict.get('source'),
                    'data_type': 'bars',
                    'record_type': 'aggregate',
                    'metadata': record_dict.get('extended_data')
                }
                db_records.append(db_record)
            
            # Insert batch
            if db_records:
                db_client.insert_market_data(db_records)
                self.logger.debug(f"Inserted {len(db_records)} aggregate records")
            
            # Clear queue
            self.batch_queues[RecordType.AGGREGATE] = []
        except Exception as e:
            self.logger.error(f"Error inserting aggregate batch: {str(e)}")

    def _store_record_in_db(self, record: StandardRecord) -> None:
        """
        Store a record in the database.
        
        Args:
            record: StandardRecord to store
        """
        try:
            # Convert record to database format
            record_dict = record.model_dump()
            
            if record.record_type == RecordType.AGGREGATE or record.record_type == RecordType.BAR:
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
                    'num_trades': record_dict.get('extended_data', {}).get('n'),  # Number of trades in aggregate
                    'source': record_dict.get('source'),
                    'data_type': 'bars',
                    'record_type': 'aggregate',
                    'metadata': record_dict.get('extended_data')
                }
                db_client.insert_market_data([db_record])
            
            elif record.record_type == RecordType.TRADE:
                # Store as trade data
                db_record = {
                    'time': record_dict.get('timestamp'),
                    'order_id': record_dict.get('trade_id', ''),
                    'trade_id': record_dict.get('trade_id'),
                    'symbol': record_dict.get('symbol'),
                    'side': 'unknown',  # Polygon doesn't provide trade side
                    'quantity': record_dict.get('volume', 0),
                    'price': record_dict.get('price', 0),
                    'exchange': record_dict.get('exchange'),
                    'source': record_dict.get('source'),
                    'metadata': record_dict.get('extended_data')
                }
                db_client.insert_trade_data([db_record])
            
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
                        **record_dict.get('extended_data', {})
                    }
                }
                db_client.insert_analytics_data([db_record])
        
        except Exception as e:
            self.logger.error(f"Error storing record in database: {str(e)}")

    def _on_error(self, ws: WebSocketApp, error: Exception) -> None:
        """
        Handle websocket errors.
        
        Args:
            ws: WebSocketApp instance
            error: Error that occurred
        """
        self.logger.error(f"Websocket error: {str(error)}")
        self.status = "error"

    def _on_close(self, ws: WebSocketApp, close_status_code: Optional[int], close_msg: Optional[str]) -> None:
        """
        Handle websocket close events.
        
        Args:
            ws: WebSocketApp instance
            close_status_code: Status code for close
            close_msg: Close message
        """
        self.is_connected = False
        self.status = "disconnected"
        self.logger.info(f"Websocket closed: {close_status_code} - {close_msg}")
        
        # Perform any queued database inserts
        self._perform_batch_inserts()
        
        # Check API key validity on unexpected disconnect
        if close_status_code not in [1000, 1001]:  # Normal close codes
            try:
                api_key_manager.validate_api_key('polygon', {'api_key': self.api_key})
            except Exception:
                pass  # Already logged in API key manager
        
        # Reconnect if needed
        if self.should_reconnect:
            # Use exponential backoff for reconnect attempts
            delay = min(self.reconnect_interval * (2 ** min(self.connection_attempts, 5)), 300)
            self.connection_attempts += 1
            self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.connection_attempts})...")
            time.sleep(delay)
            self._connect()

    def _on_open(self, ws: WebSocketApp) -> None:
        """
        Handle websocket open events.
        
        Args:
            ws: WebSocketApp instance
        """
        self.is_connected = True
        self.status = "connected"
        self.logger.info("Websocket connection established")
        
        # Reset performance tracking
        self.message_count = 0
        self.message_count_by_type = {k: 0 for k in self.message_count_by_type}
        self.start_time = time.time()
        
        # Authenticate
        self._authenticate()

    def _connect(self) -> None:
        """
        Connect to the Polygon websocket API.
        """
        if self.ws:
            self.logger.info("Closing existing websocket connection")
            try:
                self.ws.close()
            except Exception as e:
                self.logger.warning(f"Error closing existing websocket: {str(e)}")
        
        self.logger.info(f"Connecting to Polygon websocket: {self.websocket_url}")
        
        # Create websocket connection
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Start websocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _subscribe_to_channels(self) -> None:
        """
        Subscribe to the configured channels for all symbols.
        """
        if not self.ws or not self.is_connected:
            self.logger.error("Cannot subscribe: websocket not connected")
            return
        
        if not self.symbols:
            self.logger.warning("No symbols to subscribe to")
            return
        
        # Subscribe to each channel for all symbols
        for channel in self.channels:
            symbols_list = list(self.symbols)
            # Batch subscriptions to avoid large messages
            batch_size = 100
            for i in range(0, len(symbols_list), batch_size):
                batch = symbols_list[i:i+batch_size]
                subscription_message = {
                    "action": "subscribe",
                    "params": [f"{channel}.{symbol}" for symbol in batch]
                }
                self.ws.send(json.dumps(subscription_message))
                
                # Add to subscribed channels
                for symbol in batch:
                    self.subscribed_channels.add(f"{channel}.{symbol}")
                
                self.logger.info(f"Subscribed to {channel} for {len(batch)} symbols")
                # Small delay between batches to avoid rate limits
                if i + batch_size < len(symbols_list):
                    time.sleep(0.5)

    def _check_heartbeat(self) -> None:
        """
        Check if heartbeats are being received and reconnect if needed.
        """
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat > self.heartbeat_interval * 2:
                    self.logger.warning(f"No heartbeat received for {current_time - self.last_heartbeat:.1f} seconds, reconnecting...")
                    self._connect()
                    self.last_heartbeat = current_time
                
                # Also check if we need to perform a batch insert
                if self.batch_inserts and (current_time - self.last_batch_insert) > self.batch_interval:
                    self._perform_batch_inserts()
                    self.last_batch_insert = current_time
                
                # Sleep for a portion of the heartbeat interval
                time.sleep(self.heartbeat_interval / 2)
            except Exception as e:
                self.logger.error(f"Error in heartbeat checker: {str(e)}")
                time.sleep(self.heartbeat_interval / 2)

    @retry_on_rate_limit(max_retries=3)
    def start(self) -> None:
        """
        Start the websocket collector.
        """
        if not self.enabled:
            self.logger.warning("Websocket collector is disabled in configuration")
            return
        
        self.logger.info("Starting Polygon websocket collector")
        
        # Set running flag
        self.running = True
        
        # Initialize performance tracking
        self.message_count = 0
        self.message_count_by_type = {k: 0 for k in self.message_count_by_type}
        self.start_time = time.time()
        self.last_batch_insert = time.time()
        
        # Connect to websocket
        self._connect()
        
        # Start heartbeat checker
        self.heartbeat_thread = threading.Thread(target=self._check_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        self.logger.info("Polygon websocket collector started")

    def stop(self) -> None:
        """
        Stop the websocket collector.
        """
        self.logger.info("Stopping Polygon websocket collector")
        
        # Set flags
        self.running = False
        self.should_reconnect = False
        
        # Perform any pending batch inserts
        try:
            self._perform_batch_inserts()
        except Exception as e:
            self.logger.error(f"Error performing final batch inserts: {str(e)}")
        
        # Close websocket
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                self.logger.warning(f"Error closing websocket: {str(e)}")
        
        # Wait for threads to finish
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        # Log performance statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            msgs_per_sec = self.message_count / elapsed if elapsed > 0 else 0
            self.logger.info(f"Processed {self.message_count} messages in {elapsed:.2f}s ({msgs_per_sec:.2f} msgs/sec)")
            self.logger.info(f"Message breakdown: {self.message_count_by_type}")
        
        self.status = "stopped"
        self.logger.info("Polygon websocket collector stopped")

    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the subscription list.
        
        Args:
            symbol: Symbol to add
        """
        symbol = symbol.upper()
        
        if symbol in self.symbols:
            self.logger.debug(f"Symbol {symbol} already in subscription list")
            return
        
        self.symbols.add(symbol)
        self.logger.info(f"Added symbol {symbol} to subscription list")
        
        # Subscribe to channels for this symbol if already connected
        if self.is_connected:
            for channel in self.channels:
                subscription_message = {
                    "action": "subscribe",
                    "params": [f"{channel}.{symbol}"]
                }
                self.ws.send(json.dumps(subscription_message))
                self.subscribed_channels.add(f"{channel}.{symbol}")
                self.logger.info(f"Subscribed to {channel} for {symbol}")

    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the subscription list.
        
        Args:
            symbol: Symbol to remove
        """
        symbol = symbol.upper()
        
        if symbol not in self.symbols:
            self.logger.debug(f"Symbol {symbol} not in subscription list")
            return
        
        self.symbols.remove(symbol)
        self.logger.info(f"Removed symbol {symbol} from subscription list")
        
        # Unsubscribe from channels for this symbol if connected
        if self.is_connected:
            for channel in self.channels:
                channel_symbol = f"{channel}.{symbol}"
                if channel_symbol in self.subscribed_channels:
                    unsubscribe_message = {
                        "action": "unsubscribe",
                        "params": [channel_symbol]
                    }
                    self.ws.send(json.dumps(unsubscribe_message))
                    self.subscribed_channels.remove(channel_symbol)
                    self.logger.info(f"Unsubscribed from {channel} for {symbol}")

    def get_latest_data(self, symbol: str, record_type: RecordType) -> Optional[Dict[str, Any]]:
        """
        Get the latest data for a symbol and record type.
        
        Args:
            symbol: Symbol to get data for
            record_type: Type of record to get
            
        Returns:
            Latest data or None if not available
        """
        symbol = symbol.upper()
        type_str = record_type.value
        
        # Get from DataCache first (more recent)
        data_type = 'trades' if record_type == RecordType.TRADE else \
                   'quotes' if record_type == RecordType.QUOTE else \
                   'bars'
                   
        latest_ws = data_cache.get_latest_websocket_data(symbol, data_type, 1)
        if latest_ws and len(latest_ws) > 0:
            return latest_ws[0]
        
        # Fall back to Redis
        key = f"polygon:latest:{type_str}:{symbol}"
        data = redis_client.get(key)
        
        return data

    def get_recent_data(self, symbol: str, record_type: RecordType, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent data for a symbol and record type.
        
        Args:
            symbol: Symbol to get data for
            record_type: Type of record to get
            limit: Maximum number of records to return
            
        Returns:
            List of recent data records
        """
        symbol = symbol.upper()
        type_str = record_type.value
        
        # Get from Redis cache using DataCache
        data_type = 'trades' if record_type == RecordType.TRADE else \
                   'quotes' if record_type == RecordType.QUOTE else \
                   'bars'
                   
        # Get recent WebSocket data
        start_time = time.time() - 86400  # Last 24 hours
        end_time = time.time()
        ws_data = data_cache.get_websocket_data_range(symbol, data_type, start_time, end_time)
        
        # If we have enough data from WebSocket, return it
        if ws_data and len(ws_data) >= limit:
            return ws_data[-limit:]
            
        # Fall back to legacy Redis cache
        key = f"polygon:websocket:{type_str}:{symbol}:recent"
        redis_data = redis_client.lrange(key, 0, limit - 1)
        
        # Combine data sources if needed
        if ws_data and redis_data:
            # Deduplicate by timestamp if possible
            combined = ws_data
            try:
                ws_timestamps = {item['timestamp'] for item in ws_data if 'timestamp' in item}
                for item in redis_data:
                    if 'timestamp' not in item or item['timestamp'] not in ws_timestamps:
                        combined.append(item)
                
                # Sort by timestamp
                combined = sorted(combined, key=lambda x: x.get('timestamp', 0))
                
                return combined[-limit:]
            except Exception:
                # If we can't deduplicate, just combine
                return (ws_data + redis_data)[:limit]
        
        # Return whatever we have
        return ws_data or redis_data or []
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the collector.
        
        Returns:
            Status dictionary
        """
        status_dict = {
            'status': self.status,
            'connected': self.is_connected,
            'symbols': len(self.symbols),
            'channels': len(self.channels),
            'subscriptions': len(self.subscribed_channels),
            'message_count': self.message_count,
            'message_types': self.message_count_by_type,
            'batch_queues': {k.value: len(v) for k, v in self.batch_queues.items()}
        }
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            status_dict['uptime_seconds'] = elapsed
            status_dict['messages_per_second'] = self.message_count / elapsed if elapsed > 0 else 0
        
        return status_dict

    def get_active_symbols(self) -> List[str]:
        """
        Get list of currently active symbols.
        
        Returns:
            List of symbol strings
        """
        return list(self.symbols)
