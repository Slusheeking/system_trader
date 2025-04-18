#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Markets Websocket Collector for real-time market data and account updates.
This collector connects to Alpaca's websocket APIs to stream trades, quotes, bars,
and account updates in real-time.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import os
import threading
import time
import yaml
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
logger = setup_logger('alpaca_websocket_collector', category='data')

# Initialize shared clients
redis_client = get_redis_client()
db_client = get_timescale_client()
data_cache = get_data_cache()
data_cleaner = get_data_cleaner()
api_key_manager = get_api_key_manager()


class AlpacaWebsocketCollector(BaseCollector):
    """Collector for Alpaca real-time websocket data, integrating with the system architecture."""
    
    # WebSocket endpoints
    MARKET_WS_URL = "wss://stream.data.alpaca.markets/v2"
    ACCOUNT_WS_URL = "wss://api.alpaca.markets/stream"
    
    def __init__(self, config: CollectorConfig = None):
        """
        Initialize Alpaca websocket collector.
        
        Args:
            config: Optional CollectorConfig instance; loaded if None
        """
        if config is None:
            config = CollectorConfig.load('alpaca')
        super().__init__(config)
        
        # Extract configuration using API key manager
        try:
            # Get credentials from the API key manager
            credentials = api_key_manager.get_api_key('alpaca', validate=True)
            self.api_key = credentials['api_key']
            self.api_secret = credentials['api_secret']
        except APIKeyAuthenticationError as e:
            self.logger.error(f"API key authentication failed: {str(e)}")
            # Fallback to config
            self.api_key = getattr(config, 'api_key', None)
            self.api_secret = getattr(config, 'api_secret', None)
            
            if not self.api_key or not self.api_secret:
                raise ValueError("Alpaca API credentials not found. Please provide valid API key and secret.")
        
        # Get websocket specific config
        websocket_config = getattr(config, 'websocket', {})
        self.enabled = websocket_config.get('enabled', True)
        self.market_data_enabled = websocket_config.get('market_data_enabled', True)
        self.account_updates_enabled = websocket_config.get('account_updates_enabled', True)
        self.reconnect_interval = websocket_config.get('reconnect_interval_seconds', 30)
        self.heartbeat_interval = websocket_config.get('heartbeat_interval_seconds', 30)
        self.max_retries = websocket_config.get('max_retries', 5)
        self.retry_delay = websocket_config.get('retry_delay_seconds', 2)
        
        # Market data websocket
        self.market_ws: Optional[WebSocketApp] = None
        self.market_ws_thread: Optional[threading.Thread] = None
        self.market_connected = False
        self.market_authenticated = False
        self.market_connection_attempts = 0
        
        # Account websocket
        self.account_ws: Optional[WebSocketApp] = None
        self.account_ws_thread: Optional[threading.Thread] = None
        self.account_connected = False
        self.account_authenticated = False
        self.account_connection_attempts = 0
        
        # Heartbeat monitoring
        self.last_market_heartbeat = time.time()
        self.last_account_heartbeat = time.time()
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Running flag
        self.running = False
        self.should_reconnect = True
        
        # Subscriptions for market data
        self.subscriptions = {
            "trades": set(),
            "quotes": set(),
            "bars": set(),
        }
        
        # Cache TTL (in seconds)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 60)
        
        # Message deduplication
        self.processed_messages = set()
        self.processed_messages_max_size = 10000
        
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
            RecordType.BAR: [],
            'account_updates': []
        }
        self.last_batch_insert = time.time()
        self.batch_interval = websocket_config.get('batch_interval_seconds', 5)  # How often to insert batches
        
        # Performance tracking
        self.message_count = 0
        self.message_count_by_type = {
            'trade': 0,     # Market data trades
            'quote': 0,     # Market data quotes
            'bar': 0,       # Market data bars 
            'trade_update': 0,  # Account trade updates
            'account': 0    # Other account messages
        }
        self.start_time = None
        
        logger.info(f"Initialized Alpaca Websocket Collector with API key: {self.api_key[:4]}...{self.api_key[-4:]}")
    
    def _authenticate(self) -> None:
        """
        Authenticate with both Alpaca websocket APIs.
        Required by BaseCollector.
        """
        self._authenticate_market_data()
        self._authenticate_account()
    
    def _authenticate_market_data(self) -> None:
        """
        Authenticate with Alpaca market data websocket API.
        """
        if not self.market_ws or not self.market_connected:
            logger.error("Cannot authenticate: market data websocket not connected")
            return
        
        auth_message = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }
        self.market_ws.send(json.dumps(auth_message))
        logger.info("Sent authentication message to Alpaca market data websocket")
    
    def _authenticate_account(self) -> None:
        """
        Authenticate with Alpaca account websocket API.
        """
        if not self.account_ws or not self.account_connected:
            logger.error("Cannot authenticate: account websocket not connected")
            return
        
        auth_message = {
            "action": "authenticate",
            "data": {
                "key_id": self.api_key,
                "secret_key": self.api_secret
            }
        }
        self.account_ws.send(json.dumps(auth_message))
        logger.info("Sent authentication message to Alpaca account websocket")
    
    def _request_page(self, page_token: Optional[str]) -> Any:
        """
        Not used for websocket collector, but required by BaseCollector.
        """
        raise NotImplementedError("Websocket collector does not support paged requests")

    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Not used for websocket collector, but required by BaseCollector.
        """
        raise NotImplementedError("Websocket collector does not support paged requests")
    
    def _on_market_open(self, ws: WebSocketApp) -> None:
        """
        Handle market data websocket open event.
        
        Args:
            ws: WebSocketApp instance
        """
        logger.info("Market data WebSocket connection established")
        self.market_connected = True
        self.status = "market_connected"
        
        # Reset performance tracking
        self.message_count = 0
        self.message_count_by_type = {k: 0 for k in self.message_count_by_type}
        self.start_time = time.time()
        
        # Authenticate
        self._authenticate_market_data()
    
    def _on_market_message(self, ws: WebSocketApp, message: str) -> None:
        """
        Handle incoming market data websocket messages.
        
        Args:
            ws: WebSocketApp instance
            message: Message received from websocket
        """
        try:
            # Update heartbeat time
            self.last_market_heartbeat = time.time()
            
            # Parse message
            data = json.loads(message)
            
            # Check for duplicate messages
            try:
                message_hash = hash(json.dumps(data, sort_keys=True))
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
            
            # Update message count for performance tracking
            self.message_count += 1
            
            # Handle authentication response
            if data.get('T') == 'success' and data.get('msg') == 'authenticated':
                logger.info("Market data WebSocket authenticated successfully")
                self.market_authenticated = True
                self.status = "market_authenticated"
                self.market_connection_attempts = 0  # Reset connection attempts on successful auth
                
                # Resubscribe to channels
                self._resubscribe_market_data()
                return
            
            # Handle different data types
            msg_type = data.get('T')
            
            # Update message type count
            if msg_type == 't':
                self.message_count_by_type['trade'] += 1
            elif msg_type == 'q':
                self.message_count_by_type['quote'] += 1
            elif msg_type == 'b':
                self.message_count_by_type['bar'] += 1
            
            # Log periodic performance metrics
            if self.message_count % 10000 == 0 and self.start_time:
                elapsed = time.time() - self.start_time
                msgs_per_sec = self.message_count / elapsed if elapsed > 0 else 0
                self.logger.info(f"Processed {self.message_count} messages in {elapsed:.2f}s ({msgs_per_sec:.2f} msgs/sec)")
                self.logger.info(f"Message types: {self.message_count_by_type}")
            
            if msg_type == 'subscription':
                logger.info(f"Market data subscription confirmation: {data}")
            elif msg_type == 'error':
                logger.error(f"Market data WebSocket error: {data}")
            elif msg_type == 't':  # Trade
                self._process_trade(data)
            elif msg_type == 'q':  # Quote
                self._process_quote(data)
            elif msg_type == 'b':  # Bar/Aggregate
                self._process_bar(data)
            else:
                logger.debug(f"Received unknown market data message type: {msg_type}")
            
            # Check if we need to perform a batch insert
            current_time = time.time()
            if self.batch_inserts and (current_time - self.last_batch_insert) > self.batch_interval:
                self._perform_batch_inserts()
                self.last_batch_insert = current_time
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse market data message: {message}")
        except Exception as e:
            logger.error(f"Error processing market data message: {str(e)}")
            
            # Log the raw message for debugging
            try:
                truncated_message = message[:500] + "..." if len(message) > 500 else message
                logger.debug(f"Raw message that caused error: {truncated_message}")
            except Exception:
                pass
    
    def _on_market_error(self, ws: WebSocketApp, error: Exception) -> None:
        """
        Handle market data websocket errors.
        
        Args:
            ws: WebSocketApp instance
            error: Error that occurred
        """
        logger.error(f"Market data WebSocket error: {str(error)}")
        self.status = "market_error"
    
    def _on_market_close(self, ws: WebSocketApp, close_status_code: Optional[int], close_msg: Optional[str]) -> None:
        """
        Handle market data websocket close events.
        
        Args:
            ws: WebSocketApp instance
            close_status_code: Status code for close
            close_msg: Close message
        """
        self.market_connected = False
        self.market_authenticated = False
        self.status = "market_disconnected"
        logger.info(f"Market data WebSocket connection closed: {close_status_code} - {close_msg}")
        
        # Check API key validity on unexpected disconnect
        if close_status_code not in [1000, 1001]:  # Normal close codes
            try:
                api_key_manager.validate_api_key('alpaca', {'api_key': self.api_key, 'api_secret': self.api_secret})
            except Exception:
                pass  # Already logged in API key manager
        
        # Perform any queued database inserts
        self._perform_batch_inserts()
        
        # Reconnect if needed
        if self.should_reconnect:
            # Use exponential backoff for reconnect attempts
            delay = min(self.reconnect_interval * (2 ** min(self.market_connection_attempts, 5)), 300)
            self.market_connection_attempts += 1
            logger.info(f"Reconnecting market data in {delay} seconds (attempt {self.market_connection_attempts})...")
            time.sleep(delay)
            self._connect_market_data()
    
    def _on_account_open(self, ws: WebSocketApp) -> None:
        """
        Handle account websocket open event.
        
        Args:
            ws: WebSocketApp instance
        """
        logger.info("Account WebSocket connection established")
        self.account_connected = True
        if self.status != "market_error":
            self.status = "account_connected"
        
        # Authenticate
        self._authenticate_account()
    
    def _on_account_message(self, ws: WebSocketApp, message: str) -> None:
        """
        Handle incoming account websocket messages.
        
        Args:
            ws: WebSocketApp instance
            message: Message received from websocket
        """
        try:
            # Update heartbeat time
            self.last_account_heartbeat = time.time()
            
            # Parse message
            data = json.loads(message)
            
            # Check for duplicate messages
            try:
                message_hash = hash(json.dumps(data, sort_keys=True))
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
            
            # Update message count for performance tracking
            self.message_count += 1
            
            # Handle authentication response
            if data.get('stream') == 'authorization' and data.get('data', {}).get('status') == 'authorized':
                logger.info("Account WebSocket authenticated successfully")
                self.account_authenticated = True
                self.account_connection_attempts = 0  # Reset connection attempts on successful auth
                if self.status != "market_error":
                    self.status = "account_authenticated"
                
                # Subscribe to trade updates
                self._subscribe_to_trade_updates()
                return
            
            # Handle different data types
            stream = data.get('stream')
            
            # Update message type count
            if stream == 'trade_updates':
                self.message_count_by_type['trade_update'] += 1
            else:
                self.message_count_by_type['account'] += 1
            
            if stream == 'trade_updates':
                self._process_trade_update(data.get('data', {}))
            elif stream == 'listening':
                logger.info(f"Account WebSocket listening to streams: {data.get('data', {}).get('streams', [])}")
            else:
                logger.debug(f"Received unknown account message stream: {stream}")
            
            # Check if we need to perform a batch insert
            current_time = time.time()
            if self.batch_inserts and (current_time - self.last_batch_insert) > self.batch_interval:
                self._perform_batch_inserts()
                self.last_batch_insert = current_time
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse account message: {message}")
        except Exception as e:
            logger.error(f"Error processing account message: {str(e)}")
            
            # Log the raw message for debugging
            try:
                truncated_message = message[:500] + "..." if len(message) > 500 else message
                logger.debug(f"Raw message that caused error: {truncated_message}")
            except Exception:
                pass
    
    def _on_account_error(self, ws: WebSocketApp, error: Exception) -> None:
        """
        Handle account websocket errors.
        
        Args:
            ws: WebSocketApp instance
            error: Error that occurred
        """
        logger.error(f"Account WebSocket error: {str(error)}")
        self.status = "account_error"
    
    def _on_account_close(self, ws: WebSocketApp, close_status_code: Optional[int], close_msg: Optional[str]) -> None:
        """
        Handle account websocket close events.
        
        Args:
            ws: WebSocketApp instance
            close_status_code: Status code for close
            close_msg: Close message
        """
        self.account_connected = False
        self.account_authenticated = False
        self.status = "account_disconnected"
        logger.info(f"Account WebSocket connection closed: {close_status_code} - {close_msg}")
        
        # Check API key validity on unexpected disconnect
        if close_status_code not in [1000, 1001]:  # Normal close codes
            try:
                api_key_manager.validate_api_key('alpaca', {'api_key': self.api_key, 'api_secret': self.api_secret})
            except Exception:
                pass  # Already logged in API key manager
        
        # Perform any queued database inserts for account updates
        if 'account_updates' in self.batch_queues and self.batch_queues['account_updates']:
            self._insert_account_updates_batch()
        
        # Reconnect if needed
        if self.should_reconnect and self.account_updates_enabled:
            # Use exponential backoff for reconnect attempts
            delay = min(self.reconnect_interval * (2 ** min(self.account_connection_attempts, 5)), 300)
            self.account_connection_attempts += 1
            logger.info(f"Reconnecting account data in {delay} seconds (attempt {self.account_connection_attempts})...")
            time.sleep(delay)
            self._connect_account()
    
    def _connect_market_data(self) -> None:
        """
        Connect to the Alpaca market data websocket API.
        """
        if not self.market_data_enabled:
            logger.info("Market data websocket is disabled, skipping connection")
            return
            
        if self.market_ws:
            logger.info("Closing existing market data websocket connection")
            try:
                self.market_ws.close()
            except Exception as e:
                logger.warning(f"Error closing existing market data websocket: {str(e)}")
        
        logger.info(f"Connecting to Alpaca market data websocket: {self.MARKET_WS_URL}")
        
        # Create websocket connection
        self.market_ws = websocket.WebSocketApp(
            self.MARKET_WS_URL,
            on_open=self._on_market_open,
            on_message=self._on_market_message,
            on_error=self._on_market_error,
            on_close=self._on_market_close
        )
        
        # Start websocket in a separate thread
        self.market_ws_thread = threading.Thread(target=self.market_ws.run_forever)
        self.market_ws_thread.daemon = True
        self.market_ws_thread.start()
    
    def _connect_account(self) -> None:
        """
        Connect to the Alpaca account websocket API.
        """
        if not self.account_updates_enabled:
            logger.info("Account websocket is disabled, skipping connection")
            return
            
        if self.account_ws:
            logger.info("Closing existing account websocket connection")
            try:
                self.account_ws.close()
            except Exception as e:
                logger.warning(f"Error closing existing account websocket: {str(e)}")
        
        logger.info(f"Connecting to Alpaca account websocket: {self.ACCOUNT_WS_URL}")
        
        # Create websocket connection
        self.account_ws = websocket.WebSocketApp(
            self.ACCOUNT_WS_URL,
            on_open=self._on_account_open,
            on_message=self._on_account_message,
            on_error=self._on_account_error,
            on_close=self._on_account_close
        )
        
        # Start websocket in a separate thread
        self.account_ws_thread = threading.Thread(target=self.account_ws.run_forever)
        self.account_ws_thread.daemon = True
        self.account_ws_thread.start()
    
    def _process_trade(self, message: Dict[str, Any]) -> None:
        """
        Process a trade message.
        
        Args:
            message: Trade message dictionary
        """
        try:
            # Extract trade data
            symbol = message.get('S', '')  # Symbol
            price = message.get('p', 0)    # Price
            size = message.get('s', 0)     # Size
            timestamp_ns = message.get('t', 0)  # Timestamp in nanoseconds
            exchange = message.get('x', '')  # Exchange
            trade_id = message.get('i', '')  # Trade ID
            tape = message.get('z', '')    # Tape
            conditions = message.get('c', [])  # Conditions
            
            # Convert timestamp to datetime (nanoseconds to seconds)
            timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
            
            # Create StandardRecord
            record = StandardRecord(
                symbol=symbol,
                timestamp=timestamp,
                record_type=RecordType.TRADE,
                source='alpaca',
                price=Decimal(str(price)),
                volume=size,
                exchange=exchange,
                trade_id=trade_id,
                tape=tape,
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
                    
            # Store in data cache
            data_cache.add_websocket_data(symbol, 'trades', record.model_dump())
            
            # Store in database (optionally batched)
            if self.store_trades:
                if self.batch_inserts:
                    self.batch_queues[RecordType.TRADE].append(record)
                    if len(self.batch_queues[RecordType.TRADE]) >= self.batch_size:
                        self._insert_trade_batch()
                else:
                    self._store_record_in_db(record)
            
            logger.debug(f"Processed trade for {symbol}: price={price}, size={size}")
        
        except Exception as e:
            logger.error(f"Error processing trade message: {str(e)}")
    
    def _process_quote(self, message: Dict[str, Any]) -> None:
        """
        Process a quote message.
        
        Args:
            message: Quote message dictionary
        """
        try:
            # Extract quote data
            symbol = message.get('S', '')      # Symbol
            bid_price = message.get('bp', 0)   # Bid price
            bid_size = message.get('bs', 0)    # Bid size
            ask_price = message.get('ap', 0)   # Ask price
            ask_size = message.get('as', 0)    # Ask size
            timestamp_ns = message.get('t', 0) # Timestamp in nanoseconds
            exchange = message.get('x', '')    # Exchange
            conditions = message.get('c', [])  # Conditions
            
            # Convert timestamp to datetime (nanoseconds to seconds)
            timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
            
            # Create StandardRecord
            record = StandardRecord(
                symbol=symbol,
                timestamp=timestamp,
                record_type=RecordType.QUOTE,
                source='alpaca',
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
            
            # Store in data cache
            data_cache.add_websocket_data(symbol, 'quotes', record.model_dump())
            
            # Store in database (optionally batched)
            if self.store_quotes:
                if self.batch_inserts:
                    self.batch_queues[RecordType.QUOTE].append(record)
                    if len(self.batch_queues[RecordType.QUOTE]) >= self.batch_size:
                        self._insert_quote_batch()
                else:
                    self._store_record_in_db(record)
            
            logger.debug(f"Processed quote for {symbol}: bid={bid_price}x{bid_size}, ask={ask_price}x{ask_size}")
        
        except Exception as e:
            logger.error(f"Error processing quote message: {str(e)}")
    
    def _process_bar(self, message: Dict[str, Any]) -> None:
        """
        Process a bar/aggregate message.
        
        Args:
            message: Bar message dictionary
        """
        try:
            # Extract bar data
            symbol = message.get('S', '')      # Symbol
            open_price = message.get('o', 0)   # Open
            high_price = message.get('h', 0)   # High
            low_price = message.get('l', 0)    # Low
            close_price = message.get('c', 0)  # Close
            volume = message.get('v', 0)       # Volume
            timestamp_ns = message.get('t', 0) # Timestamp in nanoseconds
            vwap = message.get('vw', 0)        # VWAP
            num_trades = message.get('n', 0)   # Number of trades
            
            # Convert timestamp to datetime (nanoseconds to seconds)
            timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
            
            # Create StandardRecord
            record = StandardRecord(
                symbol=symbol,
                timestamp=timestamp,
                record_type=RecordType.BAR,
                source='alpaca',
                open=Decimal(str(open_price)),
                high=Decimal(str(high_price)),
                low=Decimal(str(low_price)),
                close=Decimal(str(close_price)),
                volume=volume,
                vwap=Decimal(str(vwap)) if vwap else None,
                extended_data={
                    **message,
                    'num_trades': num_trades
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
            
            # Store in data cache
            data_cache.add_websocket_data(symbol, 'bars', record.model_dump())
            
            # Store in database (optionally batched)
            if self.batch_inserts:
                self.batch_queues[RecordType.BAR].append(record)
                if len(self.batch_queues[RecordType.BAR]) >= self.batch_size:
                    self._insert_bar_batch()
            else:
                self._store_record_in_db(record)
            
            logger.debug(f"Processed bar for {symbol}: open={open_price}, high={high_price}, low={low_price}, close={close_price}, volume={volume}")
        
        except Exception as e:
            logger.error(f"Error processing bar message: {str(e)}")
    
    def _process_trade_update(self, message: Dict[str, Any]) -> None:
        """
        Process a trade update message.
        
        Args:
            message: Trade update message dictionary
        """
        try:
            # Extract trade update data
            event_type = message.get('event')
            order = message.get('order', {})
            timestamp = message.get('timestamp')
            
            if not timestamp:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Extract order details
            order_id = order.get('id', '')
            client_order_id = order.get('client_order_id', '')
            symbol = order.get('symbol', '')
            side = order.get('side', '')
            quantity = order.get('qty', 0)
            filled_quantity = order.get('filled_qty', 0)
            price = order.get('limit_price') or order.get('filled_avg_price', 0)
            status = order.get('status', '')
            
            # Create analytics record for monitoring
            analytics_record = {
                'time': timestamp,
                'metric_name': f'order_{event_type}',
                'metric_value': float(filled_quantity) if filled_quantity else 0.0,
                'symbol': symbol,
                'strategy_id': client_order_id[:10] if client_order_id else '',
                'dimension': side,
                'metadata': {
                    'order_id': order_id,
                    'client_order_id': client_order_id,
                    'status': status,
                    'event_type': event_type,
                    'price': float(price) if price else 0.0,
                    'quantity': float(quantity),
                    'filled_quantity': float(filled_quantity),
                    'side': side,
                    'full_message': message
                }
            }
            
            # Add to batch queue or store directly
            if self.batch_inserts:
                self.batch_queues['account_updates'].append(analytics_record)
                if len(self.batch_queues['account_updates']) >= self.batch_size:
                    self._insert_account_updates_batch()
            else:
                # Store in database
                db_client.insert_analytics_data([analytics_record])
            
            # Store in Redis cache using data_cache
            order_data = {
                'order_id': order_id,
                'client_order_id': client_order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'filled_quantity': filled_quantity,
                'price': price,
                'status': status,
                'event_type': event_type,
                'timestamp': timestamp.isoformat(),
                'source': 'alpaca',
                'full_message': message
            }
            
            # Use timestamp for sorted set score
            timestamp_epoch = timestamp.timestamp()
            data_cache.add_order_update(symbol, order_data, timestamp_epoch)
            
            logger.info(f"Processed trade update for {symbol}: {event_type}, status={status}, filled={filled_quantity}/{quantity}")
        
        except Exception as e:
            logger.error(f"Error processing trade update message: {str(e)}")
    
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
                elif record_type == RecordType.BAR:
                    self._insert_bar_batch()
                elif record_type == 'account_updates':
                    self._insert_account_updates_batch()
    
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
                    'side': 'unknown',  # Alpaca market data doesn't provide trade side 
                    'quantity': record_dict.get('volume', 0),
                    'price': record_dict.get('price', 0),
                    'exchange': record_dict.get('exchange'),
                    'source': record_dict.get('source'),
                    'data_type': 'trades',
                    'record_type': 'trade',
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
                        'source': record_dict.get('source'),
                        'data_type': 'quotes',
                        'record_type': 'quote',
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
    
    def _insert_bar_batch(self) -> None:
        """Insert a batch of bar records into the database."""
        if not self.batch_queues[RecordType.BAR]:
            return
            
        try:
            # Convert records to database format
            db_records = []
            for record in self.batch_queues[RecordType.BAR]:
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
                    'record_type': 'bar',
                    'metadata': record_dict.get('extended_data')
                }
                db_records.append(db_record)
            
            # Insert batch
            if db_records:
                db_client.insert_market_data(db_records)
                self.logger.debug(f"Inserted {len(db_records)} bar records")
            
            # Clear queue
            self.batch_queues[RecordType.BAR] = []
        except Exception as e:
            self.logger.error(f"Error inserting bar batch: {str(e)}")
    
    def _insert_account_updates_batch(self) -> None:
        """Insert a batch of account update records into the database."""
        if not self.batch_queues['account_updates']:
            return
            
        try:
            # Insert batch
            db_client.insert_analytics_data(self.batch_queues['account_updates'])
            self.logger.debug(f"Inserted {len(self.batch_queues['account_updates'])} account update records")
            
            # Clear queue
            self.batch_queues['account_updates'] = []
        except Exception as e:
            self.logger.error(f"Error inserting account updates batch: {str(e)}")
    
    def _store_record_in_db(self, record: StandardRecord) -> None:
        """
        Store a record in the database.
        
        Args:
            record: StandardRecord to store
        """
        try:
            # Convert record to database format
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
                    'num_trades': record_dict.get('extended_data', {}).get('num_trades'),
                    'source': record_dict.get('source'),
                    'data_type': 'bars',
                    'record_type': record.record_type.value,
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
                    'side': 'unknown',  # Alpaca doesn't provide trade side in market data
                    'quantity': record_dict.get('volume', 0),
                    'price': record_dict.get('price', 0),
                    'exchange': record_dict.get('exchange'),
                    'source': record_dict.get('source'),
                    'data_type': 'trades',
                    'record_type': 'trade',
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
                        'source': record_dict.get('source'),
                        'data_type': 'quotes',
                        'record_type': 'quote',
                        **record_dict.get('extended_data', {})
                    }
                }
                db_client.insert_analytics_data([db_record])
        
        except Exception as e:
            self.logger.error(f"Error storing record in database: {str(e)}")
    
    def _check_heartbeats(self) -> None:
        """
        Check if heartbeats are being received and reconnect if needed.
        """
        while self.running:
            try:
                current_time = time.time()
                
                # Check market data heartbeat
                if self.market_data_enabled and current_time - self.last_market_heartbeat > self.heartbeat_interval * 2:
                    logger.warning(f"No market data heartbeat received for {current_time - self.last_market_heartbeat:.1f} seconds, reconnecting...")
                    self._connect_market_data()
                    self.last_market_heartbeat = current_time
                
                # Check account heartbeat
                if self.account_updates_enabled and current_time - self.last_account_heartbeat > self.heartbeat_interval * 2:
                    logger.warning(f"No account heartbeat received for {current_time - self.last_account_heartbeat:.1f} seconds, reconnecting...")
                    self._connect_account()
                    self.last_account_heartbeat = current_time
                
                # Check if we need to perform a batch insert
                if self.batch_inserts and (current_time - self.last_batch_insert) > self.batch_interval:
                    self._perform_batch_inserts()
                    self.last_batch_insert = current_time
                
                # Sleep for a portion of the heartbeat interval
                time.sleep(self.heartbeat_interval / 2)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat checker: {str(e)}")
                time.sleep(self.heartbeat_interval / 2)
    
    def _subscribe_to_trade_updates(self) -> None:
        """
        Subscribe to trade updates from the account websocket.
        """
        if not self.account_ws or not self.account_connected or not self.account_authenticated:
            logger.warning("Not connected or authenticated to account websocket. Cannot subscribe to trade updates.")
            return
        
        subscription_message = {
            "action": "listen",
            "data": {
                "streams": ["trade_updates"]
            }
        }
        
        self.account_ws.send(json.dumps(subscription_message))
        logger.info("Subscribed to trade updates")
    
    def _resubscribe_market_data(self) -> None:
        """
        Resubscribe to all market data channels.
        """
        if not self.market_ws or not self.market_connected or not self.market_authenticated:
            logger.warning("Not connected or authenticated to market data websocket. Cannot resubscribe.")
            return
        
        if all(len(symbols) == 0 for symbols in self.subscriptions.values()):
            logger.warning("No market data subscriptions to restore")
            return
        
        # Create subscription message
        subscription_message = {
            "action": "subscribe"
        }
        
        # Add requested channels
        for channel in self.subscriptions:
            symbols = list(self.subscriptions[channel])
            if symbols:
                subscription_message[channel] = symbols
        
        self.market_ws.send(json.dumps(subscription_message))
        logger.info(f"Resubscribed to market data: {subscription_message}")
    
    @retry_on_rate_limit(max_retries=3)
    def subscribe(self, symbols: List[str], channels: List[str]) -> None:
        """
        Subscribe to market data channels for the given symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            channels: List of channels (trades, quotes, bars)
        """
        if not self.market_ws or not self.market_connected or not self.market_authenticated:
            logger.warning("Not connected or authenticated to market data websocket. Cannot subscribe.")
            return
        
        # Validate channels
        valid_channels = ["trades", "quotes", "bars"]
        for channel in channels:
            if channel not in valid_channels:
                logger.warning(f"Invalid channel: {channel}. Valid options are: {valid_channels}")
                return
        
        # Uppercase symbols
        symbols = [s.upper() for s in symbols]
        
        # Batch subscriptions to avoid large messages
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            symbol_batch = symbols[i:i+batch_size]
            
            # Send subscription message
            subscription_message = {
                "action": "subscribe"
            }
            
            # Add requested channels
            for channel in channels:
                subscription_message[channel] = symbol_batch
                # Update subscription tracking
                self.subscriptions[channel].update(symbol_batch)
            
            self.market_ws.send(json.dumps(subscription_message))
            logger.info(f"Subscribed to {channels} for {len(symbol_batch)} symbols")
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(symbols):
                time.sleep(0.5)
    
    def unsubscribe(self, symbols: List[str], channels: List[str]) -> None:
        """
        Unsubscribe from market data channels for the given symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            channels: List of channels (trades, quotes, bars)
        """
        if not self.market_ws or not self.market_connected or not self.market_authenticated:
            logger.warning("Not connected or authenticated to market data websocket. Cannot unsubscribe.")
            return
        
        # Validate channels
        valid_channels = ["trades", "quotes", "bars"]
        for channel in channels:
            if channel not in valid_channels:
                logger.warning(f"Invalid channel: {channel}. Valid options are: {valid_channels}")
                return
        
        # Uppercase symbols
        symbols = [s.upper() for s in symbols]
        
        # Batch unsubscriptions to avoid large messages
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            symbol_batch = symbols[i:i+batch_size]
            
            # Send unsubscription message
            unsubscription_message = {
                "action": "unsubscribe"
            }
            
            # Add requested channels
            for channel in channels:
                unsubscription_message[channel] = symbol_batch
                # Update subscription tracking
                for symbol in symbol_batch:
                    self.subscriptions[channel].discard(symbol)
            
            self.market_ws.send(json.dumps(unsubscription_message))
            logger.info(f"Unsubscribed from {channels} for {len(symbol_batch)} symbols")
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(symbols):
                time.sleep(0.5)
    
    @retry_on_rate_limit(max_retries=3)
    def start(self) -> None:
        """
        Start the websocket collector.
        """
        if not self.enabled:
            logger.warning("Websocket collector is disabled in configuration")
            return
        
        logger.info("Starting Alpaca websocket collector")
        
        # Set running flag
        self.running = True
        self.should_reconnect = True
        
        # Initialize performance tracking
        self.message_count = 0
        self.message_count_by_type = {k: 0 for k in self.message_count_by_type}
        self.start_time = time.time()
        self.last_batch_insert = time.time()
        
        # Connect to websockets
        self._connect_market_data()
        self._connect_account()
        
        # Start heartbeat checker
        self.heartbeat_thread = threading.Thread(target=self._check_heartbeats)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        logger.info("Alpaca websocket collector started")
    
    def stop(self) -> None:
        """
        Stop the websocket collector.
        """
        logger.info("Stopping Alpaca websocket collector")
        
        # Set flags
        self.running = False
        self.should_reconnect = False
        
        # Perform any pending batch inserts
        try:
            self._perform_batch_inserts()
        except Exception as e:
            logger.error(f"Error performing final batch inserts: {str(e)}")
        
        # Close websockets
        if self.market_ws:
            try:
                self.market_ws.close()
            except Exception as e:
                logger.warning(f"Error closing market data websocket: {str(e)}")
        
        if self.account_ws:
            try:
                self.account_ws.close()
            except Exception as e:
                logger.warning(f"Error closing account websocket: {str(e)}")
        
        # Wait for threads to finish
        if self.market_ws_thread and self.market_ws_thread.is_alive():
            self.market_ws_thread.join(timeout=5)
        
        if self.account_ws_thread and self.account_ws_thread.is_alive():
            self.account_ws_thread.join(timeout=5)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        # Log performance statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            msgs_per_sec = self.message_count / elapsed if elapsed > 0 else 0
            logger.info(f"Processed {self.message_count} messages in {elapsed:.2f}s ({msgs_per_sec:.2f} msgs/sec)")
            logger.info(f"Message breakdown: {self.message_count_by_type}")
        
        self.status = "stopped"
        logger.info("Alpaca websocket collector stopped")
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to all subscription channels.
        
        Args:
            symbol: Symbol to add
        """
        symbol = symbol.upper()
        
        # Check if already subscribed to any channel
        if any(symbol in channel_symbols for channel_symbols in self.subscriptions.values()):
            logger.debug(f"Symbol {symbol} already in some subscription channels")
        
        # Subscribe to all enabled channels
        channels_to_subscribe = []
        
        for channel in self.subscriptions:
            if symbol not in self.subscriptions[channel]:
                channels_to_subscribe.append(channel)
                self.subscriptions[channel].add(symbol)
        
        if channels_to_subscribe and self.market_connected and self.market_authenticated:
            self.subscribe([symbol], channels_to_subscribe)
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from all subscription channels.
        
        Args:
            symbol: Symbol to remove
        """
        symbol = symbol.upper()
        
        # Check if subscribed to any channel
        if not any(symbol in channel_symbols for channel_symbols in self.subscriptions.values()):
            logger.debug(f"Symbol {symbol} not in any subscription channel")
            return
        
        # Unsubscribe from all channels
        channels_to_unsubscribe = []
        
        for channel in self.subscriptions:
            if symbol in self.subscriptions[channel]:
                channels_to_unsubscribe.append(channel)
        
        if channels_to_unsubscribe and self.market_connected and self.market_authenticated:
            self.unsubscribe([symbol], channels_to_unsubscribe)
    
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
        key = f"alpaca:latest:{type_str}:{symbol}"
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
        key = f"alpaca:{type_str}:{symbol}:recent"
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
    
    def get_recent_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent orders.
        
        Args:
            limit: Maximum number of orders to return
            
        Returns:
            List of recent orders
        """
        # Get from Redis cache using DataCache
        orders = data_cache.get_recent_orders(limit)
        
        # Fall back to legacy Redis cache if needed
        if not orders:
            key = "alpaca:recent_orders"
            orders = redis_client.lrange(key, 0, limit - 1)
        
        return orders or []
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the collector.
        
        Returns:
            Status dictionary
        """
        status_dict = {
            "enabled": self.enabled,
            "running": self.running,
            "market_data_enabled": self.market_data_enabled,
            "account_updates_enabled": self.account_updates_enabled,
            "market_connected": self.market_connected,
            "market_authenticated": self.market_authenticated,
            "account_connected": self.account_connected,
            "account_authenticated": self.account_authenticated,
            "status": self.status,
            "subscriptions": {k: len(v) for k, v in self.subscriptions.items()},
            "message_count": self.message_count,
            "message_types": self.message_count_by_type,
            "batch_queues": {
                "trades": len(self.batch_queues[RecordType.TRADE]),
                "quotes": len(self.batch_queues[RecordType.QUOTE]),
                "bars": len(self.batch_queues[RecordType.BAR]),
                "account_updates": len(self.batch_queues['account_updates'])
            }
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
        active_symbols = set()
        for channel, symbols in self.subscriptions.items():
            active_symbols.update(symbols)
        return list(active_symbols)


# Factory function for creating an instance
def create_alpaca_websocket_collector(config: Optional[CollectorConfig] = None) -> AlpacaWebsocketCollector:
    """Create an instance of the Alpaca websocket collector."""
    return AlpacaWebsocketCollector(config)


# For backwards compatibility and testing
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    
    # Create collector
    collector = AlpacaWebsocketCollector()
    
    # Start collector
    collector.start()
    
    # Add some symbols
    collector.add_symbol("AAPL")
    collector.add_symbol("MSFT")
    collector.add_symbol("AMZN")
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop collector on keyboard interrupt
        collector.stop()