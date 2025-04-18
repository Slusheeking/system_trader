#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Order Router
-----------
Routes orders to appropriate brokers with intelligent order execution strategies.
Optimizes trade execution based on market conditions and order characteristics.
"""

import logging
import time
import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import concurrent.futures

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from utils.metrics import calculate_metrics
from trading.execution.circuit_breaker import CircuitBreaker
from trading.execution.execution_monitor import ExecutionMonitor

# Setup logging
logger = setup_logger('order_router')


class OrderType(Enum):
    """
    Types of orders.
    """
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'


class OrderSide(Enum):
    """
    Order sides (buy/sell).
    """
    BUY = 'buy'
    SELL = 'sell'


class OrderStatus(Enum):
    """
    Order statuses.
    """
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


class TimeInForce(Enum):
    """
    Time in force options.
    """
    DAY = 'day'
    GTC = 'gtc'  # Good Till Canceled
    IOC = 'ioc'  # Immediate Or Cancel
    FOK = 'fok'  # Fill Or Kill


class OrderRequest:
    """
    Represents an order request.
    """
    
    def __init__(self, symbol: str, side: OrderSide, quantity: float, order_type: OrderType,
               price: Optional[float] = None, stop_price: Optional[float] = None,
               time_in_force: TimeInForce = TimeInForce.DAY,
               client_order_id: Optional[str] = None):
        """
        Initialize an order request.
        
        Args:
            symbol: Symbol to trade
            side: Order side (buy/sell)
            quantity: Quantity to trade
            order_type: Type of order
            price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            time_in_force: Time in force
            client_order_id: Client-specified order ID
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.client_order_id = client_order_id or str(uuid.uuid4())
        self.timestamp = datetime.now()
        
        # Validate the order
        self._validate()
    
    def _validate(self):
        """
        Validate the order request.
        
        Raises:
            ValueError: If the order is invalid
        """
        # Check required fields
        if not self.symbol:
            raise ValueError("Symbol is required")
        
        if not isinstance(self.side, OrderSide):
            raise ValueError(f"Invalid order side: {self.side}")
        
        if not isinstance(self.order_type, OrderType):
            raise ValueError(f"Invalid order type: {self.order_type}")
        
        if not isinstance(self.time_in_force, TimeInForce):
            raise ValueError(f"Invalid time in force: {self.time_in_force}")
        
        # Check quantity
        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}")
        
        # Check price for limit orders
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError("Price is required for limit orders")
        
        # Check stop price for stop orders
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and self.stop_price is None:
            raise ValueError("Stop price is required for stop orders")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the order request
        """
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'client_order_id': self.client_order_id,
            'timestamp': self.timestamp.isoformat(),
        }


class Order:
    """
    Represents an order with status and execution details.
    """
    
    def __init__(self, order_request: OrderRequest):
        """
        Initialize an order from an order request.
        
        Args:
            order_request: Order request
        """
        self.request = order_request
        self.order_id = None
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.average_price = None
        self.fees = 0.0
        self.rejected_reason = None
        self.last_updated = datetime.now()
        self.fills = []
        self.broker = None
    
    def update(self, status: OrderStatus, filled_quantity: Optional[float] = None,
             average_price: Optional[float] = None, fees: Optional[float] = None,
             rejected_reason: Optional[str] = None, order_id: Optional[str] = None):
        """
        Update order status and execution details.
        
        Args:
            status: New order status
            filled_quantity: Updated filled quantity
            average_price: Updated average fill price
            fees: Updated fees
            rejected_reason: Reason for rejection (if status is REJECTED)
            order_id: Broker-assigned order ID
        """
        self.status = status
        
        if filled_quantity is not None:
            self.filled_quantity = filled_quantity
        
        if average_price is not None:
            self.average_price = average_price
        
        if fees is not None:
            self.fees = fees
        
        if rejected_reason is not None:
            self.rejected_reason = rejected_reason
        
        if order_id is not None:
            self.order_id = order_id
        
        self.last_updated = datetime.now()
    
    def add_fill(self, quantity: float, price: float, timestamp: datetime):
        """
        Add a fill to the order.
        
        Args:
            quantity: Filled quantity
            price: Fill price
            timestamp: Fill timestamp
        """
        self.fills.append({
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp
        })
        
        # Update filled quantity and average price
        total_quantity = sum(fill['quantity'] for fill in self.fills)
        total_value = sum(fill['quantity'] * fill['price'] for fill in self.fills)
        
        self.filled_quantity = total_quantity
        self.average_price = total_value / total_quantity if total_quantity > 0 else None
        
        # Update status
        if self.filled_quantity >= self.request.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the order
        """
        return {
            'order_id': self.order_id,
            'client_order_id': self.request.client_order_id,
            'symbol': self.request.symbol,
            'side': self.request.side.value,
            'quantity': self.request.quantity,
            'order_type': self.request.order_type.value,
            'price': self.request.price,
            'stop_price': self.request.stop_price,
            'time_in_force': self.request.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'fees': self.fees,
            'rejected_reason': self.rejected_reason,
            'last_updated': self.last_updated.isoformat(),
            'broker': self.broker,
            'fills': [
                {
                    'quantity': fill['quantity'],
                    'price': fill['price'],
                    'timestamp': fill['timestamp'].isoformat(),
                }
                for fill in self.fills
            ]
        }


class BrokerConfig:
    """
    Configuration for a broker.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker configuration.
        
        Args:
            config: Broker configuration dictionary
        """
        self.name = config.get('name', 'unknown')
        self.enabled = config.get('enabled', True)
        self.priority = config.get('priority', 0)
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = config.get('base_url')
        self.sandbox = config.get('sandbox', False)
        self.timeout = config.get('timeout', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.max_concurrent_orders = config.get('max_concurrent_orders', 10)
        self.markets = config.get('markets', [])  # Supported markets
        self.fee_structure = config.get('fee_structure', {})


class BaseBroker:
    """
    Base class for broker implementations.
    """
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize the broker.
        
        Args:
            config: Broker configuration
        """
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        
        # Tracking for orders
        self.orders = {}
        
        # Performance metrics
        self.metrics = {
            'order_count': 0,
            'fill_count': 0,
            'rejected_count': 0,
            'avg_execution_time': 0,
            'total_fees': 0,
        }
        
        # Initialize API client
        self._init_client()
        
        logger.info(f"Initialized broker: {self.name}")
    
    def _init_client(self):
        """
        Initialize the broker API client.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _init_client")
    
    def submit_order(self, order_request: OrderRequest) -> Order:
        """
        Submit an order to the broker.
        Must be implemented by subclasses.
        
        Args:
            order_request: Order request
            
        Returns:
            Order object
        """
        raise NotImplementedError("Subclasses must implement submit_order")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        Must be implemented by subclasses.
        
        Args:
            order_id: Order ID
            
        Returns:
            Boolean indicating success
        """
        raise NotImplementedError("Subclasses must implement cancel_order")
    
    def get_order_status(self, order_id: str) -> Order:
        """
        Get order status.
        Must be implemented by subclasses.
        
        Args:
            order_id: Order ID
            
        Returns:
            Updated Order object
        """
        raise NotImplementedError("Subclasses must implement get_order_status")
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances.
        
        Returns:
            Dictionary of asset balances
        """
        raise NotImplementedError("Subclasses must implement get_account_balance")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get broker performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def _track_order(self, order: Order):
        """
        Track an order in the broker's order dictionary.
        
        Args:
            order: Order to track
        """
        # Set broker name
        order.broker = self.name
        
        # Add to orders dictionary
        if order.order_id:
            self.orders[order.order_id] = order
        
        # Update metrics
        self.metrics['order_count'] += 1
        
        if order.status == OrderStatus.REJECTED:
            self.metrics['rejected_count'] += 1
        
        if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            self.metrics['fill_count'] += 1
            self.metrics['total_fees'] += order.fees


class AlpacaBroker(BaseBroker):
    """
    Alpaca Markets broker implementation.
    """
    
    def _init_client(self):
        """
        Initialize Alpaca API client.
        """
        try:
            import alpaca_trade_api as tradeapi
            
            self.client = tradeapi.REST(
                self.config.api_key,
                self.config.api_secret,
                base_url=self.config.base_url or ('https://paper-api.alpaca.markets' if self.config.sandbox else 'https://api.alpaca.markets'),
                api_version='v2'
            )
            
            # Test connection
            account = self.client.get_account()
            logger.info(f"Connected to Alpaca account: {account.id} (Status: {account.status})")
        
        except ImportError:
            logger.error("alpaca-trade-api package not installed")
            self.client = None
        
        except Exception as e:
            logger.error(f"Error initializing Alpaca client: {str(e)}")
            self.client = None
            self.enabled = False
    
    def submit_order(self, order_request: OrderRequest) -> Order:
        """
        Submit an order to Alpaca.
        
        Args:
            order_request: Order request
            
        Returns:
            Order object
        """
        if not self.enabled or self.client is None:
            logger.error("Alpaca broker is not enabled")
            order = Order(order_request)
            order.update(OrderStatus.REJECTED, rejected_reason="Broker not enabled")
            return order
        
        order = Order(order_request)
        
        try:
            # Convert order type
            alpaca_order_type = order_request.order_type.value
            if order_request.order_type == OrderType.TRAILING_STOP:
                alpaca_order_type = 'trailing_stop'
            
            # Submit order to Alpaca
            alpaca_order = self.client.submit_order(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=order_request.side.value,
                type=alpaca_order_type,
                time_in_force=order_request.time_in_force.value,
                limit_price=order_request.price,
                stop_price=order_request.stop_price,
                client_order_id=order_request.client_order_id
            )
            
            # Update order with Alpaca response
            order.update(
                status=OrderStatus[alpaca_order.status.upper()],
                filled_quantity=float(alpaca_order.filled_qty) if hasattr(alpaca_order, 'filled_qty') else 0.0,
                average_price=float(alpaca_order.filled_avg_price) if hasattr(alpaca_order, 'filled_avg_price') and alpaca_order.filled_avg_price else None,
                order_id=alpaca_order.id
            )
            
            # Track order
            self._track_order(order)
            
            logger.info(f"Submitted order to Alpaca: {order.order_id} ({order.request.symbol} {order.request.side.value})")
            
            return order
        
        except Exception as e:
            logger.error(f"Error submitting order to Alpaca: {str(e)}")
            order.update(OrderStatus.REJECTED, rejected_reason=str(e))
            self._track_order(order)
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with Alpaca.
        
        Args:
            order_id: Order ID
            
        Returns:
            Boolean indicating success
        """
        if not self.enabled or self.client is None:
            logger.error("Alpaca broker is not enabled")
            return False
        
        try:
            # Cancel order
            self.client.cancel_order(order_id)
            
            # Update order in tracking dictionary
            if order_id in self.orders:
                self.orders[order_id].update(OrderStatus.CANCELED)
            
            logger.info(f"Canceled Alpaca order: {order_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error canceling Alpaca order {order_id}: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get status of an Alpaca order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Updated Order object or None if not found
        """
        if not self.enabled or self.client is None:
            logger.error("Alpaca broker is not enabled")
            return None
        
        try:
            # Get order from Alpaca
            alpaca_order = self.client.get_order(order_id)
            
            # Check if order exists in tracking dictionary
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found in tracking dictionary")
                return None
            
            # Get order from tracking dictionary
            order = self.orders[order_id]
            
            # Update order with Alpaca response
            order.update(
                status=OrderStatus[alpaca_order.status.upper()],
                filled_quantity=float(alpaca_order.filled_qty) if hasattr(alpaca_order, 'filled_qty') else 0.0,
                average_price=float(alpaca_order.filled_avg_price) if hasattr(alpaca_order, 'filled_avg_price') and alpaca_order.filled_avg_price else None,
            )
            
            return order
        
        except Exception as e:
            logger.error(f"Error getting Alpaca order status for {order_id}: {str(e)}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances from Alpaca.
        
        Returns:
            Dictionary of asset balances
        """
        if not self.enabled or self.client is None:
            logger.error("Alpaca broker is not enabled")
            return {}
        
        try:
            # Get account information
            account = self.client.get_account()
            
            # Get positions
            positions = self.client.list_positions()
            
            # Create balances dictionary
            balances = {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'positions': {}
            }
            
            # Add positions
            for position in positions:
                balances['positions'][position.symbol] = {
                    'quantity': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price)
                }
            
            return balances
        
        except Exception as e:
            logger.error(f"Error getting Alpaca account balance: {str(e)}")
            return {}


class OrderRouter:
    """
    Routes orders to appropriate brokers with intelligent execution strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the order router.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.brokers = self._init_brokers()
        self.circuit_breaker = CircuitBreaker()
        self.execution_monitor = ExecutionMonitor()
        
        # Tracking for orders
        self.orders = {}
        
        # Concurrent execution
        self.max_concurrent_orders = self.config.get('max_concurrent_orders', 10)
        
        logger.info(f"Order Router initialized with {len(self.brokers)} brokers")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config dictionary
        """
        if config_path is None:
            logger.info("No config path provided, using default configuration")
            return {}
        
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _init_brokers(self) -> Dict[str, BaseBroker]:
        """
        Initialize broker instances.
        
        Returns:
            Dictionary of broker instances
        """
        brokers = {}
        
        # Get broker configurations
        broker_configs = self.config.get('brokers', {}) # Default to empty dict

        # Initialize each broker
        for name, broker_data in broker_configs.items():
            if not isinstance(broker_data, dict):
                logger.warning(f"Broker configuration for '{name}' is not a dictionary, skipping")
                continue

            # Add the name to the data dictionary for BrokerConfig
            broker_data['name'] = name

            # Create broker config object
            config = BrokerConfig(broker_data)

            try:
                # Initialize broker based on type (using the name from the key)
                if name.lower() == 'alpaca':
                    broker = AlpacaBroker(config)
                # Add other broker types here if needed
                # elif name.lower() == 'interactive_brokers':
                #     broker = InteractiveBrokersBroker(config) # Example
                else:
                    logger.warning(f"Unknown broker type: {name}")
                    continue

                # Add to brokers dictionary if enabled
                if broker.enabled:
                    brokers[name] = broker
                    logger.info(f"Initialized broker: {name}")
                else:
                    logger.warning(f"Broker {name} is not enabled")

            except Exception as e:
                logger.error(f"Error initializing broker {name}: {str(e)}")
        
        return brokers
    
    def _select_broker(self, order_request: OrderRequest) -> Optional[BaseBroker]:
        """
        Select the best broker for an order.
        
        Args:
            order_request: Order request
            
        Returns:
            Selected broker or None if no suitable broker is found
        """
        if not self.brokers:
            logger.error("No brokers available")
            return None
        
        # If only one broker, use that
        if len(self.brokers) == 1:
            return list(self.brokers.values())[0]
        
        # Score each broker
        broker_scores = {}
        for name, broker in self.brokers.items():
            # Skip disabled brokers
            if not broker.enabled:
                continue
            
            # Start with priority score from config
            score = broker.config.priority
            
            # Check if broker supports the market/symbol
            if broker.config.markets and order_request.symbol.split(':')[0] not in broker.config.markets:
                score -= 1000  # Large penalty for unsupported market
            
            # Consider order type support (not all brokers support all order types)
            # This would need to be implemented based on broker capabilities
            
            # Consider broker performance metrics
            metrics = broker.get_metrics()
            if 'rejected_count' in metrics and metrics['order_count'] > 0:
                rejection_rate = metrics['rejected_count'] / metrics['order_count']
                score -= rejection_rate * 10  # Penalty for high rejection rate
            
            if 'avg_execution_time' in metrics:
                score -= metrics['avg_execution_time'] / 10  # Penalty for slow execution
            
            broker_scores[name] = score
        
        # Select broker with highest score
        if not broker_scores:
            logger.error("No suitable brokers found")
            return None
        
        best_broker_name = max(broker_scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"Selected broker {best_broker_name} for {order_request.symbol}")
        
        return self.brokers[best_broker_name]
    
    def _check_circuit_breaker(self, order_request: OrderRequest) -> bool:
        """
        Check if circuit breaker allows the order.
        
        Args:
            order_request: Order request
            
        Returns:
            Boolean indicating if order is allowed
        """
        # Check global circuit breaker
        if not self.circuit_breaker.allow_trading():
            logger.warning("Global circuit breaker is active, order not allowed")
            return False
        
        # Check symbol-specific circuit breaker
        if not self.circuit_breaker.allow_trading_for_symbol(order_request.symbol):
            logger.warning(f"Circuit breaker for {order_request.symbol} is active, order not allowed")
            return False
        
        return True
    
    def submit_order(self, order_request: OrderRequest) -> Order:
        """
        Submit an order through the router.
        
        Args:
            order_request: Order request
            
        Returns:
            Order object
        """
        logger.info(f"Submitting order for {order_request.symbol} ({order_request.side.value})")
        
        # Check circuit breaker
        if not self._check_circuit_breaker(order_request):
            order = Order(order_request)
            order.update(OrderStatus.REJECTED, rejected_reason="Circuit breaker active")
            return order
        
        # Select broker
        broker = self._select_broker(order_request)
        if broker is None:
            order = Order(order_request)
            order.update(OrderStatus.REJECTED, rejected_reason="No suitable broker found")
            return order
        
        # Submit order to broker
        order = broker.submit_order(order_request)
        
        # Track order
        self.orders[order.request.client_order_id] = order
        
        # Monitor execution
        self.execution_monitor.track_order(order)
        
        return order
    
    def submit_orders(self, order_requests: List[OrderRequest]) -> Dict[str, Order]:
        """
        Submit multiple orders concurrently.
        
        Args:
            order_requests: List of order requests
            
        Returns:
            Dictionary of client order IDs to Order objects
        """
        logger.info(f"Submitting {len(order_requests)} orders")
        
        # Check circuit breaker for all orders
        allowed_requests = []
        rejected_orders = {}
        
        for request in order_requests:
            if self._check_circuit_breaker(request):
                allowed_requests.append(request)
            else:
                order = Order(request)
                order.update(OrderStatus.REJECTED, rejected_reason="Circuit breaker active")
                rejected_orders[request.client_order_id] = order
        
        # Submit allowed orders concurrently
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_orders) as executor:
            # Submit orders
            future_to_order = {
                executor.submit(self.submit_order, request): request.client_order_id
                for request in allowed_requests
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_order):
                client_order_id = future_to_order[future]
                try:
                    order = future.result()
                    results[client_order_id] = order
                except Exception as e:
                    logger.error(f"Error submitting order {client_order_id}: {str(e)}")
                    # Create rejected order
                    request = next(r for r in allowed_requests if r.client_order_id == client_order_id)
                    order = Order(request)
                    order.update(OrderStatus.REJECTED, rejected_reason=str(e))
                    results[client_order_id] = order
        
        # Add rejected orders
        results.update(rejected_orders)
        
        return results
    
    def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Canceling order: {client_order_id}")
        
        # Check if order exists
        if client_order_id not in self.orders:
            logger.warning(f"Order {client_order_id} not found")
            return False
        
        order = self.orders[client_order_id]
        
        # Check if order can be canceled
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            logger.warning(f"Order {client_order_id} cannot be canceled (status: {order.status.value})")
            return False
        
        # Get broker
        broker_name = order.broker
        if broker_name not in self.brokers:
            logger.error(f"Broker {broker_name} not found")
            return False
        
        broker = self.brokers[broker_name]
        
        # Cancel order
        success = broker.cancel_order(order.order_id)
        
        # Update order status if successful
        if success:
            order.update(OrderStatus.CANCELED)
        
        return success
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Tuple[int, int]:
        """
        Cancel all orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            Tuple of (canceled count, failed count)
        """
        logger.info(f"Canceling all orders{f' for {symbol}' if symbol else ''}")
        
        # Filter orders to cancel
        orders_to_cancel = []
        for order in self.orders.values():
            # Skip orders that can't be canceled
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                continue
            
            # Filter by symbol if provided
            if symbol and order.request.symbol != symbol:
                continue
            
            orders_to_cancel.append(order)
        
        canceled_count = 0
        failed_count = 0
        
        # Cancel orders concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit cancellations
            future_to_order = {
                executor.submit(self.cancel_order, order.request.client_order_id): order.request.client_order_id
                for order in orders_to_cancel
            }
            
            # Process results
            for future in concurrent.futures.as_completed(future_to_order):
                client_order_id = future_to_order[future]
                try:
                    success = future.result()
                    if success:
                        canceled_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error canceling order {client_order_id}: {str(e)}")
                    failed_count += 1
        
        logger.info(f"Canceled {canceled_count} orders, {failed_count} failed")
        return canceled_count, failed_count
    
    def get_order_status(self, client_order_id: str) -> Optional[Order]:
        """
        Get order status.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            Order object or None if not found
        """
        # Check if order exists
        if client_order_id not in self.orders:
            logger.warning(f"Order {client_order_id} not found")
            return None
        
        order = self.orders[client_order_id]
        
        # Skip update for final states
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            return order
        
        # Get broker
        broker_name = order.broker
        if broker_name not in self.brokers:
            logger.error(f"Broker {broker_name} not found")
            return order
        
        broker = self.brokers[broker_name]
        
        # Get updated order status
        updated_order = broker.get_order_status(order.order_id)
        
        # Update local order if broker returned an update
        if updated_order:
            order.update(
                status=updated_order.status,
                filled_quantity=updated_order.filled_quantity,
                average_price=updated_order.average_price,
                fees=updated_order.fees,
                rejected_reason=updated_order.rejected_reason
            )
        
        return order
    
    def get_all_orders(self, symbol: Optional[str] = None, 
                     status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get all orders, optionally filtered by symbol and status.
        
        Args:
            symbol: Optional symbol to filter orders
            status: Optional status to filter orders
            
        Returns:
            List of Order objects
        """
        # Filter orders
        filtered_orders = []
        for order in self.orders.values():
            # Filter by symbol if provided
            if symbol and order.request.symbol != symbol:
                continue
            
            # Filter by status if provided
            if status and order.status != status:
                continue
            
            filtered_orders.append(order)
        
        return filtered_orders
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            List of open Order objects
        """
        # Open order statuses
        open_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        
        # Filter orders
        open_orders = []
        for order in self.orders.values():
            # Skip closed orders
            if order.status not in open_statuses:
                continue
            
            # Filter by symbol if provided
            if symbol and order.request.symbol != symbol:
                continue
            
            open_orders.append(order)
        
        return open_orders
    
    def get_account_balances(self) -> Dict[str, Dict[str, Any]]:
        """
        Get account balances from all brokers.
        
        Returns:
            Dictionary of broker name to balances
        """
        balances = {}
        
        for name, broker in self.brokers.items():
            try:
                broker_balances = broker.get_account_balance()
                balances[name] = broker_balances
            except Exception as e:
                logger.error(f"Error getting balances from {name}: {str(e)}")
                balances[name] = {'error': str(e)}
        
        return balances
    
    def get_broker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics from all brokers.
        
        Returns:
            Dictionary of broker name to metrics
        """
        metrics = {}
        
        for name, broker in self.brokers.items():
            try:
                broker_metrics = broker.get_metrics()
                metrics[name] = broker_metrics
            except Exception as e:
                logger.error(f"Error getting metrics from {name}: {str(e)}")
                metrics[name] = {'error': str(e)}
        
        return metrics
    
    def route(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a trading signal to the appropriate broker by building an order and placing it.
        
        Args:
            signal: Dictionary containing signal information
                - symbol: Trading symbol
                - side: Trade direction ('buy' or 'sell')
                - confidence: Signal confidence score (0-1)
                - price: Current price (optional)
                - quantity: Suggested quantity (optional)
                
        Returns:
            Dictionary with order result information
        """
        logger.info(f"Routing signal for {signal.get('symbol')}")
        
        start_time = time.time()
        
        # Extract signal information
        symbol = signal.get('symbol')
        if not symbol:
            logger.error("Signal missing required 'symbol' field")
            return {'status': 'error', 'reason': 'missing_symbol'}
        
        # Determine order side
        side_str = signal.get('side', 'buy').lower()
        side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
        
        # Get quantity from signal or use default
        quantity = signal.get('quantity', 1.0)
        
        # Determine order type and price
        price = signal.get('price')
        confidence = signal.get('confidence', 0.75)
        price_buffer = 0.005 * (1 + confidence)  # 0.5% to 1% buffer based on confidence
        
        # Calculate limit price if we have a price
        if price:
            if isinstance(price, str):
                try:
                    price = float(price)
                except ValueError:
                    price = None
            
            if price and side == OrderSide.BUY:
                # Set limit price slightly above market for buy orders
                price = price * (1 + price_buffer)
            elif price and side == OrderSide.SELL:
                # Set limit price slightly below market for sell orders
                price = price * (1 - price_buffer)
        
        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            price=price,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"{side_str}_{symbol}_{int(time.time())}"
        )
        
        # Submit order
        order = self.submit_order(order_request)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Emit metrics
        metrics = {
            'order_submission_latency': latency,
            'order_count': 1,
            'order_success': 1 if order.status != OrderStatus.REJECTED else 0,
            'order_rejection': 1 if order.status == OrderStatus.REJECTED else 0
        }
        
        # Log metrics
        logger.info(f"Order routing metrics: {metrics}")
        
        # Format result
        result = {
            'order': order.to_dict(),
            'status': 'success' if order.status != OrderStatus.REJECTED else 'rejected',
            'latency': latency
        }
        
        return result
    
    def process_entry_signals(self, signals: Dict[str, Any], risk_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process entry signals from the entry timing model and submit orders.
        
        Args:
            signals: Dictionary with entry signals
            risk_info: Optional risk management information
            
        Returns:
            Dictionary with order results
        """
        logger.info(f"Processing entry signals")
        
        # Check if signals contain entry information
        entry_symbols = signals.get('entry_symbols', [])
        if not entry_symbols:
            logger.warning("No entry symbols in signals")
            return {'orders': {}, 'count': 0, 'status': 'no_entries'}
        
        # Check circuit breaker
        if not self.circuit_breaker.allow_trading():
            logger.warning("Circuit breaker is active, no orders submitted")
            return {'orders': {}, 'count': 0, 'status': 'circuit_breaker_active'}
        
        # Create order requests
        order_requests = []
        
        # Get detailed signal information if available
        detailed_signals = signals.get('signals_by_symbol', {})
        
        for symbol in entry_symbols:
            # Check circuit breaker for symbol
            if not self.circuit_breaker.allow_trading_for_symbol(symbol):
                logger.warning(f"Circuit breaker for {symbol} is active, skipping")
                continue
            
            # Default quantity and price
            quantity = 1.0
            price = None
            
            # Get detailed signal info if available
            signal_info = detailed_signals.get(symbol, [{}])[0] if detailed_signals else {}
            
            # Use confidence as price limit buffer if available
            confidence = signal_info.get('entry_confidence', 0.75)
            price_buffer = 0.005 * (1 + confidence)  # 0.5% to 1% buffer based on confidence
            
            # Get latest price for the symbol
            # In a real system, you'd use a market data provider here
            latest_price = signal_info.get('close', None)
            
            # Calculate limit price if we have the latest price
            if latest_price:
                if isinstance(latest_price, str):
                    try:
                        latest_price = float(latest_price)
                    except ValueError:
                        latest_price = None
                
                if latest_price:
                    # Set limit price slightly above market for buy orders
                    price = latest_price * (1 + price_buffer)
            
            # Get position size from risk info if available
            if risk_info and 'allocations' in risk_info:
                allocations = risk_info.get('allocations', {})
                if symbol in allocations:
                    quantity = allocations[symbol]
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.LIMIT if price else OrderType.MARKET,
                price=price,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"entry_{symbol}_{int(time.time())}"
            )
            
            order_requests.append(order_request)
        
        # Submit orders
        orders = self.submit_orders(order_requests)
        
        # Format result
        result = {
            'orders': {client_id: order.to_dict() for client_id, order in orders.items()},
            'count': len(orders),
            'successful': sum(1 for order in orders.values() if order.status not in [OrderStatus.REJECTED]),
            'rejected': sum(1 for order in orders.values() if order.status == OrderStatus.REJECTED),
            'status': 'orders_submitted'
        }
        
        return result


# Default order router instance
default_order_router = None


def get_order_router(config_path: Optional[str] = None) -> OrderRouter:
    """
    Get or create the default order router.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        OrderRouter instance
    """
    global default_order_router
    
    if default_order_router is None:
        default_order_router = OrderRouter(config_path)
    
    return default_order_router


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Order Router for Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Symbol to trade')
    parser.add_argument('--side', type=str, choices=['buy', 'sell'], help='Order side')
    parser.add_argument('--quantity', type=float, help='Order quantity')
    parser.add_argument('--order-type', type=str, choices=['market', 'limit'], default='market', help='Order type')
    parser.add_argument('--price', type=float, help='Limit price')
    
    args = parser.parse_args()
    
    # Create order router
    order_router = OrderRouter(args.config)
    
    if args.symbol and args.side and args.quantity:
        # Create order request
        order_request = OrderRequest(
            symbol=args.symbol,
            side=OrderSide.BUY if args.side == 'buy' else OrderSide.SELL,
            quantity=args.quantity,
            order_type=OrderType.LIMIT if args.order_type == 'limit' else OrderType.MARKET,
            price=args.price,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order
        order = order_router.submit_order(order_request)
        
        # Print result
        import json
        print(json.dumps(order.to_dict(), indent=2))
    else:
        parser.print_help()
