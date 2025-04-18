#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Circuit Breaker
--------------
Implements safety mechanisms to prevent excessive trading or losses.
Monitors market conditions and trading performance to automatically 
halt trading when predefined risk thresholds are breached.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import threading

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from utils.metrics import calculate_metrics
from monitoring import (
    register_component,
    update_health,
    record_heartbeat,
    record_error,
    record_metric,
    increment_counter
)

# Setup logging
logger = setup_logger('circuit_breaker')


class CircuitBreakerState(Enum):
    """
    Circuit breaker states.
    """
    CLOSED = 'closed'  # Normal operation, trading allowed
    OPEN = 'open'      # Trading halted
    HALF_OPEN = 'half_open'  # Testing if conditions have normalized


class CircuitBreakerType(Enum):
    """
    Types of circuit breakers.
    """
    GLOBAL = 'global'           # Affects all trading
    SYMBOL = 'symbol'           # Affects specific symbol
    MARKET = 'market'           # Affects specific market (e.g., equities, crypto)
    STRATEGY = 'strategy'       # Affects specific strategy
    VOLATILITY = 'volatility'   # Triggered by market volatility
    DRAWDOWN = 'drawdown'       # Triggered by portfolio drawdown
    LOSS = 'loss'               # Triggered by consecutive losses
    TECHNICAL = 'technical'     # Triggered by technical system issues
    LIQUIDITY = 'liquidity'     # Triggered by liquidity concerns
    CUSTOM = 'custom'           # Custom trigger


class CircuitBreakerTrigger:
    """
    A trigger condition for a circuit breaker.
    """
    
    def __init__(self, name: str, condition_fn: callable, reset_fn: Optional[callable] = None,
               trigger_type: CircuitBreakerType = CircuitBreakerType.CUSTOM,
               cooldown_period: int = 300):
        """
        Initialize a circuit breaker trigger.
        
        Args:
            name: Trigger name
            condition_fn: Function that returns True if the trigger condition is met
            reset_fn: Optional function that returns True if the trigger can be reset
            trigger_type: Type of trigger
            cooldown_period: Cooldown period in seconds before testing reset
        """
        self.name = name
        self.condition_fn = condition_fn
        self.reset_fn = reset_fn or (lambda: True)  # Default reset function always allows reset
        self.trigger_type = trigger_type
        self.cooldown_period = cooldown_period
        self.last_triggered = None
        self.trigger_count = 0
        self.is_active = False
    
    def check(self, context: Dict[str, Any] = None) -> bool:
        """
        Check if the trigger condition is met.
        
        Args:
            context: Optional context dictionary
            
        Returns:
            Boolean indicating if the condition is met
        """
        context = context or {}
        try:
            result = self.condition_fn(context)
            if result:
                self.last_triggered = datetime.now()
                self.trigger_count += 1
                self.is_active = True
                logger.info(f"Circuit breaker trigger {self.name} activated")
            return result
        except Exception as e:
            logger.error(f"Error checking circuit breaker trigger {self.name}: {str(e)}")
            return False
    
    def can_reset(self, context: Dict[str, Any] = None) -> bool:
        """
        Check if the trigger can be reset.
        
        Args:
            context: Optional context dictionary
            
        Returns:
            Boolean indicating if the trigger can be reset
        """
        # Check cooldown period
        if self.last_triggered is None:
            return True
        
        if (datetime.now() - self.last_triggered).total_seconds() < self.cooldown_period:
            return False
        
        context = context or {}
        try:
            result = self.reset_fn(context)
            if result:
                self.is_active = False
                logger.info(f"Circuit breaker trigger {self.name} reset")
            return result
        except Exception as e:
            logger.error(f"Error checking circuit breaker reset for {self.name}: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trigger to dictionary.
        
        Returns:
            Dictionary representation of the trigger
        """
        return {
            'name': self.name,
            'type': self.trigger_type.value,
            'is_active': self.is_active,
            'trigger_count': self.trigger_count,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'cooldown_period': self.cooldown_period
        }


class CircuitBreaker:
    """
    Implements safety mechanisms to prevent excessive trading or losses.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the circuit breaker.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.global_state = CircuitBreakerState.CLOSED
        self.symbol_states: Dict[str, CircuitBreakerState] = {}
        self.market_states: Dict[str, CircuitBreakerState] = {}
        self.strategy_states: Dict[str, CircuitBreakerState] = {}
        
        # Initialize triggers
        self.global_triggers: List[CircuitBreakerTrigger] = []
        self.symbol_triggers: Dict[str, List[CircuitBreakerTrigger]] = {}
        self.market_triggers: Dict[str, List[CircuitBreakerTrigger]] = {}
        self.strategy_triggers: Dict[str, List[CircuitBreakerTrigger]] = {}
        
        # Track override status
        self.global_override = False
        self.symbol_overrides: Set[str] = set()
        self.market_overrides: Set[str] = set()
        self.strategy_overrides: Set[str] = set()
        
        # Testing mode
        self.test_mode = self.config.get('test_mode', False)
        
        # Auto-reset thread
        self.auto_reset_enabled = self.config.get('auto_reset', {}).get('enabled', True)
        self.auto_reset_interval = self.config.get('auto_reset', {}).get('interval', 300)  # 5 minutes
        self._stop_auto_reset = threading.Event()
        self._auto_reset_thread = None
        
        # Initialize from config
        self._init_from_config()
        
        # Register with monitoring system
        register_component('circuit_breaker', self)
        # circuit_breaker_triggered is a Counter, starts at 0 automatically.
        # Initialize trading_halted gauge with global scope, starting as not halted (0).
        record_metric('trading', 'trading_halted', 0, labels={'scope': 'global'})

        # Start auto-reset thread if enabled
        if self.auto_reset_enabled:
            self._start_auto_reset_thread()
        
        logger.info("Circuit breaker initialized")
    
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
    
    def _init_from_config(self):
        """
        Initialize triggers from configuration.
        """
        # Initialize global triggers
        global_triggers = self.config.get('global_triggers', [])
        for trigger_config in global_triggers:
            self._add_global_trigger_from_config(trigger_config)
        
        # Initialize symbol triggers
        symbol_triggers = self.config.get('symbol_triggers', {})
        for symbol, triggers in symbol_triggers.items():
            for trigger_config in triggers:
                self._add_symbol_trigger_from_config(symbol, trigger_config)
        
        # Initialize market triggers
        market_triggers = self.config.get('market_triggers', {})
        for market, triggers in market_triggers.items():
            for trigger_config in triggers:
                self._add_market_trigger_from_config(market, trigger_config)
        
        # Initialize strategy triggers
        strategy_triggers = self.config.get('strategy_triggers', {})
        for strategy, triggers in strategy_triggers.items():
            for trigger_config in triggers:
                self._add_strategy_trigger_from_config(strategy, trigger_config)
    
    def _create_trigger_from_config(self, trigger_config: Dict[str, Any]) -> Optional[CircuitBreakerTrigger]:
        """
        Create a trigger from configuration.
        
        Args:
            trigger_config: Trigger configuration dictionary
            
        Returns:
            CircuitBreakerTrigger instance or None if creation fails
        """
        try:
            # Get basic params
            name = trigger_config.get('name')
            trigger_type_str = trigger_config.get('type', 'custom')
            cooldown_period = trigger_config.get('cooldown_period', 300)
            
            # Parse trigger type
            try:
                trigger_type = CircuitBreakerType(trigger_type_str)
            except ValueError:
                trigger_type = CircuitBreakerType.CUSTOM
            
            # Create the appropriate condition function based on type
            if trigger_type == CircuitBreakerType.VOLATILITY:
                # Volatility trigger
                threshold = trigger_config.get('threshold', 0.05)  # 5% by default
                lookback = trigger_config.get('lookback', 10)  # 10 periods
                
                # Volatility condition: True if volatility exceeds threshold
                def volatility_condition(context):
                    data = context.get('price_data')
                    if data is None or len(data) < lookback:
                        return False
                    returns = pd.Series(data).pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized
                    return volatility > threshold
                
                condition_fn = volatility_condition
            
            elif trigger_type == CircuitBreakerType.DRAWDOWN:
                # Drawdown trigger
                threshold = trigger_config.get('threshold', 0.10)  # 10% by default
                
                # Drawdown condition: True if drawdown exceeds threshold
                def drawdown_condition(context):
                    equity = context.get('equity')
                    peak_equity = context.get('peak_equity')
                    if equity is None or peak_equity is None or peak_equity == 0:
                        return False
                    drawdown = 1 - (equity / peak_equity)
                    return drawdown > threshold
                
                condition_fn = drawdown_condition
            
            elif trigger_type == CircuitBreakerType.LOSS:
                # Consecutive loss trigger
                threshold = trigger_config.get('threshold', 3)  # 3 consecutive losses by default
                
                # Loss condition: True if consecutive losses exceed threshold
                def loss_condition(context):
                    consecutive_losses = context.get('consecutive_losses', 0)
                    return consecutive_losses >= threshold
                
                condition_fn = loss_condition
            
            elif trigger_type == CircuitBreakerType.TECHNICAL:
                # Technical issue trigger
                error_threshold = trigger_config.get('error_threshold', 5)  # 5 errors by default
                error_window = trigger_config.get('error_window', 300)  # 5 minutes by default
                
                # Technical condition: True if errors exceed threshold in window
                def technical_condition(context):
                    errors = context.get('errors', [])
                    # Filter errors within window
                    current_time = datetime.now()
                    recent_errors = [e for e in errors if (current_time - e).total_seconds() < error_window]
                    return len(recent_errors) >= error_threshold
                
                condition_fn = technical_condition
            
            elif trigger_type == CircuitBreakerType.LIQUIDITY:
                # Liquidity trigger
                spread_threshold = trigger_config.get('spread_threshold', 0.01)  # 1% by default
                
                # Liquidity condition: True if spread exceeds threshold
                def liquidity_condition(context):
                    spread = context.get('spread', 0)
                    return spread > spread_threshold
                
                condition_fn = liquidity_condition
            
            else:
                # Custom or unknown type
                # Try to get condition function from config
                custom_condition = trigger_config.get('condition')
                if custom_condition:
                    # Parse custom condition (this is simplified and not secure)
                    try:
                        # Extremely simplified parsing - not recommended for production
                        condition_str = f"lambda context: {custom_condition}"
                        condition_fn = eval(condition_str)
                    except Exception as e:
                        logger.error(f"Error parsing custom condition: {str(e)}")
                        return None
                else:
                    # Default condition always returns False
                    condition_fn = lambda context: False
            
            # Create reset function if specified
            reset_fn = None
            custom_reset = trigger_config.get('reset')
            if custom_reset:
                try:
                    # Extremely simplified parsing - not recommended for production
                    reset_str = f"lambda context: {custom_reset}"
                    reset_fn = eval(reset_str)
                except Exception as e:
                    logger.error(f"Error parsing custom reset: {str(e)}")
            
            # Create trigger
            return CircuitBreakerTrigger(
                name=name,
                condition_fn=condition_fn,
                reset_fn=reset_fn,
                trigger_type=trigger_type,
                cooldown_period=cooldown_period
            )
        
        except Exception as e:
            logger.error(f"Error creating trigger from config: {str(e)}")
            return None
    
    def _add_global_trigger_from_config(self, trigger_config: Dict[str, Any]):
        """
        Add a global trigger from configuration.
        
        Args:
            trigger_config: Trigger configuration dictionary
        """
        trigger = self._create_trigger_from_config(trigger_config)
        if trigger:
            self.global_triggers.append(trigger)
            logger.info(f"Added global trigger: {trigger.name}")
    
    def _add_symbol_trigger_from_config(self, symbol: str, trigger_config: Dict[str, Any]):
        """
        Add a symbol trigger from configuration.
        
        Args:
            symbol: Symbol to add trigger for
            trigger_config: Trigger configuration dictionary
        """
        trigger = self._create_trigger_from_config(trigger_config)
        if trigger:
            if symbol not in self.symbol_triggers:
                self.symbol_triggers[symbol] = []
            self.symbol_triggers[symbol].append(trigger)
            logger.info(f"Added trigger for symbol {symbol}: {trigger.name}")
    
    def _add_market_trigger_from_config(self, market: str, trigger_config: Dict[str, Any]):
        """
        Add a market trigger from configuration.
        
        Args:
            market: Market to add trigger for
            trigger_config: Trigger configuration dictionary
        """
        trigger = self._create_trigger_from_config(trigger_config)
        if trigger:
            if market not in self.market_triggers:
                self.market_triggers[market] = []
            self.market_triggers[market].append(trigger)
            logger.info(f"Added trigger for market {market}: {trigger.name}")
    
    def _add_strategy_trigger_from_config(self, strategy: str, trigger_config: Dict[str, Any]):
        """
        Add a strategy trigger from configuration.
        
        Args:
            strategy: Strategy to add trigger for
            trigger_config: Trigger configuration dictionary
        """
        trigger = self._create_trigger_from_config(trigger_config)
        if trigger:
            if strategy not in self.strategy_triggers:
                self.strategy_triggers[strategy] = []
            self.strategy_triggers[strategy].append(trigger)
            logger.info(f"Added trigger for strategy {strategy}: {trigger.name}")
    
    def add_global_trigger(self, trigger: CircuitBreakerTrigger):
        """
        Add a global trigger.
        
        Args:
            trigger: CircuitBreakerTrigger instance
        """
        self.global_triggers.append(trigger)
        logger.info(f"Added global trigger: {trigger.name}")
    
    def add_symbol_trigger(self, symbol: str, trigger: CircuitBreakerTrigger):
        """
        Add a symbol trigger.
        
        Args:
            symbol: Symbol to add trigger for
            trigger: CircuitBreakerTrigger instance
        """
        if symbol not in self.symbol_triggers:
            self.symbol_triggers[symbol] = []
        self.symbol_triggers[symbol].append(trigger)
        logger.info(f"Added trigger for symbol {symbol}: {trigger.name}")
    
    def add_market_trigger(self, market: str, trigger: CircuitBreakerTrigger):
        """
        Add a market trigger.
        
        Args:
            market: Market to add trigger for
            trigger: CircuitBreakerTrigger instance
        """
        if market not in self.market_triggers:
            self.market_triggers[market] = []
        self.market_triggers[market].append(trigger)
        logger.info(f"Added trigger for market {market}: {trigger.name}")
    
    def add_strategy_trigger(self, strategy: str, trigger: CircuitBreakerTrigger):
        """
        Add a strategy trigger.
        
        Args:
            strategy: Strategy to add trigger for
            trigger: CircuitBreakerTrigger instance
        """
        if strategy not in self.strategy_triggers:
            self.strategy_triggers[strategy] = []
        self.strategy_triggers[strategy].append(trigger)
        logger.info(f"Added trigger for strategy {strategy}: {trigger.name}")
    
    def check_global_triggers(self, context: Dict[str, Any] = None) -> bool:
        """
        Check all global triggers.
        
        Args:
            context: Optional context dictionary
            
        Returns:
            Boolean indicating if any global trigger is activated
        """
        # Skip if in override mode
        if self.global_override:
            return False
        
        # Skip if already open
        if self.global_state == CircuitBreakerState.OPEN:
            return True
        
        context = context or {}
        triggered = False
        
        for trigger in self.global_triggers:
            if trigger.check(context):
                triggered = True
                logger.warning(f"Global circuit breaker triggered: {trigger.name}")
                
                # Record in monitoring system
                record_error(
                    component_name='circuit_breaker',
                    error_type='global_trigger',
                    severity='critical',
                    error_message=f"Global circuit breaker triggered: {trigger.name}"
                )
                
                # Increment counter for triggered circuit breakers
                increment_counter('trading', 'circuit_breaker_triggered', {
                    'type': 'global',
                    'trigger': trigger.name
                })
        
        if triggered:
            self.global_state = CircuitBreakerState.OPEN
            logger.warning("Global circuit breaker activated")
            
            # Update trading halted metric
            record_metric('trading', 'trading_halted', 1, {'scope': 'global'})
            
            # Update component health
            update_health('circuit_breaker', 'degraded', "Global circuit breaker activated")
        
        return triggered
    
    def check_symbol_triggers(self, symbol: str, context: Dict[str, Any] = None) -> bool:
        """
        Check triggers for a specific symbol.
        
        Args:
            symbol: Symbol to check triggers for
            context: Optional context dictionary
            
        Returns:
            Boolean indicating if any symbol trigger is activated
        """
        # Skip if global circuit breaker is open
        if self.global_state == CircuitBreakerState.OPEN:
            return True
        
        # Skip if in override mode
        if symbol in self.symbol_overrides:
            return False
        
        # Skip if already open
        if self.symbol_states.get(symbol) == CircuitBreakerState.OPEN:
            return True
        
        # Skip if no triggers for this symbol
        if symbol not in self.symbol_triggers:
            return False
        
        context = context or {}
        context['symbol'] = symbol
        triggered = False
        
        for trigger in self.symbol_triggers[symbol]:
            if trigger.check(context):
                triggered = True
                logger.warning(f"Circuit breaker triggered for symbol {symbol}: {trigger.name}")
        
        if triggered:
            self.symbol_states[symbol] = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker activated for symbol {symbol}")
        
        return triggered
    
    def check_market_triggers(self, market: str, context: Dict[str, Any] = None) -> bool:
        """
        Check triggers for a specific market.
        
        Args:
            market: Market to check triggers for
            context: Optional context dictionary
            
        Returns:
            Boolean indicating if any market trigger is activated
        """
        # Skip if global circuit breaker is open
        if self.global_state == CircuitBreakerState.OPEN:
            return True
        
        # Skip if in override mode
        if market in self.market_overrides:
            return False
        
        # Skip if already open
        if self.market_states.get(market) == CircuitBreakerState.OPEN:
            return True
        
        # Skip if no triggers for this market
        if market not in self.market_triggers:
            return False
        
        context = context or {}
        context['market'] = market
        triggered = False
        
        for trigger in self.market_triggers[market]:
            if trigger.check(context):
                triggered = True
                logger.warning(f"Circuit breaker triggered for market {market}: {trigger.name}")
        
        if triggered:
            self.market_states[market] = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker activated for market {market}")
        
        return triggered
    
    def check_strategy_triggers(self, strategy: str, context: Dict[str, Any] = None) -> bool:
        """
        Check triggers for a specific strategy.
        
        Args:
            strategy: Strategy to check triggers for
            context: Optional context dictionary
            
        Returns:
            Boolean indicating if any strategy trigger is activated
        """
        # Skip if global circuit breaker is open
        if self.global_state == CircuitBreakerState.OPEN:
            return True
        
        # Skip if in override mode
        if strategy in self.strategy_overrides:
            return False
        
        # Skip if already open
        if self.strategy_states.get(strategy) == CircuitBreakerState.OPEN:
            return True
        
        # Skip if no triggers for this strategy
        if strategy not in self.strategy_triggers:
            return False
        
        context = context or {}
        context['strategy'] = strategy
        triggered = False
        
        for trigger in self.strategy_triggers[strategy]:
            if trigger.check(context):
                triggered = True
                logger.warning(f"Circuit breaker triggered for strategy {strategy}: {trigger.name}")
        
        if triggered:
            self.strategy_states[strategy] = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker activated for strategy {strategy}")
        
        return triggered
    
    def check_all_triggers(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check all triggers.
        
        Args:
            context: Optional context dictionary
            
        Returns:
            Dictionary with triggered status for global, symbols, markets, and strategies
        """
        context = context or {}
        result = {
            'global': self.check_global_triggers(context),
            'symbols': {},
            'markets': {},
            'strategies': {}
        }
        
        # Check symbol triggers
        for symbol in self.symbol_triggers:
            result['symbols'][symbol] = self.check_symbol_triggers(symbol, context)
        
        # Check market triggers
        for market in self.market_triggers:
            result['markets'][market] = self.check_market_triggers(market, context)
        
        # Check strategy triggers
        for strategy in self.strategy_triggers:
            result['strategies'][strategy] = self.check_strategy_triggers(strategy, context)
        
        return result
    
    def check(self, metrics: Dict[str, Any]) -> None:
        """
        Check if any metrics breach thresholds and raise exception if needed.
        
        Args:
            metrics: Dictionary of metrics to check
            
        Raises:
            CircuitBreakerException: If any thresholds are breached
        """
        logger.info(f"Checking circuit breaker metrics: {metrics}")
        
        # Create context from metrics
        context = {'metrics': metrics}
        
        # Check drawdown if available
        if 'max_drawdown' in metrics:
            drawdown = abs(metrics['max_drawdown'])
            drawdown_threshold = self.config.get('thresholds', {}).get('max_drawdown', 0.15)  # Default 15%
            
            if drawdown > drawdown_threshold:
                logger.warning(f"Circuit breaker triggered: Drawdown {drawdown:.2%} exceeds threshold {drawdown_threshold:.2%}")
                self.global_state = CircuitBreakerState.OPEN
                raise CircuitBreakerException(f"Maximum drawdown threshold exceeded: {drawdown:.2%} > {drawdown_threshold:.2%}")
        
        # Check Sharpe ratio if available
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            min_sharpe = self.config.get('thresholds', {}).get('min_sharpe', 0.0)  # Default 0
            
            if sharpe < min_sharpe:
                logger.warning(f"Circuit breaker triggered: Sharpe ratio {sharpe:.2f} below threshold {min_sharpe:.2f}")
                self.global_state = CircuitBreakerState.OPEN
                raise CircuitBreakerException(f"Sharpe ratio below threshold: {sharpe:.2f} < {min_sharpe:.2f}")
        
        # Check error rate if available
        if 'error_rate' in metrics:
            error_rate = metrics['error_rate']
            max_error_rate = self.config.get('thresholds', {}).get('max_error_rate', 0.05)  # Default 5%
            
            if error_rate > max_error_rate:
                logger.warning(f"Circuit breaker triggered: Error rate {error_rate:.2%} exceeds threshold {max_error_rate:.2%}")
                self.global_state = CircuitBreakerState.OPEN
                raise CircuitBreakerException(f"Error rate threshold exceeded: {error_rate:.2%} > {max_error_rate:.2%}")
        
        # Check all triggers with the metrics context
        triggered = self.check_all_triggers(context)
        
        # If any global trigger is activated, raise exception
        if triggered.get('global', False):
            raise CircuitBreakerException("Global circuit breaker triggered")
        
        # Check if any symbol, market, or strategy triggers are activated
        for category, items in triggered.items():
            if category != 'global':
                for item, is_triggered in items.items():
                    if is_triggered:
                        raise CircuitBreakerException(f"Circuit breaker triggered for {category[:-1]} {item}")
    
    def allow_trading(self) -> bool:
        """
        Check if global trading is allowed.
        
        Returns:
            Boolean indicating if trading is allowed
        """
        # Test mode allows trading regardless of circuit breaker state
        if self.test_mode:
            return True
        
        # Global override allows trading
        if self.global_override:
            return True
        
        # Check global circuit breaker state
        return self.global_state != CircuitBreakerState.OPEN
    
    def allow_trading_for_symbol(self, symbol: str) -> bool:
        """
        Check if trading is allowed for a specific symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Boolean indicating if trading is allowed for the symbol
        """
        # Check global first
        if not self.allow_trading():
            return False
        
        # Test mode allows trading regardless of circuit breaker state
        if self.test_mode:
            return True
        
        # Symbol override allows trading
        if symbol in self.symbol_overrides:
            return True
        
        # Check symbol circuit breaker state
        if symbol in self.symbol_states and self.symbol_states[symbol] == CircuitBreakerState.OPEN:
            return False
        
        # Check market circuit breaker state (using symbol's market)
        symbol_market = self._get_symbol_market(symbol)
        if symbol_market and symbol_market in self.market_states and self.market_states[symbol_market] == CircuitBreakerState.OPEN:
            return False
        
        return True
    
    def allow_trading_for_market(self, market: str) -> bool:
        """
        Check if trading is allowed for a specific market.
        
        Args:
            market: Market to check
            
        Returns:
            Boolean indicating if trading is allowed for the market
        """
        # Check global first
        if not self.allow_trading():
            return False
        
        # Test mode allows trading regardless of circuit breaker state
        if self.test_mode:
            return True
        
        # Market override allows trading
        if market in self.market_overrides:
            return True
        
        # Check market circuit breaker state
        if market in self.market_states and self.market_states[market] == CircuitBreakerState.OPEN:
            return False
        
        return True
    
    def allow_trading_for_strategy(self, strategy: str) -> bool:
        """
        Check if trading is allowed for a specific strategy.
        
        Args:
            strategy: Strategy to check
            
        Returns:
            Boolean indicating if trading is allowed for the strategy
        """
        # Check global first
        if not self.allow_trading():
            return False
        
        # Test mode allows trading regardless of circuit breaker state
        if self.test_mode:
            return True
        
        # Strategy override allows trading
        if strategy in self.strategy_overrides:
            return True
        
        # Check strategy circuit breaker state
        if strategy in self.strategy_states and self.strategy_states[strategy] == CircuitBreakerState.OPEN:
            return False
        
        return True
    
    def reset_global(self, force: bool = False) -> bool:
        """
        Reset global circuit breaker.
        
        Args:
            force: Whether to force reset regardless of trigger conditions
            
        Returns:
            Boolean indicating if reset was successful
        """
        # Skip if not open
        if self.global_state != CircuitBreakerState.OPEN:
            return True
        
        # Force reset
        if force:
            self.global_state = CircuitBreakerState.CLOSED
            logger.info("Global circuit breaker force reset")
            return True
        
        # Check if all triggers can be reset
        context = {}
        all_reset = True
        
        for trigger in self.global_triggers:
            if trigger.is_active and not trigger.can_reset(context):
                all_reset = False
                logger.info(f"Global trigger {trigger.name} cannot be reset yet")
        
        if all_reset:
            self.global_state = CircuitBreakerState.CLOSED
            logger.info("Global circuit breaker reset")
            return True
        
        return False
    
    def reset_symbol(self, symbol: str, force: bool = False) -> bool:
        """
        Reset circuit breaker for a specific symbol.
        
        Args:
            symbol: Symbol to reset
            force: Whether to force reset regardless of trigger conditions
            
        Returns:
            Boolean indicating if reset was successful
        """
        # Skip if not open
        if symbol not in self.symbol_states or self.symbol_states[symbol] != CircuitBreakerState.OPEN:
            return True
        
        # Force reset
        if force:
            self.symbol_states[symbol] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for symbol {symbol} force reset")
            return True
        
        # Skip if no triggers for this symbol
        if symbol not in self.symbol_triggers:
            return True
        
        # Check if all triggers can be reset
        context = {'symbol': symbol}
        all_reset = True
        
        for trigger in self.symbol_triggers[symbol]:
            if trigger.is_active and not trigger.can_reset(context):
                all_reset = False
                logger.info(f"Trigger {trigger.name} for symbol {symbol} cannot be reset yet")
        
        if all_reset:
            self.symbol_states[symbol] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for symbol {symbol} reset")
            return True
        
        return False
    
    def reset_market(self, market: str, force: bool = False) -> bool:
        """
        Reset circuit breaker for a specific market.
        
        Args:
            market: Market to reset
            force: Whether to force reset regardless of trigger conditions
            
        Returns:
            Boolean indicating if reset was successful
        """
        # Skip if not open
        if market not in self.market_states or self.market_states[market] != CircuitBreakerState.OPEN:
            return True
        
        # Force reset
        if force:
            self.market_states[market] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for market {market} force reset")
            return True
        
        # Skip if no triggers for this market
        if market not in self.market_triggers:
            return True
        
        # Check if all triggers can be reset
        context = {'market': market}
        all_reset = True
        
        for trigger in self.market_triggers[market]:
            if trigger.is_active and not trigger.can_reset(context):
                all_reset = False
                logger.info(f"Trigger {trigger.name} for market {market} cannot be reset yet")
        
        if all_reset:
            self.market_states[market] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for market {market} reset")
            return True
        
        return False
    
    def reset_strategy(self, strategy: str, force: bool = False) -> bool:
        """
        Reset circuit breaker for a specific strategy.
        
        Args:
            strategy: Strategy to reset
            force: Whether to force reset regardless of trigger conditions
            
        Returns:
            Boolean indicating if reset was successful
        """
        # Skip if not open
        if strategy not in self.strategy_states or self.strategy_states[strategy] != CircuitBreakerState.OPEN:
            return True
        
        # Force reset
        if force:
            self.strategy_states[strategy] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for strategy {strategy} force reset")
            return True
        
        # Skip if no triggers for this strategy
        if strategy not in self.strategy_triggers:
            return True
        
        # Check if all triggers can be reset
        context = {'strategy': strategy}
        all_reset = True
        
        for trigger in self.strategy_triggers[strategy]:
            if trigger.is_active and not trigger.can_reset(context):
                all_reset = False
                logger.info(f"Trigger {trigger.name} for strategy {strategy} cannot be reset yet")
        
        if all_reset:
            self.strategy_states[strategy] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for strategy {strategy} reset")
            return True
        
        return False
    
    def reset_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Reset all circuit breakers.
        
        Args:
            force: Whether to force reset regardless of trigger conditions
            
        Returns:
            Dictionary with reset status for global, symbols, markets, and strategies
        """
        result = {
            'global': self.reset_global(force),
            'symbols': {},
            'markets': {},
            'strategies': {}
        }
        
        # Reset symbol circuit breakers
        for symbol in self.symbol_states:
            result['symbols'][symbol] = self.reset_symbol(symbol, force)
        
        # Reset market circuit breakers
        for market in self.market_states:
            result['markets'][market] = self.reset_market(market, force)
        
        # Reset strategy circuit breakers
        for strategy in self.strategy_states:
            result['strategies'][strategy] = self.reset_strategy(strategy, force)
        
        return result
    
    def set_global_override(self, override: bool):
        """
        Set global override.
        
        Args:
            override: Whether to override global circuit breaker
        """
        self.global_override = override
        logger.info(f"Global circuit breaker override set to {override}")
    
    def set_symbol_override(self, symbol: str, override: bool):
        """
        Set symbol override.
        
        Args:
            symbol: Symbol to set override for
            override: Whether to override symbol circuit breaker
        """
        if override:
            self.symbol_overrides.add(symbol)
        else:
            self.symbol_overrides.discard(symbol)
        logger.info(f"Circuit breaker override for symbol {symbol} set to {override}")
    
    def set_market_override(self, market: str, override: bool):
        """
        Set market override.
        
        Args:
            market: Market to set override for
            override: Whether to override market circuit breaker
        """
        if override:
            self.market_overrides.add(market)
        else:
            self.market_overrides.discard(market)
        logger.info(f"Circuit breaker override for market {market} set to {override}")
    
    def set_strategy_override(self, strategy: str, override: bool):
        """
        Set strategy override.
        
        Args:
            strategy: Strategy to set override for
            override: Whether to override strategy circuit breaker
        """
        if override:
            self.strategy_overrides.add(strategy)
        else:
            self.strategy_overrides.discard(strategy)
        logger.info(f"Circuit breaker override for strategy {strategy} set to {override}")
    
    def set_test_mode(self, test_mode: bool):
        """
        Set test mode.
        
        Args:
            test_mode: Whether to enable test mode
        """
        self.test_mode = test_mode
        logger.info(f"Circuit breaker test mode set to {test_mode}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get circuit breaker status.
        
        Returns:
            Dictionary with circuit breaker status
        """
        status = {
            'global': {
                'state': self.global_state.value,
                'override': self.global_override,
                'triggers': [trigger.to_dict() for trigger in self.global_triggers]
            },
            'symbols': {},
            'markets': {},
            'strategies': {},
            'test_mode': self.test_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add symbol status
        for symbol in self.symbol_states:
            status['symbols'][symbol] = {
                'state': self.symbol_states[symbol].value,
                'override': symbol in self.symbol_overrides,
                'triggers': [trigger.to_dict() for trigger in self.symbol_triggers.get(symbol, [])]
            }
        
        # Add market status
        for market in self.market_states:
            status['markets'][market] = {
                'state': self.market_states[market].value,
                'override': market in self.market_overrides,
                'triggers': [trigger.to_dict() for trigger in self.market_triggers.get(market, [])]
            }
        
        # Add strategy status
        for strategy in self.strategy_states:
            status['strategies'][strategy] = {
                'state': self.strategy_states[strategy].value,
                'override': strategy in self.strategy_overrides,
                'triggers': [trigger.to_dict() for trigger in self.strategy_triggers.get(strategy, [])]
            }
        
        return status
    
    def _get_symbol_market(self, symbol: str) -> Optional[str]:
        """
        Get market for a symbol.
        
        Args:
            symbol: Symbol to get market for
            
        Returns:
            Market name or None if unknown
        """
        # Check if symbol has market prefix (e.g., 'NYSE:AAPL')
        if ':' in symbol:
            return symbol.split(':')[0]
        
        # Check if US stock symbol (likely all uppercase, 1-5 characters)
        if symbol.isupper() and 1 <= len(symbol) <= 5 and symbol.isalpha():
            return 'US_STOCKS'
        
        # Check if crypto (usually has 'USD' suffix or 'BTC', 'ETH' in name)
        if 'USD' in symbol or 'BTC' in symbol or 'ETH' in symbol:
            return 'CRYPTO'
        
        # Unknown market
        return None
    
    def _start_auto_reset_thread(self):
        """
        Start the auto-reset thread.
        """
        if self._auto_reset_thread is not None and self._auto_reset_thread.is_alive():
            return
        
        self._stop_auto_reset.clear()
        self._auto_reset_thread = threading.Thread(target=self._auto_reset_loop)
        self._auto_reset_thread.daemon = True
        self._auto_reset_thread.start()
        logger.info("Auto-reset thread started")
    
    def _stop_auto_reset_thread(self):
        """
        Stop the auto-reset thread.
        """
        if self._auto_reset_thread is None or not self._auto_reset_thread.is_alive():
            return
        
        self._stop_auto_reset.set()
        self._auto_reset_thread.join(timeout=1.0)
        logger.info("Auto-reset thread stopped")
    
    def _auto_reset_loop(self):
        """
        Auto-reset loop.
        """
        while not self._stop_auto_reset.is_set():
            try:
                # Reset all circuit breakers (non-forced)
                self.reset_all(force=False)
            except Exception as e:
                logger.error(f"Error in auto-reset loop: {str(e)}")
            
            # Sleep for the interval
            self._stop_auto_reset.wait(self.auto_reset_interval)
    
    def __del__(self):
        """
        Cleanup on deletion.
        """
        self._stop_auto_reset_thread()


class CircuitBreakerException(Exception):
    """
    Exception raised when a circuit breaker is triggered.
    """
    pass


# Default circuit breaker instance
default_circuit_breaker = None


def get_circuit_breaker(config_path: Optional[str] = None) -> CircuitBreaker:
    """
    Get or create the default circuit breaker.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        CircuitBreaker instance
    """
    global default_circuit_breaker
    
    if default_circuit_breaker is None:
        default_circuit_breaker = CircuitBreaker(config_path)
    
    return default_circuit_breaker


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Circuit Breaker for Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--status', action='store_true', help='Get circuit breaker status')
    parser.add_argument('--check', action='store_true', help='Check all triggers')
    parser.add_argument('--reset', action='store_true', help='Reset all circuit breakers')
    parser.add_argument('--force', action='store_true', help='Force reset (used with --reset)')
    parser.add_argument('--symbol', type=str, help='Specific symbol')
    parser.add_argument('--market', type=str, help='Specific market')
    parser.add_argument('--strategy', type=str, help='Specific strategy')
    parser.add_argument('--override', type=str, choices=['on', 'off'], help='Set override status')
    parser.add_argument('--test-mode', type=str, choices=['on', 'off'], help='Set test mode')
    
    args = parser.parse_args()
    
    # Create circuit breaker
    circuit_breaker = CircuitBreaker(args.config)
    
    if args.test_mode is not None:
        circuit_breaker.set_test_mode(args.test_mode == 'on')
    
    if args.override is not None:
        override = args.override == 'on'
        
        if args.symbol:
            circuit_breaker.set_symbol_override(args.symbol, override)
        elif args.market:
            circuit_breaker.set_market_override(args.market, override)
        elif args.strategy:
            circuit_breaker.set_strategy_override(args.strategy, override)
        else:
            circuit_breaker.set_global_override(override)
    
    if args.reset:
        force = args.force
        
        if args.symbol:
            result = circuit_breaker.reset_symbol(args.symbol, force)
            print(f"Reset symbol {args.symbol}: {result}")
        elif args.market:
            result = circuit_breaker.reset_market(args.market, force)
            print(f"Reset market {args.market}: {result}")
        elif args.strategy:
            result = circuit_breaker.reset_strategy(args.strategy, force)
            print(f"Reset strategy {args.strategy}: {result}")
        else:
            result = circuit_breaker.reset_all(force)
            print(f"Reset all: {result}")
    
    if args.check:
        context = {}
        
        if args.symbol:
            result = circuit_breaker.check_symbol_triggers(args.symbol, context)
            print(f"Check symbol {args.symbol}: {result}")
        elif args.market:
            result = circuit_breaker.check_market_triggers(args.market, context)
            print(f"Check market {args.market}: {result}")
        elif args.strategy:
            result = circuit_breaker.check_strategy_triggers(args.strategy, context)
            print(f"Check strategy {args.strategy}: {result}")
        else:
            result = circuit_breaker.check_all_triggers(context)
            print(f"Check all: {result}")
    
    if args.status:
        status = circuit_breaker.get_status()
        import json
        print(json.dumps(status, indent=2))
