#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position Tracker
--------------
Tracks and manages trading positions.
Keeps a record of open and closed positions and provides position analytics.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
import uuid
from collections import defaultdict

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger

# Setup logging
logger = setup_logger('position_tracker')


class Position:
    """
    Represents a trading position.
    """
    
    def __init__(self, symbol: str, quantity: float, entry_price: float, 
               entry_time: datetime, strategy: Optional[str] = None,
               stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
               position_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Initialize a position.
        
        Args:
            symbol: Symbol
            quantity: Position quantity
            entry_price: Entry price
            entry_time: Entry time
            strategy: Strategy name
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_id: Optional position ID (generated if not provided)
            metadata: Optional position metadata
        """
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.strategy = strategy
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_id = position_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        
        # Exit information
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        
        # Current position information
        self.current_price = entry_price
        self.current_value = quantity * entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        
        # Performance information
        self.max_profit = 0.0
        self.max_profit_pct = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.holding_period_days = 0.0
        
        # Risk information
        self.risk_amount = 0.0
        if stop_loss:
            self.risk_amount = (entry_price - stop_loss) * quantity
        
        # Flags
        self.is_open = True
        self.is_profitable = False
        
        # Last update time
        self.last_update = entry_time
    
    def update_price(self, current_price: float, timestamp: Optional[datetime] = None):
        """
        Update position with current price.
        
        Args:
            current_price: Current price
            timestamp: Update timestamp
        """
        # Use current time if not provided
        timestamp = timestamp or datetime.now()
        
        # Calculate holding period
        self.holding_period_days = (timestamp - self.entry_time).total_seconds() / (24 * 60 * 60)
        
        # Update current information
        self.current_price = current_price
        self.current_value = self.quantity * current_price
        
        # Calculate PnL
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100
        
        # Update flags
        self.is_profitable = self.unrealized_pnl > 0
        
        # Update max profit & drawdown
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
            self.max_profit_pct = self.unrealized_pnl_pct
        
        drawdown = self.max_profit - self.unrealized_pnl
        drawdown_pct = self.max_profit_pct - self.unrealized_pnl_pct
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct
        
        # Update last update time
        self.last_update = timestamp
    
    def close_position(self, exit_price: float, exit_time: Optional[datetime] = None, 
                     exit_reason: Optional[str] = None, partial_quantity: Optional[float] = None):
        """
        Close the position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit time (current time if not provided)
            exit_reason: Reason for exit
            partial_quantity: Optional quantity to close (for partial exits)
        """
        # Use current time if not provided
        exit_time = exit_time or datetime.now()
        
        # Handle partial exits
        if partial_quantity is not None and partial_quantity < self.quantity:
            # Calculate the profit for the partial exit
            partial_profit = (exit_price - self.entry_price) * partial_quantity
            partial_profit_pct = (exit_price / self.entry_price - 1) * 100
            
            # Update the position quantity
            self.quantity -= partial_quantity
            self.current_value = self.quantity * self.current_price
            
            # Update PnL
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (self.current_price / self.entry_price - 1) * 100
            
            # Create a return dict for the partial exit
            return {
                'symbol': self.symbol,
                'quantity': partial_quantity,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'profit': partial_profit,
                'profit_pct': partial_profit_pct,
                'holding_period_days': (exit_time - self.entry_time).total_seconds() / (24 * 60 * 60),
                'exit_reason': exit_reason,
                'position_id': self.position_id,
                'is_partial': True
            }
        else:
            # Full exit
            self.exit_price = exit_price
            self.exit_time = exit_time
            self.exit_reason = exit_reason
            
            # Calculate realized PnL
            realized_pnl = (exit_price - self.entry_price) * self.quantity
            realized_pnl_pct = (exit_price / self.entry_price - 1) * 100
            
            # Update flags
            self.is_open = False
            self.is_profitable = realized_pnl > 0
            
            # Update last update time
            self.last_update = exit_time
            
            # Return exit information
            return {
                'symbol': self.symbol,
                'quantity': self.quantity,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'profit': realized_pnl,
                'profit_pct': realized_pnl_pct,
                'holding_period_days': (exit_time - self.entry_time).total_seconds() / (24 * 60 * 60),
                'max_profit': self.max_profit,
                'max_profit_pct': self.max_profit_pct,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown_pct,
                'exit_reason': exit_reason,
                'strategy': self.strategy,
                'position_id': self.position_id,
                'is_partial': False
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary.
        
        Returns:
            Dictionary representation of the position
        """
        position_dict = {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'strategy': self.strategy,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_id': self.position_id,
            'metadata': self.metadata,
            'current_price': self.current_price,
            'current_value': self.current_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'max_profit': self.max_profit,
            'max_profit_pct': self.max_profit_pct,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'holding_period_days': self.holding_period_days,
            'risk_amount': self.risk_amount,
            'is_open': self.is_open,
            'is_profitable': self.is_profitable,
            'last_update': self.last_update.isoformat()
        }
        
        # Add exit information if closed
        if not self.is_open:
            position_dict.update({
                'exit_price': self.exit_price,
                'exit_time': self.exit_time.isoformat() if self.exit_time else None,
                'exit_reason': self.exit_reason,
                'realized_pnl': (self.exit_price - self.entry_price) * self.quantity,
                'realized_pnl_pct': (self.exit_price / self.entry_price - 1) * 100
            })
        
        return position_dict


class PositionTracker:
    """
    Tracks and manages positions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize position tracker.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Position storage
        self.positions = {}  # All positions (open and closed)
        self.open_positions = {}  # Only open positions
        self.closed_positions = {}  # Only closed positions
        
        # Position grouping
        self.positions_by_symbol = defaultdict(list)
        self.positions_by_strategy = defaultdict(list)
        
        # Performance metrics
        self.total_profit = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        
        # Position history
        self.position_history = []
        self.daily_pnl = {}
        
        # For Alpaca integration
        self.alpaca_position_map = {}  # Maps Alpaca position IDs to our position IDs
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Position tracker initialized")
    
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
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def add_position(self, position_data: Dict[str, Any]) -> str:
        """
        Add a new position.
        
        Args:
            position_data: Position data dictionary
            
        Returns:
            Position ID
        """
        with self.lock:
            # Extract position data
            symbol = position_data.get('symbol')
            quantity = position_data.get('quantity')
            entry_price = position_data.get('entry_price')
            entry_time = position_data.get('entry_time', datetime.now())
            
            # Ensure entry_time is a datetime object
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except ValueError:
                    entry_time = datetime.now()
            
            # Additional position data
            strategy = position_data.get('strategy')
            stop_loss = position_data.get('stop_loss')
            take_profit = position_data.get('take_profit')
            position_id = position_data.get('position_id')
            metadata = position_data.get('metadata', {})
            
            # Create position object
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=entry_time,
                strategy=strategy,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_id=position_id,
                metadata=metadata
            )
            
            # Store position
            self.positions[position.position_id] = position
            self.open_positions[position.position_id] = position
            
            # Update groupings
            self.positions_by_symbol[symbol].append(position.position_id)
            if strategy:
                self.positions_by_strategy[strategy].append(position.position_id)
            
            # Handle Alpaca integration
            if 'alpaca_id' in metadata:
                self.alpaca_position_map[metadata['alpaca_id']] = position.position_id
            
            logger.info(f"Added position for {symbol}: {quantity} shares at {entry_price}")
            
            return position.position_id
    
    def update_position(self, symbol: str, quantity: Optional[float] = None,
                      exit_price: Optional[float] = None, exit_time: Optional[datetime] = None,
                      current_price: Optional[float] = None, exit_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a position.
        
        Args:
            symbol: Symbol
            quantity: Quantity to close (for partial exits)
            exit_price: Exit price (for closes)
            exit_time: Exit time
            current_price: Current price (for price updates)
            exit_reason: Reason for exit
            
        Returns:
            Dictionary with update result
        """
        with self.lock:
            # Find positions for the symbol
            position_ids = self.positions_by_symbol.get(symbol, [])
            
            # Filter to open positions
            open_position_ids = [pid for pid in position_ids if pid in self.open_positions]
            
            if not open_position_ids:
                logger.warning(f"No open positions found for {symbol}")
                return {'error': f"No open positions found for {symbol}"}
            
            # For simplicity, we'll work with the first open position for the symbol
            # A more sophisticated implementation would handle multiple positions differently
            position_id = open_position_ids[0]
            position = self.open_positions[position_id]
            
            # Handle position close
            if exit_price is not None:
                # Close position (fully or partially)
                result = position.close_position(
                    exit_price=exit_price,
                    exit_time=exit_time,
                    exit_reason=exit_reason,
                    partial_quantity=quantity
                )
                
                # If position is fully closed, move to closed positions
                if not position.is_open:
                    self.closed_positions[position_id] = position
                    del self.open_positions[position_id]
                    
                    # Update performance metrics
                    self.total_trades += 1
                    profit = result.get('profit', 0)
                    self.total_profit += profit
                    
                    if profit > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                    
                    # Add to position history
                    self.position_history.append(result)
                    
                    # Update daily PnL
                    exit_date = (exit_time or datetime.now()).date().isoformat()
                    if exit_date not in self.daily_pnl:
                        self.daily_pnl[exit_date] = 0
                    self.daily_pnl[exit_date] += profit
                    
                    logger.info(f"Closed position for {symbol}: {result.get('quantity')} shares at {exit_price}, P&L: {profit:.2f}")
                else:
                    # Partial close
                    logger.info(f"Partially closed position for {symbol}: {quantity} shares at {exit_price}")
                
                return result
            
            # Handle price update
            elif current_price is not None:
                # Update position with current price
                position.update_price(current_price, exit_time)
                
                logger.debug(f"Updated price for {symbol}: {current_price}")
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct
                }
            
            else:
                return {'error': "No exit price or current price provided"}
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position information for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            Position dictionary or None if not found
        """
        with self.lock:
            # Find positions for the symbol
            position_ids = self.positions_by_symbol.get(symbol, [])
            
            # Filter to open positions
            open_position_ids = [pid for pid in position_ids if pid in self.open_positions]
            
            if not open_position_ids:
                return None
            
            # Return first open position (assuming one position per symbol)
            position = self.open_positions[open_position_ids[0]]
            
            return position.to_dict()
    
    def get_position_by_id(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get position information by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position dictionary or None if not found
        """
        with self.lock:
            position = self.positions.get(position_id)
            
            if position is None:
                return None
            
            return position.to_dict()
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        with self.lock:
            return [position.to_dict() for position in self.open_positions.values()]
    
    def get_closed_positions(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get closed positions.
        
        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            List of closed position dictionaries
        """
        with self.lock:
            positions = []
            
            for position in self.closed_positions.values():
                # Skip if no exit time (shouldn't happen for closed positions)
                if not position.exit_time:
                    continue
                
                # Filter by date range if provided
                exit_date = position.exit_time.date().isoformat()
                
                if start_date and exit_date < start_date:
                    continue
                
                if end_date and exit_date > end_date:
                    continue
                
                positions.append(position.to_dict())
            
            return positions
    
    def get_positions_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            List of position dictionaries
        """
        with self.lock:
            position_ids = self.positions_by_strategy.get(strategy, [])
            
            return [self.positions[pid].to_dict() for pid in position_ids if pid in self.positions]
    
    def get_positions_by_symbol(self, symbol: str, include_closed: bool = False) -> List[Dict[str, Any]]:
        """
        Get all positions for a symbol.
        
        Args:
            symbol: Symbol
            include_closed: Whether to include closed positions
            
        Returns:
            List of position dictionaries
        """
        with self.lock:
            position_ids = self.positions_by_symbol.get(symbol, [])
            
            if not include_closed:
                position_ids = [pid for pid in position_ids if pid in self.open_positions]
            
            return [self.positions[pid].to_dict() for pid in position_ids if pid in self.positions]
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns:
            Portfolio value
        """
        with self.lock:
            return sum(position.current_value for position in self.open_positions.values())
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation by symbol.
        
        Returns:
            Dictionary of symbol to allocation percentage
        """
        with self.lock:
            portfolio_value = self.get_portfolio_value()
            
            if portfolio_value == 0:
                return {}
            
            allocations = {}
            
            for position in self.open_positions.values():
                symbol = position.symbol
                allocation = position.current_value / portfolio_value
                
                if symbol in allocations:
                    allocations[symbol] += allocation
                else:
                    allocations[symbol] = allocation
            
            return allocations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            # Calculate win rate
            win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
            
            # Calculate profit factor
            total_gains = sum(pos.get('profit', 0) for pos in self.position_history if pos.get('profit', 0) > 0)
            total_losses = sum(abs(pos.get('profit', 0)) for pos in self.position_history if pos.get('profit', 0) < 0)
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
            
            # Calculate average profit and loss
            avg_profit = total_gains / self.win_count if self.win_count > 0 else 0
            avg_loss = total_losses / self.loss_count if self.loss_count > 0 else 0
            
            # Calculate expectancy
            expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss) if self.total_trades > 0 else 0
            
            # Calculate average holding period
            avg_holding_period = sum(pos.get('holding_period_days', 0) for pos in self.position_history) / self.total_trades if self.total_trades > 0 else 0
            
            # Calculate current drawdown
            # For a proper drawdown calculation, we should track equity curve historically
            # This is a simplified version based on closed positions
            peak_equity = max(self.daily_pnl.values()) if self.daily_pnl else 0
            current_equity = self.total_profit
            current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            
            return {
                'total_profit': self.total_profit,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'total_trades': self.total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'expectancy': expectancy,
                'avg_holding_period': avg_holding_period,
                'current_drawdown': current_drawdown,
                'open_positions': len(self.open_positions),
                'portfolio_value': self.get_portfolio_value()
            }
    
    def get_daily_pnl(self) -> Dict[str, float]:
        """
        Get daily P&L.
        
        Returns:
            Dictionary of date to P&L
        """
        with self.lock:
            return self.daily_pnl
    
    def sync_positions(self, broker_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize positions with broker data.
        Specific implementation for Alpaca.
        
        Args:
            broker_positions: Positions from broker
            
        Returns:
            Dictionary with sync results
        """
        with self.lock:
            # Track stats for sync operation
            stats = {
                'added': 0,
                'updated': 0,
                'closed': 0,
                'errors': 0
            }
            
            # Process broker positions
            for symbol, position_data in broker_positions.items():
                try:
                    alpaca_id = position_data.get('alpaca_id')
                    quantity = float(position_data.get('quantity', 0))
                    avg_entry_price = float(position_data.get('avg_entry_price', 0))
                    current_price = float(position_data.get('current_price', avg_entry_price))
                    
                    # Check if position exists by Alpaca ID
                    existing_position_id = self.alpaca_position_map.get(alpaca_id)
                    
                    if existing_position_id:
                        # Position exists, update it
                        position = self.positions.get(existing_position_id)
                        
                        if position:
                            # Update based on broker data
                            if quantity == 0:
                                # Position closed at broker
                                if position.is_open:
                                    # Close position in our tracker
                                    result = position.close_position(
                                        exit_price=current_price,
                                        exit_time=datetime.now(),
                                        exit_reason='broker_sync'
                                    )
                                    
                                    # Move to closed positions
                                    self.closed_positions[existing_position_id] = position
                                    del self.open_positions[existing_position_id]
                                    
                                    stats['closed'] += 1
                                    logger.info(f"Closed position for {symbol} during sync")
                            else:
                                # Update position with current price
                                position.update_price(current_price)
                                
                                # Update quantity if different
                                if position.quantity != quantity:
                                    position.quantity = quantity
                                    position.current_value = quantity * current_price
                                
                                stats['updated'] += 1
                    else:
                        # Position doesn't exist, add it if quantity > 0
                        if quantity > 0:
                            # Create new position
                            new_position = {
                                'symbol': symbol,
                                'quantity': quantity,
                                'entry_price': avg_entry_price,
                                'entry_time': datetime.now(),
                                'strategy': 'broker_sync',
                                'metadata': {
                                    'alpaca_id': alpaca_id,
                                    'broker': 'alpaca',
                                    'sync_created': True
                                }
                            }
                            
                            # Add position
                            self.add_position(new_position)
                            
                            stats['added'] += 1
                            logger.info(f"Added position for {symbol} during sync")
                
                except Exception as e:
                    logger.error(f"Error syncing position for {symbol}: {str(e)}")
                    stats['errors'] += 1
            
            # Check for positions we have that aren't at broker
            broker_symbols = set(broker_positions.keys())
            our_symbols = set(symbol for position in self.open_positions.values() for symbol in [position.symbol])
            
            missing_at_broker = our_symbols - broker_symbols
            
            for symbol in missing_at_broker:
                # Get positions for this symbol
                position_ids = [pid for pid in self.positions_by_symbol.get(symbol, []) 
                             if pid in self.open_positions]
                
                for pid in position_ids:
                    position = self.open_positions[pid]
                    
                    # Skip positions not created through sync
                    if not position.metadata.get('sync_created', False):
                        continue
                    
                    # Close position since it's no longer at broker
                    result = position.close_position(
                        exit_price=position.current_price,
                        exit_time=datetime.now(),
                        exit_reason='broker_missing'
                    )
                    
                    # Move to closed positions
                    self.closed_positions[pid] = position
                    del self.open_positions[pid]
                    
                    stats['closed'] += 1
                    logger.info(f"Closed position for {symbol} - missing at broker")
            
            logger.info(f"Position sync completed: added={stats['added']}, updated={stats['updated']}, closed={stats['closed']}, errors={stats['errors']}")
            
            return stats
    
    def save_state(self, file_path: str) -> bool:
        """
        Save position tracker state to file.
        
        Args:
            file_path: Path to save state
            
        Returns:
            Boolean indicating success
        """
        with self.lock:
            try:
                # Prepare state dictionary
                state = {
                    'positions': {pid: pos.to_dict() for pid, pos in self.positions.items()},
                    'open_positions': list(self.open_positions.keys()),
                    'closed_positions': list(self.closed_positions.keys()),
                    'positions_by_symbol': dict(self.positions_by_symbol),
                    'positions_by_strategy': dict(self.positions_by_strategy),
                    'total_profit': self.total_profit,
                    'win_count': self.win_count,
                    'loss_count': self.loss_count,
                    'total_trades': self.total_trades,
                    'position_history': self.position_history,
                    'daily_pnl': self.daily_pnl,
                    'alpaca_position_map': self.alpaca_position_map,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save state to file
                with open(file_path, 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.info(f"Position tracker state saved to {file_path}")
                return True
            
            except Exception as e:
                logger.error(f"Error saving position tracker state: {str(e)}")
                return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Load position tracker state from file.
        
        Args:
            file_path: Path to load state from
            
        Returns:
            Boolean indicating success
        """
        with self.lock:
            try:
                # Load state from file
                with open(file_path, 'r') as f:
                    state = json.load(f)
                
                # Clear current state
                self.positions = {}
                self.open_positions = {}
                self.closed_positions = {}
                self.positions_by_symbol = defaultdict(list)
                self.positions_by_strategy = defaultdict(list)
                
                # Recreate positions
                for pid, pos_dict in state['positions'].items():
                    symbol = pos_dict['symbol']
                    quantity = pos_dict['quantity']
                    entry_price = pos_dict['entry_price']
                    entry_time = datetime.fromisoformat(pos_dict['entry_time'])
                    strategy = pos_dict.get('strategy')
                    stop_loss = pos_dict.get('stop_loss')
                    take_profit = pos_dict.get('take_profit')
                    metadata = pos_dict.get('metadata', {})
                    
                    # Create position
                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        entry_time=entry_time,
                        strategy=strategy,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_id=pid,
                        metadata=metadata
                    )
                    
                    # Restore additional properties
                    position.current_price = pos_dict.get('current_price', entry_price)
                    position.current_value = pos_dict.get('current_value', quantity * entry_price)
                    position.unrealized_pnl = pos_dict.get('unrealized_pnl', 0.0)
                    position.unrealized_pnl_pct = pos_dict.get('unrealized_pnl_pct', 0.0)
                    position.max_profit = pos_dict.get('max_profit', 0.0)
                    position.max_profit_pct = pos_dict.get('max_profit_pct', 0.0)
                    position.max_drawdown = pos_dict.get('max_drawdown', 0.0)
                    position.max_drawdown_pct = pos_dict.get('max_drawdown_pct', 0.0)
                    position.holding_period_days = pos_dict.get('holding_period_days', 0.0)
                    position.is_open = pos_dict.get('is_open', True)
                    position.is_profitable = pos_dict.get('is_profitable', False)
                    
                    if not position.is_open:
                        # Restore exit information
                        position.exit_price = pos_dict.get('exit_price')
                        position.exit_time = datetime.fromisoformat(pos_dict.get('exit_time')) if pos_dict.get('exit_time') else None
                        position.exit_reason = pos_dict.get('exit_reason')
                    
                    # Store position
                    self.positions[pid] = position
                    
                    # Update collections
                    if position.is_open:
                        self.open_positions[pid] = position
                    else:
                        self.closed_positions[pid] = position
                    
                    self.positions_by_symbol[symbol].append(pid)
                    if strategy:
                        self.positions_by_strategy[strategy].append(pid)
                
                # Restore performance metrics
                self.total_profit = state.get('total_profit', 0.0)
                self.win_count = state.get('win_count', 0)
                self.loss_count = state.get('loss_count', 0)
                self.total_trades = state.get('total_trades', 0)
                
                # Restore history
                self.position_history = state.get('position_history', [])
                self.daily_pnl = state.get('daily_pnl', {})
                
                # Restore Alpaca mapping
                self.alpaca_position_map = state.get('alpaca_position_map', {})
                
                logger.info(f"Position tracker state loaded from {file_path}")
                return True
            
            except Exception as e:
                logger.error(f"Error loading position tracker state: {str(e)}")
                return False


# Default position tracker instance
default_position_tracker = None


def get_position_tracker(config_path: Optional[str] = None) -> PositionTracker:
    """
    Get or create the default position tracker.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        PositionTracker instance
    """
    global default_position_tracker
    
    if default_position_tracker is None:
        default_position_tracker = PositionTracker(config_path)
    
    return default_position_tracker


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Position Tracker')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--save', type=str, help='Save state to file')
    parser.add_argument('--load', type=str, help='Load state from file')
    parser.add_argument('--add', action='store_true', help='Add a test position')
    parser.add_argument('--positions', action='store_true', help='Print all positions')
    parser.add_argument('--metrics', action='store_true', help='Print performance metrics')
    
    args = parser.parse_args()
    
    # Create position tracker
    position_tracker = PositionTracker(args.config)
    
    if args.load:
        # Load state from file
        success = position_tracker.load_state(args.load)
        print(f"Loaded state from {args.load}: {success}")
    
    if args.add:
        # Add a test position
        position_id = position_tracker.add_position({
            'symbol': 'AAPL',
            'quantity': 10,
            'entry_price': 150.0,
            'entry_time': datetime.now(),
            'strategy': 'test',
            'stop_loss': 145.0,
            'take_profit': 160.0
        })
        print(f"Added test position: {position_id}")
    
    if args.positions:
        # Print all positions
        positions = position_tracker.get_all_positions()
        print(f"Open positions ({len(positions)}):")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} shares at {pos['entry_price']}, P&L: {pos['unrealized_pnl']:.2f}")
    
    if args.metrics:
        # Print performance metrics
        metrics = position_tracker.get_performance_metrics()
        print("Performance metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    if args.save:
        # Save state to file
        success = position_tracker.save_state(args.save)
        print(f"Saved state to {args.save}: {success}")