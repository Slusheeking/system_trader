#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Strategy
--------------
Implements strategy composition and signal processing logic.
Combines model signals into actionable trading strategies.
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
import uuid
import os

# Import your project's modules - adjust paths as needed
from utils.logging import setup_logger
from utils.config_loader import ConfigLoader
from utils.metrics import calculate_metrics
from models.market_regime.features import MarketRegimeFeatures
from models.market_regime.model import EnhancedMarketRegimeModel
from orchestration.decision_framework import DecisionFramework
from orchestration.adaptive_thresholds import AdaptiveThresholds
from trading.execution.order_router import OrderRouter, OrderRequest, OrderSide, OrderType, TimeInForce

# Try to import sentiment analysis functionality
try:
    from ml_training_engine_modified import generate_sentiment_features
    SENTIMENT_FEATURES_AVAILABLE = True
except ImportError:
    SENTIMENT_FEATURES_AVAILABLE = False

# Setup logging
logger = setup_logger('trading_strategy')

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading', 'config', 'strategy.yaml')


class StrategyType(Enum):
    """
    Types of trading strategies.
    """
    MOMENTUM = 'momentum'
    MEAN_REVERSION = 'mean_reversion'
    BREAKOUT = 'breakout'
    TREND_FOLLOWING = 'trend_following'
    STATISTICAL_ARBITRAGE = 'statistical_arbitrage'
    SENTIMENT_BASED = 'sentiment_based'
    CUSTOM = 'custom'


class PositionSizing:
    """
    Position sizing methods.
    """
    
    @staticmethod
    def fixed_dollar(equity: float, position_size: float) -> float:
        """
        Fixed dollar position sizing.
        
        Args:
            equity: Account equity
            position_size: Position size in dollars
            
        Returns:
            Position size in dollars
        """
        return min(position_size, equity)
    
    @staticmethod
    def fixed_percent(equity: float, percent: float) -> float:
        """
        Fixed percent position sizing.
        
        Args:
            equity: Account equity
            percent: Position size as percentage of equity (0-1)
            
        Returns:
            Position size in dollars
        """
        return equity * max(0.0, min(1.0, percent))
    
    @staticmethod
    def volatility_based(equity: float, percent: float, volatility: float,
                       volatility_multiplier: float = 1.0) -> float:
        """
        Volatility-based position sizing.
        
        Args:
            equity: Account equity
            percent: Maximum position size as percentage of equity (0-1)
            volatility: Asset volatility (e.g., ATR as percentage of price)
            volatility_multiplier: Multiplier for volatility adjustment
            
        Returns:
            Position size in dollars
        """
        # Adjust percent based on volatility
        adjusted_percent = percent / (volatility * volatility_multiplier) if volatility > 0 else percent
        
        # Cap at original percent
        adjusted_percent = min(adjusted_percent, percent)
        
        return equity * max(0.0, min(1.0, adjusted_percent))
    
    @staticmethod
    def kelly_criterion(equity: float, win_rate: float, win_loss_ratio: float,
                      max_percent: float = 0.2) -> float:
        """
        Kelly criterion position sizing.
        
        Args:
            equity: Account equity
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Ratio of average win to average loss
            max_percent: Maximum position size as percentage of equity (0-1)
            
        Returns:
            Position size in dollars
        """
        # Calculate Kelly percent
        kelly_percent = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply half-Kelly for safety and cap at max_percent
        half_kelly = kelly_percent * 0.5
        capped_kelly = max(0.0, min(max_percent, half_kelly))
        
        return equity * capped_kelly


class TradingStrategy:
    """
    Base class for trading strategies.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize trading strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.strategy_type = StrategyType(config.get('type', 'custom'))
        
        # Position sizing parameters
        self.position_sizing_method = config.get('position_sizing', {}).get('method', 'fixed_percent')
        self.position_size_value = config.get('position_sizing', {}).get('value', 0.02)  # 2% default
        self.max_position_size = config.get('position_sizing', {}).get('max_size', 0.05)  # 5% default
        
        # Risk parameters
        self.max_drawdown = config.get('risk', {}).get('max_drawdown', 0.10)  # 10% max drawdown
        self.stop_loss_pct = config.get('risk', {}).get('stop_loss', 0.02)  # 2% stop loss
        self.take_profit_pct = config.get('risk', {}).get('take_profit', 0.05)  # 5% take profit
        
        # Performance tracking
        self.win_rate = 0.5  # Default win rate
        self.win_loss_ratio = 1.0  # Default win/loss ratio
        self.trades = []
        self.open_positions = {}
        
        logger.info(f"Initialized {self.strategy_type.value} strategy: {self.name}")
    
    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals from models.
        Must be implemented by subclasses.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with trading decisions
        """
        raise NotImplementedError("Subclasses must implement process_signals")
    
    def calculate_position_size(self, symbol: str, equity: float, price: float, 
                              volatility: Optional[float] = None) -> float:
        """
        Calculate position size for a symbol.
        
        Args:
            symbol: Symbol to calculate position size for
            equity: Account equity
            price: Current price
            volatility: Optional volatility measure (e.g., ATR ratio)
            
        Returns:
            Position size in dollars
        """
        if self.position_sizing_method == 'fixed_dollar':
            position_dollars = PositionSizing.fixed_dollar(equity, self.position_size_value)
        
        elif self.position_sizing_method == 'fixed_percent':
            position_dollars = PositionSizing.fixed_percent(equity, self.position_size_value)
        
        elif self.position_sizing_method == 'volatility_based' and volatility is not None:
            position_dollars = PositionSizing.volatility_based(
                equity, self.position_size_value, volatility, 2.0
            )
        
        elif self.position_sizing_method == 'kelly_criterion':
            position_dollars = PositionSizing.kelly_criterion(
                equity, self.win_rate, self.win_loss_ratio, self.max_position_size
            )
        
        else:
            # Default to fixed percent
            position_dollars = PositionSizing.fixed_percent(equity, self.position_size_value)
        
        # Calculate quantity
        quantity = position_dollars / price if price > 0 else 0
        
        return quantity
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """
        Update strategy performance metrics.
        
        Args:
            trade_result: Trade result dictionary
        """
        # Add to trades list
        self.trades.append(trade_result)
        
        # Calculate win rate
        wins = sum(1 for trade in self.trades if trade.get('profit_pct', 0) > 0)
        self.win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0.5
        
        # Calculate win/loss ratio
        winning_trades = [trade for trade in self.trades if trade.get('profit_pct', 0) > 0]
        losing_trades = [trade for trade in self.trades if trade.get('profit_pct', 0) <= 0]
        
        avg_win = sum(trade.get('profit_pct', 0) for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(abs(trade.get('profit_pct', 0)) for trade in losing_trades) / len(losing_trades) if losing_trades else 1
        
        self.win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        logger.info(f"Updated performance for strategy {self.name}: Win rate = {self.win_rate:.2f}, Win/loss ratio = {self.win_loss_ratio:.2f}")
    
    def handle_exit_signal(self, symbol: str, reason: str = 'signal') -> Dict[str, Any]:
        """
        Handle exit signal for a position.
        
        Args:
            symbol: Symbol to exit
            reason: Reason for exit
            
        Returns:
            Exit decision dictionary
        """
        if symbol not in self.open_positions:
            return {
                'action': 'none',
                'reason': 'no_position',
                'symbol': symbol
            }
        
        position = self.open_positions[symbol]
        
        exit_decision = {
            'action': 'exit',
            'symbol': symbol,
            'quantity': position.get('quantity', 0),
            'entry_price': position.get('entry_price', 0),
            'entry_time': position.get('entry_time'),
            'hold_time': (datetime.now() - position.get('entry_time')).total_seconds() / 3600 if position.get('entry_time') else 0,
            'reason': reason
        }
        
        return exit_decision


class MomentumStrategy(TradingStrategy):
    """
    Momentum trading strategy.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize momentum strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Momentum-specific parameters
        self.momentum_threshold = config.get('momentum_threshold', 0.03)  # 3% momentum
        self.momentum_window = config.get('momentum_window', 20)  # 20 periods
        self.min_momentum_percentile = config.get('min_momentum_percentile', 0.8)  # Top 20%
    
    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals for momentum strategy.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with trading decisions
        """
        entry_symbols = signals.get('positions', [])
        exit_symbols = signals.get('exits', [])
        market_regime = signals.get('market_regime', 'unknown')
        
        decisions = {
            'entries': [],
            'exits': [],
            'updates': [],
            'timestamp': datetime.now().isoformat(),
            'strategy': self.name,
            'market_regime': market_regime
        }
        
        # Process exits first
        for symbol in exit_symbols:
            if symbol in self.open_positions:
                exit_decision = self.handle_exit_signal(symbol, 'model_signal')
                decisions['exits'].append(exit_decision)
        
        # Skip entries in bear market for momentum strategy
        if market_regime == 'bear':
            logger.info(f"Skipping momentum entries in bear market")
            return decisions
        
        # Process entries
        for position in entry_symbols:
            symbol = position.get('symbol')
            
            # Skip if we already have a position
            if symbol in self.open_positions:
                continue
            
            # Check if momentum is strong enough
            confidence = position.get('entry_confidence', 0.5)
            momentum = position.get('momentum', 0)
            
            entry_price = position.get('price', 0)
            if entry_price <= 0:
                continue
            
            # Calculate position size
            equity = position.get('equity', 100000)  # Default equity
            volatility = position.get('volatility', None)
            quantity = self.calculate_position_size(symbol, equity, entry_price, volatility)
            
            # Create entry decision
            if (confidence >= self.config.get('min_confidence', 0.7) and
                momentum >= self.momentum_threshold):
                
                entry_decision = {
                    'action': 'entry',
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'strategy': self.name,
                    'entry_time': datetime.now()
                }
                
                decisions['entries'].append(entry_decision)
                
                # Track position
                self.open_positions[symbol] = entry_decision
        
        return decisions


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion trading strategy.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Mean reversion-specific parameters
        self.oversold_threshold = config.get('oversold_threshold', 30)  # RSI below 30
        self.overbought_threshold = config.get('overbought_threshold', 70)  # RSI above 70
        self.mean_reversion_strength = config.get('mean_reversion_strength', 0.7)  # Strength threshold
    
    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals for mean reversion strategy.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with trading decisions
        """
        entry_symbols = signals.get('positions', [])
        exit_symbols = signals.get('exits', [])
        market_regime = signals.get('market_regime', 'unknown')
        
        decisions = {
            'entries': [],
            'exits': [],
            'updates': [],
            'timestamp': datetime.now().isoformat(),
            'strategy': self.name,
            'market_regime': market_regime
        }
        
        # Process exits first
        for symbol in exit_symbols:
            if symbol in self.open_positions:
                exit_decision = self.handle_exit_signal(symbol, 'model_signal')
                decisions['exits'].append(exit_decision)
        
        # Skip entries in highly trending markets for mean reversion
        if market_regime in ['strong_bull', 'strong_bear']:
            logger.info(f"Skipping mean reversion entries in {market_regime} market")
            return decisions
        
        # Process entries
        for position in entry_symbols:
            symbol = position.get('symbol')
            
            # Skip if we already have a position
            if symbol in self.open_positions:
                continue
            
            # Check if mean reversion signal is strong enough
            confidence = position.get('entry_confidence', 0.5)
            rsi = position.get('rsi', 50)
            bollinger_position = position.get('bollinger_position', 0.5)
            
            entry_price = position.get('price', 0)
            if entry_price <= 0:
                continue
            
            # Calculate position size
            equity = position.get('equity', 100000)  # Default equity
            volatility = position.get('volatility', None)
            quantity = self.calculate_position_size(symbol, equity, entry_price, volatility)
            
            # Create entry decision
            mean_reversion_signal = False
            
            # Long entry (oversold)
            if rsi <= self.oversold_threshold or bollinger_position <= 0.2:
                mean_reversion_signal = True
                signal_type = 'oversold'
            
            if (confidence >= self.config.get('min_confidence', 0.7) and
                mean_reversion_signal):
                
                entry_decision = {
                    'action': 'entry',
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'strategy': self.name,
                    'entry_time': datetime.now(),
                    'signal_type': signal_type
                }
                
                decisions['entries'].append(entry_decision)
                
                # Track position
                self.open_positions[symbol] = entry_decision
        
        return decisions


class TrendFollowingStrategy(TradingStrategy):
    """
    Trend following trading strategy.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize trend following strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Trend following-specific parameters
        self.trend_strength_threshold = config.get('trend_strength_threshold', 25)  # ADX above 25
        self.ma_crossover_confirm = config.get('ma_crossover_confirm', True)  # Require MA crossover
    
    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals for trend following strategy.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with trading decisions
        """
        entry_symbols = signals.get('positions', [])
        exit_symbols = signals.get('exits', [])
        market_regime = signals.get('market_regime', 'unknown')
        
        decisions = {
            'entries': [],
            'exits': [],
            'updates': [],
            'timestamp': datetime.now().isoformat(),
            'strategy': self.name,
            'market_regime': market_regime
        }
        
        # Process exits first
        for symbol in exit_symbols:
            if symbol in self.open_positions:
                exit_decision = self.handle_exit_signal(symbol, 'model_signal')
                decisions['exits'].append(exit_decision)
        
        # Skip entries in neutral market for trend following
        if market_regime == 'neutral':
            logger.info(f"Skipping trend following entries in neutral market")
            return decisions
        
        # Process entries
        for position in entry_symbols:
            symbol = position.get('symbol')
            
            # Skip if we already have a position
            if symbol in self.open_positions:
                continue
            
            # Check if trend signal is strong enough
            confidence = position.get('entry_confidence', 0.5)
            adx = position.get('adx', 0)
            
            entry_price = position.get('price', 0)
            if entry_price <= 0:
                continue
            
            # Calculate position size
            equity = position.get('equity', 100000)  # Default equity
            volatility = position.get('volatility', None)
            quantity = self.calculate_position_size(symbol, equity, entry_price, volatility)
            
            # Check trend conditions
            strong_trend = adx >= self.trend_strength_threshold
            trend_direction = position.get('trend_direction', 'unknown')
            ma_crossover = position.get('ma_crossover', False)
            
            # Create entry decision
            trend_signal = strong_trend
            if self.ma_crossover_confirm:
                trend_signal = trend_signal and ma_crossover
            
            if (confidence >= self.config.get('min_confidence', 0.7) and
                trend_signal and trend_direction == 'up'):
                
                entry_decision = {
                    'action': 'entry',
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'strategy': self.name,
                    'entry_time': datetime.now(),
                    'trend_strength': adx
                }
                
                decisions['entries'].append(entry_decision)
                
                # Track position
                self.open_positions[symbol] = entry_decision
        
        return decisions


class SentimentBasedStrategy(TradingStrategy):
    """
    Sentiment-based trading strategy.
    Uses news and social media sentiment to make trading decisions.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize sentiment-based strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        
        # Sentiment-specific parameters
        self.sentiment_threshold = config.get('sentiment_threshold', 0.5)  # Sentiment score threshold
        self.sentiment_lookback = config.get('sentiment_lookback', 3)  # Days to look back
        self.news_weight = config.get('news_weight', 0.6)  # Weight for news sentiment
        self.social_weight = config.get('social_weight', 0.4)  # Weight for social sentiment
        self.min_sentiment_magnitude = config.get('min_sentiment_magnitude', 0.2)  # Minimum sentiment magnitude
        
        # Sentiment analysis availability
        self.use_ml_sentiment = config.get('use_ml_sentiment', SENTIMENT_FEATURES_AVAILABLE)
        
        logger.info(f"Initialized sentiment-based strategy: {self.name}")
    
    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals for sentiment-based strategy.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with trading decisions
        """
        entry_symbols = signals.get('positions', [])
        exit_symbols = signals.get('exits', [])
        market_regime = signals.get('market_regime', 'unknown')
        
        decisions = {
            'entries': [],
            'exits': [],
            'updates': [],
            'timestamp': datetime.now().isoformat(),
            'strategy': self.name,
            'market_regime': market_regime
        }
        
        # Process exits first
        for symbol in exit_symbols:
            if symbol in self.open_positions:
                exit_decision = self.handle_exit_signal(symbol, 'model_signal')
                decisions['exits'].append(exit_decision)
        
        # Skip entries in bear market unless sentiment is very strong
        if market_regime == 'bear':
            logger.info(f"Being more selective with sentiment entries in bear market")
            # We'll still process entries but with higher thresholds
        
        # Process entries
        for position in entry_symbols:
            symbol = position.get('symbol')
            
            # Skip if we already have a position
            if symbol in self.open_positions:
                continue
            
            # Check sentiment signals
            news_sentiment = position.get('news_sentiment', 0)
            social_sentiment = position.get('social_sentiment', 0)
            news_magnitude = position.get('news_sentiment_magnitude', 0)
            social_magnitude = position.get('social_sentiment_magnitude', 0)
            
            # Check for ML-based sentiment if available
            ml_sentiment_score = position.get('news_ml_sentiment_score', 0)
            ml_sentiment_magnitude = position.get('news_ml_sentiment_magnitude', 0)
            
            # Calculate composite sentiment score
            if self.use_ml_sentiment and ml_sentiment_score != 0:
                # Use ML-based sentiment if available
                composite_sentiment = ml_sentiment_score
                sentiment_magnitude = ml_sentiment_magnitude
            else:
                # Use weighted average of news and social sentiment
                composite_sentiment = (
                    (news_sentiment * self.news_weight) +
                    (social_sentiment * self.social_weight)
                )
                sentiment_magnitude = (
                    (news_magnitude * self.news_weight) +
                    (social_magnitude * self.social_weight)
                )
            
            entry_price = position.get('price', 0)
            if entry_price <= 0:
                continue
            
            # Calculate position size
            equity = position.get('equity', 100000)  # Default equity
            volatility = position.get('volatility', None)
            quantity = self.calculate_position_size(symbol, equity, entry_price, volatility)
            
            # Adjust threshold based on market regime
            sentiment_threshold = self.sentiment_threshold
            if market_regime == 'bear':
                sentiment_threshold = self.sentiment_threshold * 1.5  # 50% higher threshold in bear markets
            
            # Create entry decision
            if (composite_sentiment >= sentiment_threshold and
                sentiment_magnitude >= self.min_sentiment_magnitude):
                
                entry_decision = {
                    'action': 'entry',
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'strategy': self.name,
                    'entry_time': datetime.now(),
                    'sentiment_score': composite_sentiment,
                    'sentiment_magnitude': sentiment_magnitude
                }
                
                decisions['entries'].append(entry_decision)
                
                # Track position
                self.open_positions[symbol] = entry_decision
                
                logger.info(f"Sentiment-based entry for {symbol}: score={composite_sentiment:.2f}, magnitude={sentiment_magnitude:.2f}")
        
        return decisions


class StrategyComposer:
    """
    Composes multiple trading strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize strategy composer.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Use ConfigLoader to load the strategy configuration
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        try:
            self.config = ConfigLoader.load(self.config_path)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
        
        self.strategies = {}
        
        # Initialize market regime components
        self._init_market_regime_components()
        
        # Initialize strategies
        self._init_strategies()
        
        # Connect to other components
        self.decision_framework = DecisionFramework()
        self.adaptive_thresholds = AdaptiveThresholds()
        self.order_router = OrderRouter()
        
        logger.info(f"Strategy composer initialized with {len(self.strategies)} strategies")
    
    def _init_market_regime_components(self):
        """
        Initialize market regime model and feature extractor.
        """
        try:
            # Get market regime config
            market_regime_config = self.config.get('market_regime', {})
            
            # Initialize feature extractor
            self.feature_extractor = MarketRegimeFeatures(market_regime_config)
            logger.info("Initialized market regime feature extractor")
            
            # Initialize market regime model
            self.market_regime_model = EnhancedMarketRegimeModel(market_regime_config)
            logger.info("Initialized market regime model")
            
            # Track metrics for model performance
            self.model_metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'predictions_count': 0
            }
        except Exception as e:
            logger.error(f"Error initializing market regime components: {str(e)}")
            self.feature_extractor = None
            self.market_regime_model = None
    
    def generate_signals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from raw market data.
        
        Args:
            raw_df: DataFrame with raw market data
            
        Returns:
            DataFrame with signals
        """
        if self.feature_extractor is None or self.market_regime_model is None:
            logger.error("Market regime components not initialized")
            return pd.DataFrame()
        
        try:
            logger.info(f"Generating signals from {len(raw_df)} rows of data")
            
            # Generate features
            start_time = time.time()
            features = self.feature_extractor.generate_features(raw_df)
            feature_time = time.time() - start_time
            logger.info(f"Generated {len(features.columns)} features in {feature_time:.2f} seconds")
            
            # Predict regime
            start_time = time.time()
            regime_data = self.market_regime_model.predict_regime(features)
            prediction_time = time.time() - start_time
            logger.info(f"Predicted regimes in {prediction_time:.2f} seconds")
            
            # Create signals DataFrame
            signals = pd.DataFrame({
                'timestamp': regime_data.index,
                'regime': regime_data['regime'],
                'regime_hmm': regime_data.get('regime_hmm', regime_data['regime']),
                'regime_xgb': regime_data.get('regime_xgb', regime_data['regime'])
            })
            
            # Add regime probabilities if available
            regime_names = ['trending_up', 'trending_down', 'high_volatility', 'low_volatility']
            for regime in regime_names:
                prob_col = f'prob_{regime}'
                if prob_col in regime_data.columns:
                    signals[prob_col] = regime_data[prob_col]
            
            # Log signal statistics
            regime_counts = signals['regime'].value_counts()
            logger.info(f"Signal distribution: {regime_counts.to_dict()}")
            
            # Update metrics
            self._update_model_metrics(regime_data)
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
    
    def _update_model_metrics(self, regime_data: pd.DataFrame):
        """
        Update model performance metrics.
        
        Args:
            regime_data: DataFrame with regime predictions
        """
        try:
            # Update prediction count
            self.model_metrics['predictions_count'] += len(regime_data)
            
            # If actual regimes are available, calculate accuracy metrics
            if 'actual_regime' in regime_data.columns and 'regime' in regime_data.columns:
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                
                y_true = regime_data['actual_regime']
                y_pred = regime_data['regime']
                
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted'
                )
                
                # Update metrics with exponential moving average
                alpha = 0.1  # Weight for new observations
                self.model_metrics['accuracy'] = (1 - alpha) * self.model_metrics['accuracy'] + alpha * accuracy
                self.model_metrics['precision'] = (1 - alpha) * self.model_metrics['precision'] + alpha * precision
                self.model_metrics['recall'] = (1 - alpha) * self.model_metrics['recall'] + alpha * recall
                self.model_metrics['f1_score'] = (1 - alpha) * self.model_metrics['f1_score'] + alpha * f1
                
                logger.info(f"Updated model metrics: accuracy={self.model_metrics['accuracy']:.4f}, "
                           f"f1_score={self.model_metrics['f1_score']:.4f}")
        except Exception as e:
            logger.error(f"Error updating model metrics: {str(e)}")
    
    def _init_strategies(self):
        """
        Initialize trading strategies from configuration.
        """
        strategies_config = self.config.get('strategies', {})
        
        for name, config in strategies_config.items():
            strategy_type = config.get('type', 'custom')
            
            try:
                if strategy_type == 'momentum':
                    self.strategies[name] = MomentumStrategy(name, config)
                elif strategy_type == 'mean_reversion':
                    self.strategies[name] = MeanReversionStrategy(name, config)
                elif strategy_type == 'trend_following':
                    self.strategies[name] = TrendFollowingStrategy(name, config)
                elif strategy_type == 'sentiment_based':
                    self.strategies[name] = SentimentBasedStrategy(name, config)
                else:
                    logger.warning(f"Unknown strategy type: {strategy_type}")
            except Exception as e:
                logger.error(f"Error initializing strategy {name}: {str(e)}")
    
    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals through all trading strategies.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with combined trading decisions
        """
        combined_decisions = {
            'entries': [],
            'exits': [],
            'updates': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Process signals through each strategy
        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            
            try:
                strategy_decisions = strategy.process_signals(signals)
                
                # Combine decisions
                combined_decisions['entries'].extend(strategy_decisions.get('entries', []))
                combined_decisions['exits'].extend(strategy_decisions.get('exits', []))
                combined_decisions['updates'].extend(strategy_decisions.get('updates', []))
                
                logger.debug(f"Strategy {name} generated {len(strategy_decisions.get('entries', []))} entries and {len(strategy_decisions.get('exits', []))} exits")
            
            except Exception as e:
                logger.error(f"Error processing signals for strategy {name}: {str(e)}")
        
        return combined_decisions
    
    def execute_decisions(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trading decisions.
        
        Args:
            decisions: Dictionary with trading decisions
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'orders': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute entries
        entry_orders = []
        for entry in decisions.get('entries', []):
            try:
                # Create order request
                order_request = OrderRequest(
                    symbol=entry['symbol'],
                    side=OrderSide.BUY,
                    quantity=entry['quantity'],
                    order_type=OrderType.LIMIT,
                    price=entry['entry_price'] * 1.01,  # 1% above for limit orders
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"entry_{entry['symbol']}_{int(time.time())}"
                )
                
                entry_orders.append(order_request)
            
            except Exception as e:
                logger.error(f"Error creating entry order for {entry['symbol']}: {str(e)}")
        
        # Execute exits
        exit_orders = []
        for exit_decision in decisions.get('exits', []):
            try:
                # Create order request
                order_request = OrderRequest(
                    symbol=exit_decision['symbol'],
                    side=OrderSide.SELL,
                    quantity=exit_decision['quantity'],
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"exit_{exit_decision['symbol']}_{int(time.time())}"
                )
                
                exit_orders.append(order_request)
            
            except Exception as e:
                logger.error(f"Error creating exit order for {exit_decision['symbol']}: {str(e)}")
        
        # Submit orders through order router
        if entry_orders:
            entry_results = self.order_router.submit_orders(entry_orders)
            results['entries'] = entry_results
        
        if exit_orders:
            exit_results = self.order_router.submit_orders(exit_orders)
            results['exits'] = exit_results
        
        logger.info(f"Executed {len(entry_orders)} entries and {len(exit_orders)} exits")
        
        return results
    
    def process_model_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process signals from models and execute trading decisions.
        
        Args:
            signals: Dictionary with model signals
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        # Process signals through strategies
        decisions = self.process_signals(signals)
        
        # Execute trading decisions
        results = self.execute_decisions(decisions)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate performance metrics if we have returns data
        performance_metrics = {}
        if 'returns' in signals:
            try:
                returns = signals['returns']
                performance_metrics = calculate_metrics(returns)
                logger.info(f"Strategy performance: Sharpe={performance_metrics['sharpe_ratio']:.2f}, "
                           f"Max DD={performance_metrics['max_drawdown']:.2%}")
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {str(e)}")
        
        return {
            'decisions': decisions,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'performance_metrics': performance_metrics,
            'model_metrics': self.model_metrics
        }
    
    def get_strategy(self, name: str) -> Optional[TradingStrategy]:
        """
        Get a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(name)


# Default strategy composer instance
default_strategy_composer = None


def get_strategy_composer(config_path: Optional[str] = None) -> StrategyComposer:
    """
    Get or create the default strategy composer.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        StrategyComposer instance
    """
    global default_strategy_composer
    
    if default_strategy_composer is None:
        config_path = config_path or DEFAULT_CONFIG_PATH
        logger.info(f"Creating strategy composer with config: {config_path}")
        default_strategy_composer = StrategyComposer(config_path)
    
    return default_strategy_composer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Strategy Composition')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--signals', type=str, help='Path to signals JSON file')
    parser.add_argument('--strategy', type=str, help='Strategy to use (if not using all)')
    parser.add_argument('--data', type=str, help='Path to market data CSV file')
    
    args = parser.parse_args()
    
    # Create strategy composer
    strategy_composer = StrategyComposer(args.config)
    
    if args.data:
        # Load market data and generate signals
        try:
            market_data = pd.read_csv(args.data, parse_dates=['timestamp'])
            signals_df = strategy_composer.generate_signals(market_data)
            print(f"Generated {len(signals_df)} signals")
            print(signals_df.head())
            
            # Save signals to file if needed
            output_path = args.data.replace('.csv', '_signals.csv')
            signals_df.to_csv(output_path, index=False)
            print(f"Saved signals to {output_path}")
        except Exception as e:
            print(f"Error processing market data: {str(e)}")
    
    elif args.signals:
        # Load signals from file
        with open(args.signals, 'r') as f:
            signals = json.load(f)
        
        # Process signals
        if args.strategy:
            # Use specific strategy
            strategy = strategy_composer.get_strategy(args.strategy)
            if strategy:
                decisions = strategy.process_signals(signals)
                print(json.dumps(decisions, indent=2))
            else:
                print(f"Strategy {args.strategy} not found")
        else:
            # Use all strategies
            results = strategy_composer.process_model_signals(signals)
            print(json.dumps(results, indent=2))
    else:
        parser.print_help()
