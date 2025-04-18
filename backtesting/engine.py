#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtesting engine for running trading strategies on historical data.
"""
import logging
import pandas as pd

from utils.config_loader import ConfigLoader
from trading.strategy import get_strategy_composer


class BacktestingEngine:
    """
    Engine to run backtests for specified trading strategy.
    """

    def __init__(self, config_path: str):
        """
        Initialize backtesting engine.

        Args:
            config_path: Path to backtesting configuration file.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.config = ConfigLoader.load(config_path)
            self.logger.info(f"Loaded backtest config from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load backtest config: {e}")
            raise

        # Data settings
        self.data_path = self.config.get('data', {}).get('path')
        if not self.data_path:
            msg = "Data path not specified in backtest config"
            self.logger.error(msg)
            raise ValueError(msg)

        # Strategy settings
        strat_cfg = self.config.get('strategy', {})
        self.strategy_name = strat_cfg.get('name')
        if not self.strategy_name:
            msg = "Strategy name not specified in backtest config"
            self.logger.error(msg)
            raise ValueError(msg)

        # Strategy composer initialization
        strat_config_path = strat_cfg.get('config_path')
        try:
            self.strategy_composer = get_strategy_composer(strat_config_path)
            self.strategy = self.strategy_composer.get_strategy(self.strategy_name)
            if self.strategy is None:
                raise ValueError(f"Strategy '{self.strategy_name}' not found"
                                 )
            self.logger.info(f"Initialized strategy '{self.strategy_name}'")
        except Exception as e:
            self.logger.error(f"Error initializing strategy composer: {e}")
            raise

        # Placeholder for loaded data and trades
        self.data = None
        self._trade_records = []

    def load_data(self):
        """
        Load historical data for backtesting.
        """
        try:
            df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
            self.logger.info(f"Loaded {len(df)} rows of market data from {self.data_path}")
            self.data = df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def run_strategy(self):
        """
        Run the strategy over loaded data, collecting entry and exit signals.
        """
        if self.data is None:
            msg = "Data not loaded. Call load_data() before run_strategy()."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Generate regime signals DataFrame
        try:
            signals_df = self.strategy_composer.generate_signals(self.data)
            self.logger.info(f"Generated signals: {len(signals_df)} rows")
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise

        # Iterate through each signal point
        for idx, row in signals_df.iterrows():
            # Build minimal signals dict for strategy
            sig = {
                'positions': row.get('positions', []),
                'exits': row.get('exits', []),
                'market_regime': row.get('regime'),
                'timestamp': row.get('timestamp')
            }
            try:
                decisions = self.strategy.process_signals(sig)
                # record entries
                for entry in decisions.get('entries', []):
                    record = entry.copy()
                    record['type'] = 'entry'
                    record['timestamp'] = sig['timestamp']
                    self._trade_records.append(record)
                # record exits
                for exit_order in decisions.get('exits', []):
                    record = exit_order.copy()
                    record['type'] = 'exit'
                    record['timestamp'] = sig['timestamp']
                    self._trade_records.append(record)
            except Exception as e:
                self.logger.error(f"Error processing signals at index {idx}: {e}")

        self.logger.info(f"Collected {len(self._trade_records)} trade records")

    def collect_trades(self) -> pd.DataFrame:
        """
        Build a DataFrame of all collected trades.

        Returns:
            DataFrame containing trade details.
        """
        try:
            trades_df = pd.DataFrame(self._trade_records)
            self.logger.info(f"Created trades DataFrame with {len(trades_df)} rows")
            return trades_df
        except Exception as e:
            self.logger.error(f"Error creating trades DataFrame: {e}")
            raise

    def run_backtest(self) -> pd.DataFrame:
        """
        Orchestrate loading data, running strategy, and collecting trades.

        Returns:
            DataFrame of trades executed during backtest.
        """
        self.load_data()
        self.run_strategy()
        return self.collect_trades()
