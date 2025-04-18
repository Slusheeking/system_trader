#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtesting Task Module
-----------------------
Defines the BacktestingTask for scheduling and executing backtests.
"""

import logging
import schedule
from system_trader.backtesting.engine import BacktestingEngine

logger = logging.getLogger(__name__)

class BacktestingTask:
    @staticmethod
    def schedule(cfg: dict) -> None:
        """
        Schedule the backtesting task based on provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing scheduling parameters under 'backtesting'.
        """
        back_cfg = cfg.get('backtesting', {})
        sched = back_cfg.get('schedule', {})
        frequency = sched.get('frequency', 'daily').lower()
        time_str = sched.get('time', '00:00')

        if frequency == 'daily':
            schedule.every().day.at(time_str).do(BacktestingTask().run, cfg)
            logger.info(f"Backtesting scheduled daily at {time_str}")
        elif frequency == 'hourly':
            schedule.every().hour.at(time_str).do(BacktestingTask().run, cfg)
            logger.info(f"Backtesting scheduled hourly at {time_str}")
        elif frequency == 'weekly':
            days = sched.get('days', ['monday'])
            for day in days:
                day_lower = day.lower()
                if hasattr(schedule.every(), day_lower):
                    getattr(schedule.every(), day_lower).at(time_str).do(BacktestingTask().run, cfg)
                    logger.info(f"Backtesting scheduled weekly on {day} at {time_str}")
                else:
                    logger.warning(f"Invalid weekday '{day}', skipping schedule entry.")
        else:
            raise ValueError(f"Unsupported frequency '{frequency}' for backtesting schedule")

    def run(self, cfg: dict) -> dict:
        """
        Execute the backtesting engine and return summary metrics.

        Args:
            cfg (dict): Configuration dictionary containing 'backtesting' parameters under 'params'.

        Returns:
            dict: Summary metrics from the backtest. Empty dict on failure.
        """
        try:
            back_cfg = cfg.get('backtesting', {})
            params = back_cfg.get('params', {})
            engine = BacktestingEngine(**params)
            metrics = engine.run_backtest()
            logger.info("Backtesting completed successfully.")
            logger.info(f"Summary metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error("Backtesting failed:", exc_info=True)
            return {}
