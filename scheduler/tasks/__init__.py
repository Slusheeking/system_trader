"""
Tasks module for the scheduler package.

This module contains task definitions for scheduled operations.
"""

from scheduler.tasks.data_collection_task import DataCollectionTask
from scheduler.tasks.market_analysis_task import MarketAnalysisTask
from scheduler.tasks.backtesting_task import BacktestingTask
from scheduler.tasks.training_task import TrainingTask
from scheduler.tasks.notification_task import NotificationTask

__all__ = [
    'DataCollectionTask',
    'MarketAnalysisTask',
    'BacktestingTask',
    'TrainingTask',
    'NotificationTask'
]
