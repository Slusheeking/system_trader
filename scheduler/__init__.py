"""
Scheduler package for system_trader.

This package contains modules for scheduling and executing trading tasks.
"""

from scheduler.worker_pool import WorkerPool
from scheduler.task_scheduler import discover_tasks, main

__all__ = [
    'WorkerPool',
    'discover_tasks',
    'main'
]
