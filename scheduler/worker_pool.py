#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Worker Pool Module
-----------------
This module provides a thread pool implementation for executing tasks concurrently.
"""

import concurrent.futures
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for callable return type
T = TypeVar('T')


class WorkerPool:
    """
    A worker pool that wraps ThreadPoolExecutor to execute tasks concurrently.
    
    This class provides a simple interface for submitting tasks to a thread pool
    and handles logging of task execution events.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the worker pool with configuration.
        
        Args:
            config: Dictionary containing configuration parameters.
                   Expected to have a 'max_workers' key specifying the 
                   maximum number of worker threads.
        """
        self.max_workers = config.get('max_workers', 4)
        logger.info(f"Initializing worker pool with {self.max_workers} workers")
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

    def submit_task(self, task: Callable[..., T], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """
        Submit a task for execution in the worker pool.
        
        Args:
            task: The callable to execute
            *args: Positional arguments to pass to the task
            **kwargs: Keyword arguments to pass to the task
            
        Returns:
            A Future representing the execution of the task
        """
        task_name = getattr(task, '__name__', str(task))
        logger.info(f"Submitting task: {task_name}")
        
        # Create a wrapper function to add logging
        def task_wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting task: {task_name}")
                result = task(*args, **kwargs)
                logger.info(f"Task completed successfully: {task_name}")
                return result
            except Exception as e:
                logger.error(f"Task failed: {task_name} - {str(e)}", exc_info=True)
                raise
        
        # Submit the wrapped task
        return self.executor.submit(task_wrapper, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the worker pool.
        
        Args:
            wait: If True, wait for all pending tasks to complete before shutting down.
                 If False, cancel pending tasks and shutdown immediately.
        """
        logger.info(f"Shutting down worker pool (wait={wait})")
        self.executor.shutdown(wait=wait)
        logger.info("Worker pool shutdown complete")