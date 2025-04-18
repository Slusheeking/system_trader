#!/usr/bin/env python3
"""
Task Scheduler
--------------
Scheduler to manage and execute system trading tasks based on configuration.
"""
import os
import sys
import time
import logging
import importlib
import schedule
from typing import Dict, Any

# Add the project root to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_loader import ConfigLoader
from scheduler.worker_pool import WorkerPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('task_scheduler')


def discover_tasks(tasks_dir: str) -> Dict[str, Any]:
    """
    Dynamically discover task modules in the tasks directory.
    
    Args:
        tasks_dir: Directory containing task modules
        
    Returns:
        Dictionary mapping task names to task modules
    """
    tasks = {}
    
    # Get the absolute path to the tasks directory
    abs_tasks_dir = os.path.abspath(tasks_dir)
    
    # Ensure the tasks directory exists
    if not os.path.exists(abs_tasks_dir):
        logger.warning(f"Tasks directory not found: {abs_tasks_dir}")
        return tasks
    
    # Get all Python files in the tasks directory
    task_files = [f for f in os.listdir(abs_tasks_dir) 
                 if f.endswith('.py') and not f.startswith('__')]
    
    for task_file in task_files:
        # Convert filename to module name
        module_name = task_file[:-3]  # Remove .py extension
        
        try:
            # Import the module
            module_path = f"scheduler.tasks.{module_name}"
            module = importlib.import_module(module_path)
            
            # Check if the module has a schedule function
            if hasattr(module, 'schedule'):
                tasks[module_name] = module
                logger.info(f"Discovered task: {module_name}")
            else:
                logger.warning(f"Module {module_name} does not have a schedule function")
        except ImportError as e:
            logger.error(f"Failed to import task module {module_name}: {e}")
    
    return tasks


def main():
    """
    Main entry point for the task scheduler.
    """
    logger.info("Starting task scheduler")
    
    try:
        # Load scheduler configuration
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'config', 'scheduler.yaml'))
        
        logger.info(f"Loading configuration from {config_path}")
        config = ConfigLoader.load(config_path)
        
        # Initialize worker pool
        worker_count = config.get('worker_pool', {}).get('max_workers', 4)
        logger.info(f"Initializing worker pool with {worker_count} workers")
        worker_pool = WorkerPool({'max_workers': worker_count})
        
        # Discover and register tasks
        tasks_dir = os.path.join(os.path.dirname(__file__), 'tasks')
        tasks = discover_tasks(tasks_dir)
        
        if not tasks:
            logger.warning("No tasks discovered")
        
        # Register tasks with the scheduler
        for task_name, task_module in tasks.items():
            if task_name in config.get('tasks', {}):
                task_config = config['tasks'][task_name]
                logger.info(f"Registering task: {task_name} with config: {task_config}")
                task_module.schedule(schedule, task_config, worker_pool)
            else:
                logger.warning(f"No configuration found for task: {task_name}")
        
        # Start the scheduler loop
        interval = config.get('scheduler', {}).get('check_interval', 60)
        logger.info(f"Starting scheduler loop with interval: {interval} seconds")
        
        while True:
            schedule.run_pending()
            time.sleep(interval)
            
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in task scheduler: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down task scheduler")


if __name__ == '__main__':
    main()
