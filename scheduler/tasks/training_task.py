#!/usr/bin/env python3
"""
Defines TrainingTask for scheduling and running the automated model training pipeline.
Features:
- Daily scheduled training
- GPU acceleration
- Hyperparameter optimization with Optuna
- ONNX model conversion
- MLflow tracking and model registry
"""

from utils.logging import setup_logger
import schedule
from typing import Dict, Any, List, Optional

from ml_training_engine import run_training


class TrainingTask:
    """
    Task to run automated model training daily.
    """

    @staticmethod
    def schedule(scheduler: Any, config: Dict[str, Any], worker_pool: Any) -> None:
        """
        Schedule the TrainingTask based on provided configuration.
        
        Args:
            scheduler: The scheduler instance to register jobs with
            config: Configuration dictionary containing scheduling parameters
            worker_pool: Worker pool for executing tasks
        """
        logger = setup_logger(__name__, category='models')

        # Get frequency from config (default to daily)
        frequency = config.get('frequency', 'daily').lower()
        time_str = config.get('time', '01:00')  # Default to 1 AM
        
        task_instance = TrainingTask()
        
        # Schedule based on frequency
        if frequency == 'daily':
            scheduler.every().day.at(time_str).do(
                worker_pool.submit_task, task_instance.run
            )
            logger.info(f"Scheduled daily model training at {time_str}")
        elif frequency == 'weekly':
            day = config.get('day_of_week', 'saturday').lower()
            if day == 'monday':
                scheduler.every().monday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            elif day == 'tuesday':
                scheduler.every().tuesday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            elif day == 'wednesday':
                scheduler.every().wednesday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            elif day == 'thursday':
                scheduler.every().thursday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            elif day == 'friday':
                scheduler.every().friday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            elif day == 'saturday':
                scheduler.every().saturday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            elif day == 'sunday':
                scheduler.every().sunday.at(time_str).do(
                    worker_pool.submit_task, task_instance.run
                )
            logger.info(f"Scheduled weekly model training on {day} at {time_str}")
        else:
            logger.warning(f"Unsupported frequency '{frequency}'. Defaulting to daily at 01:00")
            scheduler.every().day.at('01:00').do(
                worker_pool.submit_task, task_instance.run
            )

    def run(self):
        """
        Execute the training pipeline and return True if successful, False otherwise.
        """
        logger = setup_logger(__name__, category='models')
        try:
            logger.info("Starting automated model training")
            success = run_training()
            if success:
                logger.info("Automated model training completed successfully")
            else:
                logger.error("Automated model training completed with errors")
            return success
        except Exception as e:
            logger.error(f"Automated model training failed with error: {e}", exc_info=True)
            return False


# Module-level schedule function that delegates to the class method
def schedule(scheduler_module, config, worker_pool):
    """
    Module-level schedule function that delegates to the TrainingTask.schedule method.
    
    Args:
        scheduler_module: The schedule module
        config: Configuration dictionary
        worker_pool: Worker pool for executing tasks
    """
    return TrainingTask.schedule(scheduler_module, config, worker_pool)
