#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collection Task Module
--------------------------
Defines the DataCollectionTask for scheduling and executing data collection operations.
"""

import logging
import schedule
from typing import Dict, Any, List, Optional

from data.collectors.factory import CollectorFactory
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class DataCollectionTask:
    """
    Task to manage scheduled data collection operations.
    
    This task handles the collection of market data from various sources
    according to the configured schedule.
    """
    
    @staticmethod
    def schedule(scheduler: Any, config: Dict[str, Any], worker_pool: Any) -> None:
        """
        Schedule the data collection task based on provided configuration.
        
        Args:
            scheduler: The scheduler instance to register jobs with
            config: Configuration dictionary containing scheduling parameters
            worker_pool: Worker pool for executing tasks
        """
        sched_cfg = config.get('schedule', {})
        frequency = sched_cfg.get('frequency', 'hourly').lower()
        time_str = sched_cfg.get('time', ':00')
        
        task_instance = DataCollectionTask()
        
        if frequency == 'daily':
            scheduler.every().day.at(time_str).do(
                worker_pool.submit_task, task_instance.run, config
            )
            logger.info(f"Data collection scheduled daily at {time_str}")
        elif frequency == 'hourly':
            scheduler.every().hour.at(time_str).do(
                worker_pool.submit_task, task_instance.run, config
            )
            logger.info(f"Data collection scheduled hourly at {time_str}")
        elif frequency == 'minutes':
            interval = sched_cfg.get('interval', 5)
            scheduler.every(interval).minutes.do(
                worker_pool.submit_task, task_instance.run, config
            )
            logger.info(f"Data collection scheduled every {interval} minutes")
        elif frequency == 'seconds':
            interval = sched_cfg.get('interval', 10)
            scheduler.every(interval).seconds.do(
                worker_pool.submit_task, task_instance.run, config
            )
            logger.info(f"Data collection scheduled every {interval} seconds")
        else:
            logger.warning(f"Unsupported frequency '{frequency}' for data collection schedule")
    
    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data collection process.
        
        Args:
            config: Configuration dictionary containing data collection parameters
        
        Returns:
            Dict containing collection results and statistics
        """
        try:
            logger.info("Starting data collection task")
            
            params = config.get('params', {})
            sources = params.get('sources', [])
            asset_types = params.get('asset_types', [])
            
            results = {}
            
            for source in sources:
                try:
                    collector = CollectorFactory.create(source)
                    source_results = collector.collect(asset_types=asset_types)
                    results[source] = {
                        'status': 'success',
                        'items_collected': len(source_results),
                        'details': source_results.get('summary', {})
                    }
                    logger.info(f"Collected data from {source}: {len(source_results)} items")
                except Exception as e:
                    logger.error(f"Failed to collect data from {source}: {str(e)}", exc_info=True)
                    results[source] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Calculate overall statistics
            success_count = sum(1 for s in results.values() if s.get('status') == 'success')
            total_items = sum(s.get('items_collected', 0) for s in results.values())
            
            summary = {
                'total_sources': len(sources),
                'successful_sources': success_count,
                'failed_sources': len(sources) - success_count,
                'total_items_collected': total_items
            }
            
            logger.info(f"Data collection completed: {summary}")
            
            return {
                'summary': summary,
                'details': results
            }
            
        except Exception as e:
            logger.error(f"Data collection task failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
