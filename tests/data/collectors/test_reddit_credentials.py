#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Reddit credentials
Verifies if the Reddit credentials are correctly configured and working.
"""

import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reddit_test")

try:
    from data.collectors.reddit_collector import RedditCollector
    from config.collector_config import get_collector_config
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def test_reddit_authentication():
    """Test Reddit authentication."""
    try:
        # Get Reddit collector config
        logger.info("Loading Reddit collector configuration...")
        config = get_collector_config('reddit')
        
        # Print config details (without sensitive info)
        logger.info(f"Client ID: {config.client_id[:4]}...{config.client_id[-4:] if len(config.client_id) > 8 else ''}")
        logger.info(f"Username: {config.username}")
        logger.info(f"User Agent: {config.user_agent if hasattr(config, 'user_agent') else 'Not set'}")
        
        # Initialize collector
        logger.info("Initializing Reddit collector...")
        collector = RedditCollector(config)
        
        # Try to authenticate
        logger.info("Testing authentication...")
        collector._authenticate()
        
        # If we got here, authentication was successful
        logger.info("Authentication successful!")
        
        # Try a simple data collection operation
        logger.info("Testing data collection...")
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        data = collector.collect(start_time, end_time)
        
        logger.info(f"Successfully collected {len(data)} records")
        return True
        
    except Exception as e:
        logger.error(f"Error during Reddit authentication test: {str(e)}")
        return False

if __name__ == "__main__":
    if test_reddit_authentication():
        print("✅ Reddit credentials are working properly!")
        sys.exit(0)
    else:
        print("❌ Reddit credentials test failed!")
        sys.exit(1)