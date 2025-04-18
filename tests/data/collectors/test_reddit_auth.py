#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Reddit Authentication Test
Tests only the Reddit API authentication with minimal dependencies
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reddit_auth_test")

try:
    import praw
    from config.collector_config import get_collector_config
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def test_reddit_auth():
    """Test Reddit authentication directly using PRAW."""
    try:
        # Get Reddit collector config
        logger.info("Loading Reddit collector configuration...")
        config = get_collector_config('reddit')
        
        # Extract credentials
        client_id = config.client_id
        client_secret = config.client_secret
        user_agent = config.user_agent
        username = getattr(config, 'username', None)
        
        # Log credential info (without exposing sensitive data)
        logger.info(f"Client ID: {client_id[:4]}...{client_id[-4:] if len(client_id) > 8 else ''}")
        logger.info(f"Username: {username}")
        logger.info(f"User Agent: {user_agent}")
        
        # Initialize Reddit client
        logger.info("Initializing Reddit client...")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=None  # Using read-only mode
        )
        
        # Test authentication
        logger.info("Testing authentication...")
        if reddit.read_only:
            logger.info("Reddit client is in read-only mode (expected)")
        
        # Try a simple API call
        subreddit_name = "wallstreetbets"
        logger.info(f"Attempting to access r/{subreddit_name}...")
        subreddit = reddit.subreddit(subreddit_name)
        
        # Get subreddit info
        display_name = subreddit.display_name
        subscribers = subreddit.subscribers
        logger.info(f"Successfully accessed r/{display_name} with {subscribers} subscribers")
        
        # Test retrieving a single post
        logger.info("Retrieving a top post...")
        for post in subreddit.hot(limit=1):
            logger.info(f"Retrieved post: '{post.title[:50]}...' with {post.score} score")
            break
        
        logger.info("All Reddit API tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during Reddit authentication test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_reddit_auth():
        print("\n✅ Reddit authentication successful!")
        sys.exit(0)
    else:
        print("\n❌ Reddit authentication failed!")
        sys.exit(1)