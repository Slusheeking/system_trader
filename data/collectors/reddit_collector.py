#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reddit Data Collector Module
---------------------------
This module collects data from Reddit, focusing on stock-related subreddits
like WallStreetBets, stocks, investing, etc. to generate sentiment signals.
"""

import praw
import prawcore
import pandas as pd
import numpy as np
import re
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from utils.logging import setup_logger
from config.collector_config import CollectorConfig
from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord, RecordType
from data.database.timeseries_db import get_timescale_client
from data.database.redis_client import get_redis_client
from nlp.sentiment_analyzer import SentimentAnalyzer
from data.processors.data_cache import get_data_cache
from data.collectors.api_key_manager import get_api_key_manager, APIKeyAuthenticationError, APIKeyRateLimitError, retry_on_rate_limit

# Setup logging
logger = setup_logger('reddit_collector', category='data')

# Initialize shared clients
db_client = get_timescale_client()
redis_client = get_redis_client()
data_cache = get_data_cache()
api_key_manager = get_api_key_manager()


class RedditCollector(BaseCollector):
    """
    Reddit data collector for stock-related subreddits.
    
    This collector fetches posts and comments from popular stock subreddits,
    analyzes the content for sentiment and mentions of specific tickers,
    and stores the processed data for use in trading models.
    """
    
    def __init__(self, config: CollectorConfig = None):
        """
        Initialize the Reddit collector.
        
        Args:
            config: Optional CollectorConfig instance; loaded if None
        """
        if config is None:
            config = CollectorConfig.load('reddit')
        super().__init__(config)
        
        # Get credentials from API key manager
        try:
            credentials = api_key_manager.get_api_key('reddit', validate=False)
            client_id = credentials.get('client_id')
            client_secret = credentials.get('client_secret')
            username = credentials.get('username')
            password = credentials.get('password')
            user_agent = credentials.get('user_agent', 'system_trader/1.0')
            
            logger.info(f"Using Reddit API credentials from API key manager")
        except Exception as e:
            logger.warning(f"Failed to get Reddit API credentials from API key manager: {e}")
            # Fall back to config
            client_id = getattr(config, 'client_id', None)
            client_secret = getattr(config, 'client_secret', None)
            username = getattr(config, 'username', None)
            password = getattr(config, 'password', None)
            user_agent = getattr(config, 'user_agent', 'system_trader/1.0')
            
            if not client_id or not client_secret:
                logger.warning("No valid Reddit API credentials found. Read-only access may be limited.")
        
        # Initialize PRAW client
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password
        )
        
        # Subreddits to monitor
        self.subreddits = getattr(config, 'subreddits', [
            'wallstreetbets', 'stocks', 'investing', 'options',
            'SecurityAnalysis', 'StockMarket', 'pennystocks'
        ])
        
        # Limits and settings
        self.post_limit = getattr(config, 'post_limit', 100)
        self.comment_limit = getattr(config, 'comment_limit', 50)
        self.min_score = getattr(config, 'min_score', 5)
        self.include_comments = getattr(config, 'include_comments', True)
        
        # Initialize sentiment analyzer
        use_finbert = getattr(config, 'use_finbert', False)
        model_path = getattr(config, 'finbert_path', None)
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert, model_path)
        
        # Ticker regex and blacklist
        self.ticker_regex = r'\$([A-Z]{1,5})\b|\b([A-Z]{1,4})\b'  # Matches $AAPL or AAPL
        self.ticker_blacklist = set([
            'A', 'I', 'DD', 'FOR', 'BE', 'IT', 'ALL', 'ARE', 'ON', 'GO', 'NOW', 'CEO',
            'CFO', 'CTO', 'NEW', 'AM', 'PM', 'USA', 'UK', 'EU', 'IMO', 'ATH', 'IPO',
            'EPS', 'PE', 'MACD', 'RSI', 'ATM', 'OTM', 'ITM', 'YOLO', 'FOMO', 'FUD',
            'EOD', 'EST', 'PST', 'EDT', 'PDT'
        ])
        
        # Track processed items to avoid duplicates
        self.processed_items = set()
        self.processed_items_max_size = 10000  # Maximum size before clearing
        
        # Cache TTL (in seconds)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 3600)  # 1 hour
        
        # Stock symbols to track (empty = track all found)
        self.track_symbols = set(getattr(config, 'track_symbols', []))
        
        # Current collection parameters
        self._current_subreddit = None
        self._collection_start = None
        self._collection_end = None
        
        # Batch processing settings
        self.batch_size = getattr(config, 'batch_size', 100)
        self.batch_inserts = getattr(config, 'batch_inserts', True)
        self.db_batch = []
        
        # Data cleaning and validation
        self.clean_data = getattr(config, 'clean_data', True)
        self.validate_data = getattr(config, 'validate_data', True)
        
        # Rate limiting settings
        self.max_retries = getattr(config, 'max_retries', 3)
        self.retry_delay = getattr(config, 'retry_delay_seconds', 5)
        
        logger.info(f"Initialized Reddit collector for subreddits: {', '.join(self.subreddits)}")
    
    @retry_on_rate_limit(max_retries=3)
    def _authenticate(self) -> None:
        """
        Test authentication with Reddit API.
        """
        try:
            # Simple call to test credentials
            user = self.reddit.user.me()
            if user:
                logger.info(f"Successfully authenticated with Reddit API as user: {user.name}")
                # Register successful authentication with API key manager
                api_key_manager.register_api_key_success('reddit', {
                    'client_id': self.reddit.config.client_id,
                    'client_secret': self.reddit.config.client_secret
                })
            else:
                logger.info("Successfully connected to Reddit API in read-only mode")
        except prawcore.exceptions.OAuthException as e:
            # Notify API key manager of authentication failure
            api_key_manager.handle_authentication_failure('reddit', {
                'client_id': self.reddit.config.client_id,
                'client_secret': self.reddit.config.client_secret
            })
            raise CollectorError(f"Reddit authentication failed: {str(e)}")
        except prawcore.exceptions.ResponseException as e:
            if '429' in str(e):
                # Handle rate limit
                retry_after = 60  # Default to 60 seconds if not specified
                logger.warning(f"Reddit API rate limit reached: {str(e)}")
                api_key_manager.handle_rate_limit('reddit', retry_after=retry_after)
                raise APIKeyRateLimitError(f"Reddit API rate limit reached: {str(e)}", retry_after=retry_after)
            logger.warning(f"Reddit API response error during authentication: {str(e)}")
            # Continue anyway - read-only access may still work
        except Exception as e:
            logger.warning(f"Could not get user info: {str(e)}")
            # Continue anyway - read-only access may still work
    
    @retry_on_rate_limit(max_retries=3)
    def _request_page(self, page_token: Optional[str] = None) -> Any:
        """
        Request a page of data from Reddit.
        
        Args:
            page_token: Optional pagination token
            
        Returns:
            List of posts or comments
        """
        try:
            # If collecting for a specific time range, use page_token for sorting/timeframe
            sort_by = 'hot'
            time_filter = 'day'
            
            if page_token:
                # Parse page token (format: "subreddit:sorting:timeframe")
                token_parts = page_token.split(':')
                if len(token_parts) >= 3:
                    subreddit_name = token_parts[0]
                    sort_by = token_parts[1]
                    time_filter = token_parts[2]
                else:
                    subreddit_name = page_token
            else:
                # Default to first subreddit
                subreddit_name = self.subreddits[0]
            
            # Check data cache first
            cache_key = f"reddit:{subreddit_name}:{sort_by}:{time_filter}"
            cached_posts = data_cache.get_cached_data(cache_key)
            if cached_posts and len(cached_posts) > 10:  # Only use if we have a reasonable number of posts
                logger.info(f"Using {len(cached_posts)} cached posts from r/{subreddit_name}")
                return cached_posts
            
            # Store current subreddit for use in _parse
            self._current_subreddit = subreddit_name
            
            # Get subreddit
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on sorting
            if sort_by == 'hot':
                posts = list(subreddit.hot(limit=self.post_limit))
            elif sort_by == 'new':
                posts = list(subreddit.new(limit=self.post_limit))
            elif sort_by == 'top':
                posts = list(subreddit.top(time_filter=time_filter, limit=self.post_limit))
            elif sort_by == 'rising':
                posts = list(subreddit.rising(limit=self.post_limit))
            else:
                posts = list(subreddit.hot(limit=self.post_limit))
            
            # Cache the posts
            if posts:
                # We can't directly cache PRAW objects, so we'll cache their IDs
                post_ids = [post.id for post in posts]
                data_cache.set_cached_data(cache_key, post_ids, self.cache_ttl)
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name} sorted by {sort_by}")
            return posts
        
        except prawcore.exceptions.RequestException as e:
            raise CollectorError(f"Reddit API request failed: {str(e)}")
        except prawcore.exceptions.ResponseException as e:
            if '429' in str(e):
                # Handle rate limit
                retry_after = 60  # Default to 60 seconds if not specified
                logger.warning(f"Reddit API rate limit reached: {str(e)}")
                api_key_manager.handle_rate_limit('reddit', retry_after=retry_after)
                raise APIKeyRateLimitError(f"Reddit API rate limit reached: {str(e)}", retry_after=retry_after)
            raise CollectorError(f"Reddit API response error: {str(e)}")
        except prawcore.exceptions.PrawcoreException as e:
            raise CollectorError(f"PRAW core error: {str(e)}")
        except Exception as e:
            raise CollectorError(f"Error requesting Reddit data: {str(e)}")
    
    def _parse(self, raw: Any) -> Tuple[List[StandardRecord], Optional[str]]:
        """
        Parse raw Reddit posts into StandardRecord objects.
        
        Args:
            raw: Raw data from Reddit API
            
        Returns:
            Tuple of (records list, next_page_token)
        """
        posts = raw
        records = []
        
        # Keep track of unique processed tickers for this batch
        processed_tickers = set()
        
        for post in posts:
            try:
                # Skip if already processed or score too low
                if post.id in self.processed_items or post.score < self.min_score:
                    continue
                
                # Check if processed_items is getting too large
                if len(self.processed_items) > self.processed_items_max_size:
                    # Clear half of the oldest items
                    self.processed_items = set(list(self.processed_items)[self.processed_items_max_size//2:])
                
                # Add to processed items
                self.processed_items.add(post.id)
                
                # Extract tickers from title and body
                title_tickers = self._extract_tickers(post.title)
                body_tickers = self._extract_tickers(post.selftext if hasattr(post, 'selftext') else '')
                
                # Combine tickers
                all_tickers = title_tickers.union(body_tickers)
                
                # Skip if no valid tickers found and we have a tracking list
                if self.track_symbols and not any(ticker in self.track_symbols for ticker in all_tickers):
                    continue
                
                # Analyze sentiment
                text = f"{post.title} {post.selftext if hasattr(post, 'selftext') else ''}"
                sentiment = self.sentiment_analyzer.analyze(text)
                
                # Convert UTC timestamp to datetime with timezone
                post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                
                # Create records for each ticker mention
                for ticker in all_tickers:
                    # Skip if specific tracking and not in tracked symbols
                    if self.track_symbols and ticker not in self.track_symbols:
                        continue
                    
                    # Track this ticker for the subreddit
                    processed_tickers.add(ticker)
                    
                    record = StandardRecord(
                        symbol=ticker,
                        timestamp=post_time,
                        record_type=RecordType.SOCIAL,
                        source='reddit',
                        sentiment_score=sentiment['score'],
                        sentiment_magnitude=sentiment['magnitude'],
                        extended_data={
                            'id': post.id,
                            'title': post.title,
                            'body': post.selftext if hasattr(post, 'selftext') else '',
                            'author': str(post.author) if post.author else '[deleted]',
                            'url': post.url,
                            'permalink': post.permalink,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio if hasattr(post, 'upvote_ratio') else 0,
                            'num_comments': post.num_comments,
                            'subreddit': post.subreddit.display_name,
                            'flair': post.link_flair_text if hasattr(post, 'link_flair_text') else None,
                            'post_type': 'submission',
                            'tickers': list(all_tickers),
                            'sentiment': sentiment
                        }
                    )
                    
                    # Clean data if enabled
                    if self.clean_data and hasattr(data_cache, 'get_data_cleaner'):
                        cleaner = data_cache.get_data_cleaner()
                        record_dict = record.model_dump()
                        clean_data = cleaner.clean_social_data(record_dict, 'reddit', ticker)
                        
                        # Update record with cleaned data
                        for key, value in clean_data.items():
                            if hasattr(record, key):
                                setattr(record, key, value)
                    
                    # Validate data if enabled
                    if self.validate_data and hasattr(data_cache, 'get_data_cleaner'):
                        cleaner = data_cache.get_data_cleaner()
                        if not cleaner.validate_social_data(record.model_dump(), 'reddit'):
                            logger.warning(f"Skipping invalid Reddit data for {ticker}")
                            continue
                    
                    records.append(record)
                    
                    # Add record to data cache
                    data_cache.add_social_data(ticker, record.model_dump())
                    
                    # Add to database batch
                    if self.batch_inserts:
                        self._add_to_db_batch(record)
                
                # Process comments if enabled
                if self.include_comments:
                    comment_records = self._process_comments(post, all_tickers)
                    records.extend(comment_records)
                    
                    # Add comment records to data cache and DB batch
                    for comment_record in comment_records:
                        ticker = comment_record.symbol
                        data_cache.add_social_data(ticker, comment_record.model_dump())
                        
                        if self.batch_inserts:
                            self._add_to_db_batch(comment_record)
            
            except Exception as e:
                logger.error(f"Error processing Reddit post {post.id}: {str(e)}")
        
        # Determine next page token (move to next subreddit)
        next_token = None
        if self._current_subreddit:
            # Find current subreddit index
            try:
                current_idx = self.subreddits.index(self._current_subreddit)
                # Move to next subreddit if available
                if current_idx < len(self.subreddits) - 1:
                    next_subreddit = self.subreddits[current_idx + 1]
                    next_token = f"{next_subreddit}:hot:day"
            except ValueError:
                # If current subreddit not in list, start from beginning
                if self.subreddits:
                    next_token = f"{self.subreddits[0]}:hot:day"
        
        # Cache sentiment data for processed tickers
        for ticker in processed_tickers:
            ticker_records = [r for r in records if r.symbol == ticker]
            if ticker_records:
                self._cache_sentiment_data(ticker, ticker_records)
        
        # Flush the database batch if we've completed all subreddits
        if self.batch_inserts and (not next_token or next_token.startswith(self.subreddits[0])):
            self._flush_db_batch()
        
        logger.info(f"Parsed {len(records)} records from Reddit posts")
        return records, next_token
    
    def _add_to_db_batch(self, record: StandardRecord) -> None:
        """
        Add a record to the database batch.
        
        Args:
            record: StandardRecord to add to batch
        """
        record_dict = record.model_dump()
        extended_data = record_dict.get('extended_data', {})
        
        db_record = {
            'time': record_dict.get('timestamp'),
            'symbol': record_dict.get('symbol'),
            'source': 'reddit',
            'platform': 'reddit',
            'subreddit': extended_data.get('subreddit'),
            'post_id': extended_data.get('id'),
            'parent_id': extended_data.get('post_id'),
            'author': extended_data.get('author'),
            'content_type': extended_data.get('post_type', 'unknown'),
            'sentiment_score': record_dict.get('sentiment_score'),
            'sentiment_magnitude': record_dict.get('sentiment_magnitude'),
            'score': extended_data.get('score', 0),
            'metadata': extended_data
        }
        
        self.db_batch.append(db_record)
        
        # Flush batch if it reaches the batch size
        if len(self.db_batch) >= self.batch_size:
            self._flush_db_batch()
    
    def _flush_db_batch(self) -> None:
        """Flush the database batch."""
        if not self.db_batch:
            return
            
        try:
            count = db_client.insert_social_data(self.db_batch)
            logger.info(f"Stored {count} Reddit records in TimescaleDB (batch size: {len(self.db_batch)})")
            self.db_batch = []
        except Exception as e:
            logger.error(f"Error storing Reddit batch in TimescaleDB: {str(e)}")
    
    def _process_comments(self, post: Any, post_tickers: Set[str]) -> List[StandardRecord]:
        """
        Process comments for a post.
        
        Args:
            post: Reddit post
            post_tickers: Tickers already found in the post
            
        Returns:
            List of StandardRecord objects for comments
        """
        records = []
        
        try:
            # Get top comments
            post.comments.replace_more(limit=0)  # Skip "more comments" items
            comments = list(post.comments.list())[:self.comment_limit]
            
            for comment in comments:
                try:
                    # Skip if already processed or score too low
                    if comment.id in self.processed_items or comment.score < self.min_score:
                        continue
                    
                    # Add to processed items
                    self.processed_items.add(comment.id)
                    
                    # Extract tickers from comment
                    comment_tickers = self._extract_tickers(comment.body if hasattr(comment, 'body') else '')
                    
                    # Combine with post tickers
                    all_tickers = comment_tickers.union(post_tickers)
                    
                    # Skip if no valid tickers found and we have a tracking list
                    if self.track_symbols and not any(ticker in self.track_symbols for ticker in all_tickers):
                        continue
                    
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer.analyze(comment.body if hasattr(comment, 'body') else '')
                    
                    # Comment time with timezone
                    comment_time = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
                    
                    # Create records for each ticker mention
                    for ticker in all_tickers:
                        # Skip if specific tracking and not in tracked symbols
                        if self.track_symbols and ticker not in self.track_symbols:
                            continue
                        
                        record = StandardRecord(
                            symbol=ticker,
                            timestamp=comment_time,
                            record_type=RecordType.SOCIAL,
                            source='reddit',
                            sentiment_score=sentiment['score'],
                            sentiment_magnitude=sentiment['magnitude'],
                            extended_data={
                                'id': comment.id,
                                'body': comment.body if hasattr(comment, 'body') else '',
                                'author': str(comment.author) if comment.author else '[deleted]',
                                'score': comment.score,
                                'permalink': comment.permalink if hasattr(comment, 'permalink') else None,
                                'subreddit': comment.subreddit.display_name,
                                'post_id': post.id,
                                'post_title': post.title,
                                'post_type': 'comment',
                                'tickers': list(all_tickers),
                                'sentiment': sentiment,
                                'in_post': ticker in post_tickers,
                                'in_comment': ticker in comment_tickers
                            }
                        )
                        
                        # Clean data if enabled
                        if self.clean_data and hasattr(data_cache, 'get_data_cleaner'):
                            cleaner = data_cache.get_data_cleaner()
                            record_dict = record.model_dump()
                            clean_data = cleaner.clean_social_data(record_dict, 'reddit', ticker)
                            
                            # Update record with cleaned data
                            for key, value in clean_data.items():
                                if hasattr(record, key):
                                    setattr(record, key, value)
                        
                        # Validate data if enabled
                        if self.validate_data and hasattr(data_cache, 'get_data_cleaner'):
                            cleaner = data_cache.get_data_cleaner()
                            if not cleaner.validate_social_data(record.model_dump(), 'reddit'):
                                logger.warning(f"Skipping invalid Reddit comment data for {ticker}")
                                continue
                                
                        records.append(record)
                
                except Exception as e:
                    logger.error(f"Error processing Reddit comment {comment.id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing comments for post {post.id}: {str(e)}")
        
        return records
    
    def _extract_tickers(self, text: str) -> Set[str]:
        """
        Extract stock tickers from text.
        
        Args:
            text: Text to extract tickers from
            
        Returns:
            Set of tickers found in text
        """
        if not text:
            return set()
        
        # Find all potential tickers
        matches = re.findall(self.ticker_regex, text)
        
        # Flatten matches (regex groups) and filter blacklisted terms
        tickers = set()
        for match in matches:
            ticker = next((t for t in match if t), None)
            if ticker and ticker not in self.ticker_blacklist:
                tickers.add(ticker)
        
        return tickers
    
    @retry_on_rate_limit(max_retries=3)
    def collect(self, start: datetime, end: datetime, page_token: Optional[str] = None) -> List[StandardRecord]:
        """
        Collect data for all subreddits between start and end datetimes.
        
        Args:
            start: Start datetime
            end: End datetime
            page_token: Optional starting page token
            
        Returns:
            List of StandardRecord objects
        """
        # Store the collection time range
        self._collection_start = start
        self._collection_end = end
        
        all_records = []
        
        # If page_token is provided, use it directly
        if page_token:
            try:
                records = super().collect(start, end, page_token)
                all_records.extend(records)
                return all_records
            except Exception as e:
                logger.error(f"Error collecting with token {page_token}: {str(e)}")
                # Continue with normal collection
        
        # Reset database batch
        self.db_batch = []
        
        # Process each subreddit
        for subreddit in self.subreddits:
            logger.info(f"Collecting Reddit data from r/{subreddit}")
            
            # Set page token to current subreddit with sorting and time filter
            page_token = f"{subreddit}:hot:day"
            
            # Collect for this subreddit
            try:
                records = super().collect(start, end, page_token)
                all_records.extend(records)
                
                logger.info(f"Collected {len(records)} records from r/{subreddit}")
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit}: {str(e)}")
                continue
        
        # Flush any remaining database records
        if self.batch_inserts and self.db_batch:
            self._flush_db_batch()
        
        return all_records
    
    def _store_records_in_db(self, records: List[StandardRecord]) -> None:
        """
        Store records in database and cache.
        
        Args:
            records: List of StandardRecord objects
        """
        if not records:
            return
            
        try:
            # Group records by ticker
            ticker_data = {}
            
            for record in records:
                symbol = record.symbol
                if symbol not in ticker_data:
                    ticker_data[symbol] = []
                ticker_data[symbol].append(record)
            
            # Store in TimescaleDB using social_data table
            db_records = []
            
            for symbol, ticker_records in ticker_data.items():
                # Convert records to DB format
                for record in ticker_records:
                    record_dict = record.model_dump()
                    extended_data = record_dict.get('extended_data', {})
                    
                    db_record = {
                        'time': record_dict.get('timestamp'),
                        'symbol': record_dict.get('symbol'),
                        'source': 'reddit',
                        'platform': 'reddit',
                        'subreddit': extended_data.get('subreddit'),
                        'post_id': extended_data.get('id'),
                        'parent_id': extended_data.get('post_id'),
                        'author': extended_data.get('author'),
                        'content_type': extended_data.get('post_type', 'unknown'),
                        'sentiment_score': record_dict.get('sentiment_score'),
                        'sentiment_magnitude': record_dict.get('sentiment_magnitude'),
                        'score': extended_data.get('score', 0),
                        'metadata': extended_data
                    }
                    
                    db_records.append(db_record)
                    
                    # Add to data cache
                    data_cache.add_social_data(symbol, record_dict)
            
            # Insert in batches to avoid large queries
            batch_size = self.batch_size
            for i in range(0, len(db_records), batch_size):
                batch = db_records[i:i+batch_size]
                try:
                    count = db_client.insert_social_data(batch)
                    logger.info(f"Stored {count} Reddit records in TimescaleDB (batch {i//batch_size + 1})")
                except Exception as e:
                    logger.error(f"Error storing Reddit batch in TimescaleDB: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in _store_records_in_db: {str(e)}")
    
    def _cache_sentiment_data(self, symbol: str, records: List[StandardRecord]) -> None:
        """
        Cache aggregate sentiment data in Redis.
        
        Args:
            symbol: Ticker symbol
            records: List of StandardRecord objects for this symbol
        """
        if not records:
            return
            
        try:
            # Calculate aggregate sentiment metrics
            sentiment_scores = [r.sentiment_score for r in records if hasattr(r, 'sentiment_score') and r.sentiment_score is not None]
            
            if not sentiment_scores:
                return
            
            # Calculate statistics
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Calculate weighted sentiment (by score/upvotes)
            weighted_scores = []
            weights = []
            
            for r in records:
                if hasattr(r, 'sentiment_score') and r.sentiment_score is not None and hasattr(r, 'extended_data'):
                    score = r.extended_data.get('score', 1)
                    if score <= 0:
                        score = 1  # Minimum weight = 1
                    weighted_scores.append(r.sentiment_score * score)
                    weights.append(score)
            
            weighted_sum = sum(weighted_scores)
            weight_sum = sum(weights)
            weighted_avg = weighted_sum / weight_sum if weight_sum > 0 else avg_sentiment
            
            # Calculate mention count and score
            mention_count = len(records)
            total_score = sum(r.extended_data.get('score', 0) for r in records if hasattr(r, 'extended_data'))
            
            # Number of bullish vs bearish mentions
            bullish_count = sum(1 for r in records if hasattr(r, 'sentiment_score') and r.sentiment_score is not None and r.sentiment_score > 0.05)
            bearish_count = sum(1 for r in records if hasattr(r, 'sentiment_score') and r.sentiment_score is not None and r.sentiment_score < -0.05)
            neutral_count = mention_count - bullish_count - bearish_count
            
            # Calculate bull/bear ratio
            bull_bear_ratio = bullish_count / bearish_count if bearish_count > 0 else float('inf')
            
            # Count by subreddit
            subreddit_counts = {}
            for r in records:
                if hasattr(r, 'extended_data'):
                    subreddit = r.extended_data.get('subreddit')
                    if subreddit:
                        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
            
            # Create cache data
            cache_data = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'avg_sentiment': float(avg_sentiment),
                'weighted_avg_sentiment': float(weighted_avg),
                'mention_count': mention_count,
                'total_score': total_score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'bull_bear_ratio': float(bull_bear_ratio) if bull_bear_ratio != float('inf') else 999.99,
                'subreddit_counts': subreddit_counts,
                'source': 'reddit',
                'collection_start': self._collection_start.isoformat() if self._collection_start else None,
                'collection_end': self._collection_end.isoformat() if self._collection_end else None
            }
            
            # Store in data_cache
            data_cache.set_social_sentiment(symbol, cache_data)
            
            # Store in Redis for backward compatibility
            cache_key = f"reddit:sentiment:{symbol}"
            redis_client.set(cache_key, cache_data, self.cache_ttl)
            
            # Also store in time series list
            timeseries_key = f"reddit:sentiment:{symbol}:history"
            redis_client.lpush(timeseries_key, cache_data)
            redis_client.ltrim(timeseries_key, 0, 99)  # Keep last 100 entries
            redis_client.expire(timeseries_key, self.cache_ttl * 24)  # 24x longer expiry for history
            
            # Store in a global list of all recent sentiment data
            global_key = "reddit:sentiment:recent"
            sentiment_entry = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'sentiment': weighted_avg,
                'mentions': mention_count,
                'bullish': bullish_count,
                'bearish': bearish_count
            }
            
            # Store in data_cache global list
            data_cache.add_global_sentiment(sentiment_entry)
            
            # Store in Redis for backward compatibility
            redis_client.lpush(global_key, sentiment_entry)
            redis_client.ltrim(global_key, 0, 499)  # Keep last 500 entries
            redis_client.expire(global_key, self.cache_ttl * 12)  # 12x longer expiry
            
            logger.debug(f"Cached Reddit sentiment data for {symbol} in data cache and Redis")
        
        except Exception as e:
            logger.error(f"Error caching Reddit sentiment data: {str(e)}")

    def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get cached sentiment data for a symbol.
        
        Args:
            symbol: Symbol to get sentiment for
            
        Returns:
            Sentiment data dictionary or empty dict if not found
        """
        try:
            # Get from data_cache first
            sentiment_data = data_cache.get_social_sentiment(symbol)
            if sentiment_data:
                return sentiment_data
                
            # Fall back to Redis for backward compatibility
            cache_key = f"reddit:sentiment:{symbol}"
            data = redis_client.get(cache_key)
            
            if data:
                return data
            
            # Return empty dict if not found
            return {
                'symbol': symbol,
                'avg_sentiment': 0.0,
                'mention_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'source': 'reddit'
            }
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

    def get_sentiment_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get cached sentiment history for a symbol.
        
        Args:
            symbol: Symbol to get sentiment history for
            limit: Maximum number of records to return
            
        Returns:
            List of sentiment data dictionaries
        """
        try:
            # Get from data_cache first
            history = data_cache.get_social_sentiment_history(symbol, limit)
            if history:
                return history
                
            # Fall back to Redis for backward compatibility
            key = f"reddit:sentiment:{symbol}:history"
            data = redis_client.lrange(key, 0, limit - 1)
            
            return data or []
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment history for {symbol}: {str(e)}")
            return []
    
    @retry_on_rate_limit(max_retries=3)
    def get_recent_mentions(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent mentions of a symbol or all symbols.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of recent mentions
        """
        try:
            # Get from data_cache first
            mentions = data_cache.get_recent_social_data(symbol, 'reddit', limit) if symbol else data_cache.get_all_recent_social_data('reddit', limit)
            if mentions:
                return mentions
                
            # Fall back to database query
            if symbol:
                # Query timescale for specific symbol
                query = """
                SELECT time, symbol, content_type, author, sentiment_score, score, 
                       metadata->'body' as content, metadata->'post_title' as post_title, 
                       metadata->'permalink' as permalink, metadata->'subreddit' as subreddit
                FROM social_data 
                WHERE symbol = %s AND source = 'reddit'
                ORDER BY time DESC
                LIMIT %s
                """
                
                results = db_client.execute_query(query, (symbol, limit))
                
                # Format results
                mentions = []
                for row in results:
                    mentions.append({
                        'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else row['time'],
                        'symbol': row['symbol'],
                        'content_type': row['content_type'],
                        'author': row['author'],
                        'sentiment_score': float(row['sentiment_score']) if row['sentiment_score'] is not None else 0.0,
                        'score': row['score'],
                        'content': row['content'],
                        'post_title': row['post_title'],
                        'permalink': row['permalink'],
                        'subreddit': row['subreddit']
                    })
                
                return mentions
            else:
                # Query timescale for all symbols
                query = """
                SELECT time, symbol, content_type, author, sentiment_score, score, 
                       metadata->'body' as content, metadata->'post_title' as post_title, 
                       metadata->'permalink' as permalink, metadata->'subreddit' as subreddit
                FROM social_data 
                WHERE source = 'reddit'
                ORDER BY time DESC
                LIMIT %s
                """
                
                results = db_client.execute_query(query, (limit,))
                
                # Format results
                mentions = []
                for row in results:
                    mentions.append({
                        'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else row['time'],
                        'symbol': row['symbol'],
                        'content_type': row['content_type'],
                        'author': row['author'],
                        'sentiment_score': float(row['sentiment_score']) if row['sentiment_score'] is not None else 0.0,
                        'score': row['score'],
                        'content': row['content'],
                        'post_title': row['post_title'],
                        'permalink': row['permalink'],
                        'subreddit': row['subreddit']
                    })
                
                return mentions
        except Exception as e:
            logger.error(f"Error getting recent Reddit mentions: {str(e)}")
            return []
    
    @retry_on_rate_limit(max_retries=3)
    def get_trending_symbols(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get trending symbols by number of mentions and sentiment.
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of trending symbols with metrics
        """
        try:
            # Get from data_cache first
            trending = data_cache.get_trending_symbols('reddit', limit)
            if trending:
                return trending
                
            # Fall back to database query
            query = """
            SELECT symbol, 
                   COUNT(*) as mention_count,
                   AVG(sentiment_score) as avg_sentiment,
                   SUM(CASE WHEN sentiment_score > 0.05 THEN 1 ELSE 0 END) as bullish_count,
                   SUM(CASE WHEN sentiment_score < -0.05 THEN 1 ELSE 0 END) as bearish_count
            FROM social_data 
            WHERE source = 'reddit' 
              AND time > NOW() - INTERVAL '24 hours'
            GROUP BY symbol
            ORDER BY mention_count DESC
            LIMIT %s
            """
            
            results = db_client.execute_query(query, (limit,))
            
            # Format results
            trending = []
            for row in results:
                trending.append({
                    'symbol': row['symbol'],
                    'mention_count': row['mention_count'],
                    'avg_sentiment': float(row['avg_sentiment']) if row['avg_sentiment'] is not None else 0.0,
                    'bullish_count': row['bullish_count'],
                    'bearish_count': row['bearish_count'],
                    'bull_bear_ratio': row['bullish_count'] / row['bearish_count'] if row['bearish_count'] > 0 else float('inf')
                })
            
            # Cache the results
            if trending:
                data_cache.set_trending_symbols('reddit', trending)
            
            return trending
        except Exception as e:
            logger.error(f"Error getting trending Reddit symbols: {str(e)}")
            return []


# Factory function for creating an instance
def create_reddit_collector(config: Optional[CollectorConfig] = None) -> RedditCollector:
    """Create an instance of the Reddit collector."""
    return RedditCollector(config)


if __name__ == "__main__":
    import logging
    from datetime import timedelta
    
    # Set logging level for testing
    logger.setLevel(logging.INFO)
    
    # Create collector
    collector = RedditCollector()
    
    # Define date range for collection
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)  # Last 24 hours
    
    # Collect data
    records = collector.collect(start, end)
    print(f"Collected {len(records)} Reddit records")
    
    # Get trending symbols
    trending = collector.get_trending_symbols(10)
    print("\nTrending symbols:")
    for symbol in trending:
        print(f"{symbol['symbol']}: {symbol['mention_count']} mentions, {symbol['avg_sentiment']:.2f} sentiment")
    
    # Get sentiment for a specific symbol
    if trending:
        top_symbol = trending[0]['symbol']
        sentiment = collector.get_sentiment(top_symbol)
        print(f"\nSentiment for {top_symbol}: {sentiment['avg_sentiment']:.2f}")
