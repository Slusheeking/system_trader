#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Key Manager
---------------
Secure management of API keys with rotation, validation, and error handling.
"""

import os
import yaml
import json
import logging
import time
import hashlib
import base64
from typing import Dict, Any, Optional, List, Tuple
import requests
from datetime import datetime, timedelta
from functools import wraps

from utils.logging import setup_logger
from data.database.redis_client import get_redis_client

# Setup logging with sensitive data masking
logger = setup_logger('api_key_manager', category='security')
redis_client = get_redis_client()

# Constants
DEFAULT_CONFIG_PATH = 'config/api_credentials.yaml'
KEY_CACHE_TTL = 3600  # Cache API keys for 1 hour
MAX_RETRIES = 3  # Max retries for API connections
AUTHENTICATION_TIMEOUT = 15  # Seconds before timing out auth requests
KEY_ROTATION_INTERVAL = 30  # Days before suggesting key rotation


class APIKeyException(Exception):
    """Base exception for API key errors."""
    pass


class APIKeyNotFoundError(APIKeyException):
    """Exception raised when an API key is not found."""
    pass


class APIKeyAuthenticationError(APIKeyException):
    """Exception raised when authentication with an API key fails."""
    pass


class APIKeyRateLimitError(APIKeyException):
    """Exception raised when an API rate limit is exceeded."""
    pass


class APIKeyManager:
    """
    Manager for secure API key handling, rotation, and validation.
    
    Features:
    - Secure loading from environment variables or config files
    - Key validation before use
    - Rotation tracking and suggestions
    - Rate limit tracking
    - Endpoint testing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API key manager.
        
        Args:
            config_path: Path to API credentials config file
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.credentials = {}
        self.last_loaded = 0
        self.key_usage = {}
        self._load_credentials()
        
        # Load cached validation status
        self.validation_cache = {}
        self._load_validation_cache()
    
    def _load_credentials(self) -> None:
        """Load API credentials from config file or environment variables."""
        try:
            # Only reload if file has changed or 5 minutes have passed
            if (os.path.exists(self.config_path) and 
                (os.path.getmtime(self.config_path) > self.last_loaded or 
                 time.time() - self.last_loaded > 300)):
                
                with open(self.config_path, 'r') as file:
                    self.credentials = yaml.safe_load(file) or {}
                
                self.last_loaded = time.time()
                logger.info(f"Loaded API credentials from {self.config_path}")
            
            # Check for environment variables - they override file config
            for service in list(self.credentials.keys()) + ['POLYGON', 'ALPACA', 'YAHOO', 'UNUSUAL_WHALES']:
                env_key = f"{service.upper()}_API_KEY"
                if env_key in os.environ:
                    if service not in self.credentials:
                        self.credentials[service] = {}
                    
                    self.credentials[service]['api_key'] = os.environ[env_key]
                    logger.info(f"Using {service} API key from environment variable")
                
                # Check for secondary keys/secrets
                env_secret = f"{service.upper()}_API_SECRET"
                if env_secret in os.environ:
                    if service not in self.credentials:
                        self.credentials[service] = {}
                    
                    self.credentials[service]['api_secret'] = os.environ[env_secret]
        
        except Exception as e:
            logger.error(f"Error loading API credentials: {str(e)}")
            # Continue with potentially incomplete credentials
    
    def _load_validation_cache(self) -> None:
        """Load cached API key validation results from Redis."""
        if redis_client:
            try:
                validation_data = redis_client.get('api_key_validation_cache')
                if validation_data:
                    self.validation_cache = validation_data
                    logger.debug("Loaded API key validation cache from Redis")
            except Exception as e:
                logger.warning(f"Error loading API key validation cache: {str(e)}")
    
    def _save_validation_cache(self) -> None:
        """Save API key validation results to Redis."""
        if redis_client:
            try:
                redis_client.set('api_key_validation_cache', self.validation_cache, KEY_CACHE_TTL)
                logger.debug("Saved API key validation cache to Redis")
            except Exception as e:
                logger.warning(f"Error saving API key validation cache: {str(e)}")
    
    def _mask_key(self, key: str) -> str:
        """
        Mask an API key for logging.
        
        Args:
            key: API key to mask
            
        Returns:
            Masked API key (first 4 and last 4 characters visible)
        """
        if not key or len(key) < 10:
            return "****"
        
        return f"{key[:4]}...{key[-4:]}"
    
    def _hash_key(self, key: str) -> str:
        """
        Create a unique hash of an API key for caching.
        
        Args:
            key: API key to hash
            
        Returns:
            Hash of the API key
        """
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def get_api_key(self, service: str, validate: bool = True) -> Dict[str, Any]:
        """
        Get API credentials for a service.
        
        Args:
            service: Service name (polygon, alpaca, etc.)
            validate: Whether to validate the key before returning
            
        Returns:
            Dictionary with API credentials
            
        Raises:
            APIKeyNotFoundError: If key is not found
            APIKeyAuthenticationError: If validation fails
        """
        service = service.lower()
        
        # Reload credentials if needed
        self._load_credentials()
        
        if service not in self.credentials:
            error_msg = f"No API credentials found for {service}"
            logger.error(error_msg)
            raise APIKeyNotFoundError(error_msg)
        
        creds = self.credentials[service].copy()
        
        # Track usage for rotation suggestions
        if service not in self.key_usage:
            self.key_usage[service] = {'count': 0, 'first_used': datetime.now()}
        self.key_usage[service]['count'] += 1
        self.key_usage[service]['last_used'] = datetime.now()
        
        # Check if key should be rotated
        first_used = self.key_usage[service].get('first_used')
        if first_used and (datetime.now() - first_used).days > KEY_ROTATION_INTERVAL:
            logger.warning(f"{service} API key has been used for over {KEY_ROTATION_INTERVAL} days "
                          f"and should be rotated")
        
        # Validate key if requested
        if validate:
            is_valid = self.validate_api_key(service, creds)
            if not is_valid:
                error_msg = f"API key validation failed for {service}"
                logger.error(error_msg)
                raise APIKeyAuthenticationError(error_msg)
        
        # Log masked key for debugging
        if 'api_key' in creds:
            logger.debug(f"Retrieved API key for {service}: {self._mask_key(creds['api_key'])}")
        
        return creds
    
    def validate_api_key(self, service: str, credentials: Dict[str, Any]) -> bool:
        """
        Validate API key with the service.
        
        Args:
            service: Service name
            credentials: Credentials dictionary
            
        Returns:
            Boolean indicating validity
        """
        # Check cache first to avoid unnecessary API calls
        if 'api_key' in credentials:
            key_hash = self._hash_key(credentials['api_key'])
            
            if key_hash in self.validation_cache:
                cache_entry = self.validation_cache[key_hash]
                # Only use cache if it's less than 1 hour old
                if time.time() - cache_entry.get('timestamp', 0) < KEY_CACHE_TTL:
                    logger.debug(f"Using cached validation result for {service}")
                    return cache_entry.get('valid', False)
        
        # Perform validation based on service
        is_valid = False
        validation_message = "Validation not implemented"
        
        try:
            # Polygon.io API validation
            if service == 'polygon':
                is_valid, validation_message = self._validate_polygon(credentials)
            
            # Alpaca API validation
            elif service == 'alpaca':
                is_valid, validation_message = self._validate_alpaca(credentials)
            
            # Yahoo Finance API validation
            elif service == 'yahoo':
                is_valid, validation_message = self._validate_yahoo(credentials)
            
            # Unusual Whales API validation
            elif service == 'unusual_whales':
                is_valid, validation_message = self._validate_unusual_whales(credentials)
            
            else:
                logger.warning(f"No validation method implemented for {service}")
                is_valid = True  # Assume valid if we don't know how to check
        except Exception as e:
            logger.error(f"Error validating {service} API key: {str(e)}")
            validation_message = str(e)
            is_valid = False
        
        # Cache the result
        if 'api_key' in credentials:
            key_hash = self._hash_key(credentials['api_key'])
            self.validation_cache[key_hash] = {
                'valid': is_valid,
                'timestamp': time.time(),
                'message': validation_message
            }
            self._save_validation_cache()
        
        logger.info(f"API key validation for {service}: {'SUCCESS' if is_valid else 'FAILED'}")
        return is_valid
    
    def _validate_polygon(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate Polygon.io API key.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            Tuple of (validity boolean, validation message)
        """
        if 'api_key' not in credentials:
            return False, "No API key provided"
        
        api_key = credentials['api_key']
        
        # Test with a simple endpoint
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02?apiKey={api_key}"
        
        try:
            response = requests.get(url, timeout=AUTHENTICATION_TIMEOUT)
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code == 403:
                return False, "API key is invalid"
            elif response.status_code == 429:
                return False, "Rate limit exceeded"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except requests.exceptions.Timeout:
            return False, "API request timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
    
    def _validate_alpaca(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate Alpaca API key.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            Tuple of (validity boolean, validation message)
        """
        if 'api_key' not in credentials or 'api_secret' not in credentials:
            return False, "API key or secret not provided"
        
        api_key = credentials['api_key']
        api_secret = credentials['api_secret']
        
        # Test with account endpoint
        url = "https://api.alpaca.markets/v2/account"
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=AUTHENTICATION_TIMEOUT)
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code == 401:
                return False, "API key is invalid"
            elif response.status_code == 429:
                return False, "Rate limit exceeded"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except requests.exceptions.Timeout:
            return False, "API request timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
    
    def _validate_yahoo(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate Yahoo Finance API key.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            Tuple of (validity boolean, validation message)
        """
        if 'api_key' not in credentials:
            return False, "No API key provided"
        
        api_key = credentials['api_key']
        
        # Yahoo Finance API through RapidAPI
        url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/market/v2/get-quotes"
        headers = {
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
        }
        params = {'region': 'US', 'symbols': 'AAPL'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=AUTHENTICATION_TIMEOUT)
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code in (401, 403):
                return False, "API key is invalid"
            elif response.status_code == 429:
                return False, "Rate limit exceeded"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except requests.exceptions.Timeout:
            return False, "API request timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
    
    def _validate_unusual_whales(self, credentials: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate Unusual Whales API key.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            Tuple of (validity boolean, validation message)
        """
        if 'api_key' not in credentials:
            return False, "No API key provided"
        
        api_key = credentials['api_key']
        
        # Unusual Whales doesn't have a dedicated validation endpoint,
        # so we'll use a simple endpoint to check
        url = "https://api.unusualwhales.com/api/flow/stocks"
        headers = {'Authorization': f"Bearer {api_key}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=AUTHENTICATION_TIMEOUT)
            
            if response.status_code == 200:
                return True, "API key is valid"
            elif response.status_code in (401, 403):
                return False, "API key is invalid"
            elif response.status_code == 429:
                return False, "Rate limit exceeded"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except requests.exceptions.Timeout:
            return False, "API request timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
    
    def test_all_connections(self) -> Dict[str, Any]:
        """
        Test all API connections and return results.
        
        Returns:
            Dictionary with test results for each service
        """
        results = {}
        
        for service in self.credentials:
            try:
                creds = self.credentials[service]
                is_valid, message = False, "Validation not attempted"
                
                if 'api_key' in creds:
                    is_valid, message = self.validate_api_key(service, creds)
                
                results[service] = {
                    'valid': is_valid,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                results[service] = {
                    'valid': False,
                    'message': f"Error testing connection: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def handle_rate_limit(self, service: str, retry_after: Optional[int] = None) -> None:
        """
        Handle rate limit exceeded situation.
        
        Args:
            service: Service name
            retry_after: Seconds to wait before retry (if provided by API)
        """
        # Get the default retry time if not provided
        if retry_after is None:
            # Default values based on service
            defaults = {
                'polygon': 60,   # 1 minute
                'alpaca': 300,   # 5 minutes
                'yahoo': 60,     # 1 minute 
                'unusual_whales': 600  # 10 minutes
            }
            retry_after = defaults.get(service.lower(), 60)
        
        # Store the rate limit info in Redis
        if redis_client:
            key = f"rate_limit:{service.lower()}"
            data = {
                'service': service.lower(),
                'exceeded_at': datetime.now().isoformat(),
                'retry_after': retry_after,
                'can_retry_at': (datetime.now() + timedelta(seconds=retry_after)).isoformat()
            }
            redis_client.set(key, data, retry_after + 30)  # Add buffer to expiry
        
        logger.warning(f"Rate limit exceeded for {service}. Retry after {retry_after} seconds.")
    
    def can_make_request(self, service: str) -> bool:
        """
        Check if we can make a request to the service (not rate limited).
        
        Args:
            service: Service name
            
        Returns:
            Boolean indicating if request can be made
        """
        if redis_client:
            key = f"rate_limit:{service.lower()}"
            rate_limit_info = redis_client.get(key)
            
            if rate_limit_info:
                # Still rate limited
                can_retry_at = datetime.fromisoformat(rate_limit_info['can_retry_at'])
                if datetime.now() < can_retry_at:
                    logger.info(f"Still rate limited for {service}. Can retry at {can_retry_at}")
                    return False
        
        return True
    
    def rotate_key(self, service: str, new_key: str, new_secret: Optional[str] = None) -> bool:
        """
        Rotate an API key for a service.
        
        Args:
            service: Service name
            new_key: New API key
            new_secret: New API secret (if applicable)
            
        Returns:
            Boolean indicating success
        """
        service = service.lower()
        
        try:
            # Validate the new key first
            temp_creds = {'api_key': new_key}
            if new_secret:
                temp_creds['api_secret'] = new_secret
            
            is_valid = self.validate_api_key(service, temp_creds)
            
            if not is_valid:
                logger.error(f"New {service} API key validation failed")
                return False
            
            # Update in-memory credentials
            if service not in self.credentials:
                self.credentials[service] = {}
            
            self.credentials[service]['api_key'] = new_key
            if new_secret:
                self.credentials[service]['api_secret'] = new_secret
            
            # Reset usage tracking
            self.key_usage[service] = {'count': 0, 'first_used': datetime.now()}
            
            # Try to update the config file
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as file:
                        config = yaml.safe_load(file) or {}
                    
                    if service not in config:
                        config[service] = {}
                    
                    config[service]['api_key'] = new_key
                    if new_secret:
                        config[service]['api_secret'] = new_secret
                    
                    with open(self.config_path, 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
                    
                    logger.info(f"Updated {service} API key in config file")
                except Exception as e:
                    logger.error(f"Failed to update config file: {str(e)}")
                    # Continue with in-memory update
            
            logger.info(f"Rotated API key for {service}")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating {service} API key: {str(e)}")
            return False


# Function decorators for API key handling

def with_api_key(service: str, validate: bool = True):
    """
    Decorator to inject API key into function.
    
    Args:
        service: Service name
        validate: Whether to validate the key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get API key manager
            key_manager = APIKeyManager()
            
            try:
                # Get credentials for service
                credentials = key_manager.get_api_key(service, validate)
                
                # Check for rate limiting
                if not key_manager.can_make_request(service):
                    raise APIKeyRateLimitError(f"Rate limit exceeded for {service}")
                
                # Add credentials to kwargs
                kwargs['credentials'] = credentials
                
                return func(*args, **kwargs)
                
            except APIKeyException as e:
                # Handle key-related exceptions
                logger.error(f"API key error for {service}: {str(e)}")
                raise
                
        return wrapper
    return decorator


def retry_on_rate_limit(max_retries: int = MAX_RETRIES, initial_delay: float = 1.0):
    """
    Decorator to retry on rate limit errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key_manager = APIKeyManager()
            service = getattr(args[0], 'name', 'unknown') if args else 'unknown'
            
            retries = 0
            delay = initial_delay
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except APIKeyRateLimitError as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {service}")
                        raise
                    
                    # Calculate backoff delay (with jitter)
                    delay = delay * 2 + (0.1 * (hash(str(args)) % 10))
                    
                    # Log and wait
                    logger.warning(f"Rate limit hit for {service}, retry {retries}/{max_retries} after {delay:.2f}s")
                    key_manager.handle_rate_limit(service, int(delay))
                    time.sleep(delay)
                
        return wrapper
    return decorator


# Module-level instance for convenience
_api_key_manager = None

def get_api_key_manager() -> APIKeyManager:
    """
    Get the API key manager instance.
    
    Returns:
        APIKeyManager instance
    """
    global _api_key_manager
    
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    
    return _api_key_manager


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API Key Manager")
    parser.add_argument('--validate', action='store_true', help='Validate all API keys')
    parser.add_argument('--service', type=str, help='Test specific service')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='API credentials config path')
    
    args = parser.parse_args()
    
    # Create API key manager
    manager = APIKeyManager(args.config)
    
    if args.validate:
        if args.service:
            try:
                creds = manager.get_api_key(args.service, validate=False)
                valid, message = manager.validate_api_key(args.service, creds)
                print(f"{args.service} API key validation: {'Valid' if valid else 'Invalid'}")
                print(f"Message: {message}")
            except APIKeyException as e:
                print(f"Error: {str(e)}")
        else:
            results = manager.test_all_connections()
            print("API Connection Test Results:")
            for service, result in results.items():
                print(f"{service}: {'Valid' if result['valid'] else 'Invalid'} - {result['message']}")
    else:
        # Default action - list available services
        print("Available API Services:")
        for service in manager.credentials:
            has_key = 'api_key' in manager.credentials[service]
            has_secret = 'api_secret' in manager.credentials[service]
            print(f"  - {service} ({'API Key' + (' + Secret' if has_secret else '') if has_key else 'No Key'})")