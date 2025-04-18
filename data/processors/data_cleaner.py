#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Cleaner Module
-----------------
Provides data cleaning and preprocessing functionality for market data.
Handles missing values, outliers, data normalization, and specialized
cleaning for both WebSocket real-time data and REST API historical data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import threading

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('data_cleaner', category='data')


class DataCleaner:
    """
    Data cleaning and preprocessing for market data.
    Provides methods for handling missing values, outliers, normalization,
    and specialized cleaning for WebSocket and REST API data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure cleaning options
        self.handle_missing = self.config.get('handle_missing', True)
        self.handle_outliers = self.config.get('handle_outliers', True)
        self.normalize_data = self.config.get('normalize_data', False)
        
        # Configure missing value strategy
        self.missing_strategy = self.config.get('missing_strategy', 'ffill')
        self.max_gap_size = self.config.get('max_gap_size', 5)
        
        # Configure outlier detection
        self.outlier_method = self.config.get('outlier_method', 'zscore')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        
        # Configure normalization
        self.norm_method = self.config.get('norm_method', 'minmax')
        
        # WebSocket specific configurations
        self.ws_max_deviation = self.config.get('ws_max_deviation', 0.1)  # 10% deviation threshold
        self.ws_smooth_window = self.config.get('ws_smooth_window', 3)    # Smoothing window size
        self.ws_enable_smoothing = self.config.get('ws_enable_smoothing', True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # State for real-time data processing
        self._last_values = {}  # Symbol -> Dict of last known values by field
        self._missing_count = {}  # Symbol -> Dict of consecutive missing values by field
        
        logger.info("DataCleaner initialized")
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            logger.warning("Empty DataFrame provided, nothing to clean")
            return data
        
        logger.info(f"Cleaning data with shape {data.shape}")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Process each symbol separately if 'symbol' column exists
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            result_dfs = []
            
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                
                # Sort by timestamp to ensure correct processing
                if 'timestamp' in symbol_data.columns:
                    symbol_data = symbol_data.sort_values('timestamp')
                
                # Apply cleaning steps
                symbol_data = self._clean_single_series(symbol_data)
                
                result_dfs.append(symbol_data)
            
            # Combine results
            result = pd.concat(result_dfs, ignore_index=True)
        else:
            # Sort by timestamp if available
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Apply cleaning steps
            result = self._clean_single_series(df)
        
        logger.info(f"Cleaning complete, output shape: {result.shape}")
        
        return result
    
    def _clean_single_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a single time series (one symbol).
        
        Args:
            data: DataFrame for a single symbol
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Handle missing values
        if self.handle_missing:
            df = self._handle_missing_values(df)
        
        # Handle outliers
        if self.handle_outliers:
            df = self._handle_outliers(df)
        
        # Normalize data
        if self.normalize_data:
            df = self._normalize_data(df)
        
        return df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with potentially missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df = data.copy()
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            logger.debug("No missing values found")
            return df
        
        logger.info(f"Handling {missing_count} missing values using strategy: {self.missing_strategy}")
        
        # Price columns that should be handled together
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        # Handle missing values based on strategy
        if self.missing_strategy == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
            
        elif self.missing_strategy == 'ffill':
            # Forward fill (use previous value)
            df = df.fillna(method='ffill', limit=self.max_gap_size)
            # If any NaNs remain at the beginning, backward fill
            df = df.fillna(method='bfill')
            
        elif self.missing_strategy == 'linear':
            # Linear interpolation
            df = df.interpolate(method='linear', limit=self.max_gap_size, limit_direction='both')
            
        elif self.missing_strategy == 'time':
            # Time-based interpolation if timestamp is available
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                df = df.interpolate(method='time', limit=self.max_gap_size, limit_direction='both')
                df = df.reset_index()
            else:
                # Fall back to linear interpolation
                df = df.interpolate(method='linear', limit=self.max_gap_size, limit_direction='both')
        
        # Special handling for OHLC data to maintain integrity
        if len(available_price_cols) >= 2:
            # Ensure high is the highest value
            if 'high' in df.columns:
                for col in available_price_cols:
                    if col != 'high':
                        df['high'] = np.maximum(df['high'], df[col])
            
            # Ensure low is the lowest value
            if 'low' in df.columns:
                for col in available_price_cols:
                    if col != 'low':
                        df['low'] = np.minimum(df['low'], df[col])
        
        # Fill any remaining NaNs with column means (only for numeric columns)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean(numeric_only=True))
        
        # If any NaNs still remain, fill with zeros
        df = df.fillna(0)
        
        return df
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in the data.
        
        Args:
            data: DataFrame with potential outliers
            
        Returns:
            DataFrame with outliers handled
        """
        df = data.copy()
        
        # Columns to check for outliers (numeric columns only)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Exclude certain columns from outlier detection
        exclude_cols = ['timestamp', 'symbol', 'volume']
        outlier_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not outlier_cols:
            logger.debug("No columns available for outlier detection")
            return df
        
        logger.info(f"Checking for outliers in {len(outlier_cols)} columns using method: {self.outlier_method}")
        
        # Detect and handle outliers based on method
        if self.outlier_method == 'zscore':
            # Z-score method
            for col in outlier_cols:
                # Calculate z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                
                # Identify outliers
                outliers = z_scores > self.outlier_threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.debug(f"Found {outlier_count} outliers in column {col}")
                    
                    # Replace outliers with column median
                    df.loc[outliers, col] = df[col].median()
        
        elif self.outlier_method == 'iqr':
            # Interquartile Range method
            for col in outlier_cols:
                # Calculate IQR
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Define bounds
                lower_bound = q1 - (self.outlier_threshold * iqr)
                upper_bound = q3 + (self.outlier_threshold * iqr)
                
                # Identify outliers
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.debug(f"Found {outlier_count} outliers in column {col}")
                    
                    # Replace outliers with bounds
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
        
        elif self.outlier_method == 'winsorize':
            # Winsorization method
            for col in outlier_cols:
                # Calculate percentiles
                lower_percentile = 0.01
                upper_percentile = 0.99
                
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)
                
                # Identify outliers
                outliers_lower = df[col] < lower_bound
                outliers_upper = df[col] > upper_bound
                outlier_count = outliers_lower.sum() + outliers_upper.sum()
                
                if outlier_count > 0:
                    logger.debug(f"Found {outlier_count} outliers in column {col}")
                    
                    # Replace outliers with bounds
                    df.loc[outliers_lower, col] = lower_bound
                    df.loc[outliers_upper, col] = upper_bound
        
        return df
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric data columns.
        
        Args:
            data: DataFrame with data to normalize
            
        Returns:
            DataFrame with normalized data
        """
        df = data.copy()
        
        # Columns to normalize (numeric columns only)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Exclude certain columns from normalization
        exclude_cols = ['timestamp', 'symbol', 'volume']
        norm_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not norm_cols:
            logger.debug("No columns available for normalization")
            return df
        
        logger.info(f"Normalizing {len(norm_cols)} columns using method: {self.norm_method}")
        
        # Normalize data based on method
        if self.norm_method == 'minmax':
            # Min-Max normalization
            for col in norm_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.5  # Default value if all values are the same
        
        elif self.norm_method == 'zscore':
            # Z-score normalization
            for col in norm_cols:
                mean = df[col].mean()
                std = df[col].std()
                
                # Avoid division by zero
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0  # Default value if all values are the same
        
        elif self.norm_method == 'robust':
            # Robust scaling using median and IQR
            for col in norm_cols:
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Avoid division by zero
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr
                else:
                    df[col] = 0  # Default value if all values are the same
        
        return df
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the data.
        
        Args:
            data: DataFrame with potential duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        df = data.copy()
        
        # Check for duplicates
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            # Check for duplicates based on timestamp and symbol
            dup_count = df.duplicated(subset=['timestamp', 'symbol']).sum()
            
            if dup_count > 0:
                logger.info(f"Removing {dup_count} duplicate rows")
                df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        else:
            # Check for duplicates across all columns
            dup_count = df.duplicated().sum()
            
            if dup_count > 0:
                logger.info(f"Removing {dup_count} duplicate rows")
                df = df.drop_duplicates(keep='first')
        
        return df
    
    def resample_data(self, data: pd.DataFrame, freq: str = '1D') -> pd.DataFrame:
        """
        Resample time series data to a different frequency.
        
        Args:
            data: DataFrame with time series data
            freq: Target frequency (e.g., '1D', '1H', '5min')
            
        Returns:
            Resampled DataFrame
        """
        df = data.copy()
        
        # Check if timestamp column exists
        if 'timestamp' not in df.columns:
            logger.error("Cannot resample data without timestamp column")
            return df
        
        logger.info(f"Resampling data to frequency: {freq}")
        
        # Process each symbol separately if 'symbol' column exists
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            result_dfs = []
            
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol].copy()
                
                # Set timestamp as index
                symbol_data = symbol_data.set_index('timestamp')
                
                # Resample data
                resampled = self._resample_ohlcv(symbol_data, freq)
                
                # Reset index and add symbol column
                resampled = resampled.reset_index()
                resampled['symbol'] = symbol
                
                result_dfs.append(resampled)
            
            # Combine results
            result = pd.concat(result_dfs, ignore_index=True)
        else:
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Resample data
            result = self._resample_ohlcv(df, freq)
            
            # Reset index
            result = result.reset_index()
        
        return result
    
    def _resample_ohlcv(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different frequency.
        
        Args:
            data: DataFrame with timestamp index
            freq: Target frequency
            
        Returns:
            Resampled DataFrame
        """
        # Define aggregation functions for OHLCV data
        agg_dict = {}
        
        # Price columns
        if 'open' in data.columns:
            agg_dict['open'] = 'first'
        if 'high' in data.columns:
            agg_dict['high'] = 'max'
        if 'low' in data.columns:
            agg_dict['low'] = 'min'
        if 'close' in data.columns:
            agg_dict['close'] = 'last'
        
        # Volume column
        if 'volume' in data.columns:
            agg_dict['volume'] = 'sum'
        
        # Other numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'mean'
        
        # Resample data
        resampled = data.resample(freq).agg(agg_dict)
        
        # Fill missing values
        resampled = resampled.fillna(method='ffill')
        
        return resampled
    
    def filter_by_date(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: DataFrame with time series data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame
        """
        df = data.copy()
        
        # Check if timestamp column exists
        if 'timestamp' not in df.columns:
            logger.error("Cannot filter data without timestamp column")
            return df
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by start date
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]
            logger.info(f"Filtered data after {start_date}")
        
        # Filter by end date
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]
            logger.info(f"Filtered data before {end_date}")
        
        return df
    
    def filter_by_volume(self, data: pd.DataFrame, min_volume: float = None) -> pd.DataFrame:
        """
        Filter data by minimum volume.
        
        Args:
            data: DataFrame with market data
            min_volume: Minimum volume threshold
            
        Returns:
            Filtered DataFrame
        """
        df = data.copy()
        
        # Check if volume column exists
        if 'volume' not in df.columns:
            logger.error("Cannot filter data without volume column")
            return df
        
        # Filter by minimum volume
        if min_volume is not None:
            df = df[df['volume'] >= min_volume]
            logger.info(f"Filtered data with volume >= {min_volume}")
        
        return df
    
    def adjust_for_splits_dividends(self, data: pd.DataFrame, adjustment_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Adjust price data for stock splits and dividends.
        
        Args:
            data: DataFrame with market data
            adjustment_data: DataFrame with split and dividend information
            
        Returns:
            Adjusted DataFrame
        """
        df = data.copy()
        
        # Check if required columns exist
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Cannot adjust data, missing columns: {missing_cols}")
            return df
        
        # If no adjustment data provided, return original data
        if adjustment_data is None or adjustment_data.empty:
            logger.warning("No adjustment data provided, returning original data")
            return df
        
        logger.info("Adjusting price data for splits and dividends")
        
        # Process each symbol separately
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Get adjustment data for this symbol
            symbol_adj = adjustment_data[adjustment_data['symbol'] == symbol].copy()
            
            if symbol_adj.empty:
                # No adjustments for this symbol
                result_dfs.append(symbol_data)
                continue
            
            # Sort data by timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            symbol_adj = symbol_adj.sort_values('timestamp')
            
            # Apply adjustments
            symbol_data = self._apply_adjustments(symbol_data, symbol_adj)
            
            result_dfs.append(symbol_data)
        
        # Combine results
        result = pd.concat(result_dfs, ignore_index=True)
        
        return result
    
    def _apply_adjustments(self, data: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
        """
        Apply split and dividend adjustments to price data.
        
        Args:
            data: DataFrame with price data for a single symbol
            adjustments: DataFrame with adjustment data for the symbol
            
        Returns:
            Adjusted DataFrame
        """
        df = data.copy()
        
        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if not pd.api.types.is_datetime64_any_dtype(adjustments['timestamp']):
            adjustments['timestamp'] = pd.to_datetime(adjustments['timestamp'])
        
        # Process each adjustment
        for _, adj in adjustments.iterrows():
            adj_date = adj['timestamp']
            
            # Apply adjustment to data before the adjustment date
            mask = df['timestamp'] < adj_date
            
            if 'split_ratio' in adj and not pd.isna(adj['split_ratio']):
                # Apply split adjustment
                split_ratio = adj['split_ratio']
                
                if split_ratio != 0 and split_ratio != 1:
                    df.loc[mask, 'open'] = df.loc[mask, 'open'] / split_ratio
                    df.loc[mask, 'high'] = df.loc[mask, 'high'] / split_ratio
                    df.loc[mask, 'low'] = df.loc[mask, 'low'] / split_ratio
                    df.loc[mask, 'close'] = df.loc[mask, 'close'] / split_ratio
                    
                    # Adjust volume in the opposite direction
                    df.loc[mask, 'volume'] = df.loc[mask, 'volume'] * split_ratio
            
            if 'dividend_amount' in adj and not pd.isna(adj['dividend_amount']):
                # Apply dividend adjustment
                dividend = adj['dividend_amount']
                
                if dividend > 0:
                    # Get close price on the day before dividend
                    prev_close = df.loc[df['timestamp'] < adj_date, 'close'].iloc[-1] if any(mask) else 0
                    
                    if prev_close > 0:
                        # Calculate adjustment factor
                        factor = (prev_close - dividend) / prev_close
                        
                        # Apply adjustment
                        df.loc[mask, 'open'] = df.loc[mask, 'open'] * factor
                        df.loc[mask, 'high'] = df.loc[mask, 'high'] * factor
                        df.loc[mask, 'low'] = df.loc[mask, 'low'] * factor
                        df.loc[mask, 'close'] = df.loc[mask, 'close'] * factor
        
        return df
    
    # ---- WebSocket-specific methods for real-time data ----
    
    def clean_websocket_data(self, data: Dict[str, Any], data_type: str, symbol: str) -> Dict[str, Any]:
        """
        Clean a single WebSocket data point in real-time.
        This is optimized for speed with minimal overhead for real-time processing.
        
        Args:
            data: Dictionary with WebSocket data
            data_type: Type of data (trades, quotes, aggregates)
            symbol: Ticker symbol
            
        Returns:
            Cleaned data dictionary
        """
        with self._lock:
            # Make a copy to avoid modifying the original
            cleaned_data = data.copy()
            
            # Get last values for this symbol
            if symbol not in self._last_values:
                self._last_values[symbol] = {}
                self._missing_count[symbol] = {}
            
            # Simple validation - remove NaN/None values
            for key, value in list(cleaned_data.items()):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    # Check if we have a previous value
                    if key in self._last_values[symbol]:
                        # Use last known value
                        cleaned_data[key] = self._last_values[symbol][key]
                        
                        # Increment missing count
                        self._missing_count[symbol][key] = self._missing_count[symbol].get(key, 0) + 1
                    else:
                        # No previous value, remove field
                        del cleaned_data[key]
                else:
                    # Update last known value
                    self._last_values[symbol][key] = value
                    
                    # Reset missing count
                    self._missing_count[symbol][key] = 0
            
            # Check for suspicious price movements in trade data
            if data_type == 'trades' and 'price' in cleaned_data and 'price' in self._last_values[symbol]:
                last_price = self._last_values[symbol]['price']
                current_price = cleaned_data['price']
                
                # Check if price change exceeds threshold
                if last_price > 0:
                    price_change = abs(current_price - last_price) / last_price
                    if price_change > self.ws_max_deviation:
                        logger.warning(f"Suspicious price movement for {symbol}: {last_price} -> {current_price}")
                        
                        # For extreme deviations, keep the last known value
                        if price_change > self.ws_max_deviation * 2:
                            cleaned_data['price'] = last_price
                            # Add a flag indicating this was corrected
                            cleaned_data['price_corrected'] = True
            
            # Apply smoothing to trades data if enabled
            if data_type == 'trades' and self.ws_enable_smoothing:
                cleaned_data = self._smooth_websocket_data(cleaned_data, symbol)
            
            return cleaned_data
    
    def _smooth_websocket_data(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Apply smoothing to WebSocket data to reduce noise.
        
        Args:
            data: Dictionary with WebSocket data
            symbol: Ticker symbol
            
        Returns:
            Smoothed data dictionary
        """
        # Only smooth numeric values like price
        for key in ['price', 'size']:
            if key in data and key in self._last_values[symbol]:
                # Get current value
                current_value = data[key]
                
                # Get last value
                last_value = self._last_values[symbol][key]
                
                # Apply simple exponential smoothing
                alpha = 0.3  # Smoothing factor
                smoothed_value = alpha * current_value + (1 - alpha) * last_value
                
                # Round to appropriate precision
                if key == 'price':
                    # Round to 2 decimal places for price
                    smoothed_value = round(smoothed_value, 2)
                elif key == 'size':
                    # Round to integer for size
                    smoothed_value = round(smoothed_value)
                
                # Update with smoothed value
                data[key] = smoothed_value
        
        return data
    
    def merge_websocket_and_rest_data(self, ws_data: List[Dict[str, Any]], 
                                     rest_data: pd.DataFrame, 
                                     symbol: str) -> pd.DataFrame:
        """
        Merge WebSocket and REST API data for a symbol.
        
        Args:
            ws_data: List of WebSocket data points
            rest_data: DataFrame with REST API data
            symbol: Ticker symbol
            
        Returns:
            Merged DataFrame
        """
        if not ws_data:
            logger.debug(f"No WebSocket data for {symbol}, returning REST data only")
            return rest_data
        
        if rest_data.empty:
            logger.debug(f"No REST data for {symbol}, converting WebSocket data to DataFrame")
            # Convert WebSocket data to DataFrame
            ws_df = pd.DataFrame(ws_data)
            
            # Add symbol column if not present
            if 'symbol' not in ws_df.columns:
                ws_df['symbol'] = symbol
            
            return ws_df
        
        logger.info(f"Merging WebSocket and REST data for {symbol}")
        
        # Convert WebSocket data to DataFrame
        ws_df = pd.DataFrame(ws_data)
        
        # Add symbol column if not present
        if 'symbol' not in ws_df.columns:
            ws_df['symbol'] = symbol
        
        # Ensure timestamp is datetime
        if 'timestamp' in ws_df.columns:
            ws_df['timestamp'] = pd.to_datetime(ws_df['timestamp'], unit='s')
        
        if 'timestamp' in rest_data.columns:
            rest_data['timestamp'] = pd.to_datetime(rest_data['timestamp'])
        
        # Get the latest timestamp from REST data
        if not rest_data.empty and 'timestamp' in rest_data.columns:
            latest_rest_time = rest_data['timestamp'].max()
            
            # Filter WebSocket data to only include newer data points
            if 'timestamp' in ws_df.columns:
                ws_df = ws_df[ws_df['timestamp'] > latest_rest_time]
        
        # Combine the two DataFrames
        merged_df = pd.concat([rest_data, ws_df], ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values('timestamp')
        
        # Remove duplicates based on timestamp
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset='timestamp', keep='last')
        
        return merged_df
    
    def validate_websocket_data(self, data: Dict[str, Any], data_type: str) -> bool:
        """
        Validate WebSocket data for consistency.
        
        Args:
            data: Dictionary with WebSocket data
            data_type: Type of data (trades, quotes, aggregates)
            
        Returns:
            Boolean indicating validity
        """
        # Basic validation - check for required fields
        if data_type == 'trades':
            required_fields = ['price', 'size', 'timestamp']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field in trades data: {field}")
                    return False
            
            # Validate price and size
            if data['price'] <= 0:
                logger.warning(f"Invalid price in trades data: {data['price']}")
                return False
            
            if data['size'] <= 0:
                logger.warning(f"Invalid size in trades data: {data['size']}")
                return False
        
        elif data_type == 'quotes':
            required_fields = ['bid', 'ask', 'bidsize', 'asksize', 'timestamp']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field in quotes data: {field}")
                    return False
            
            # Validate bid and ask
            if data['bid'] <= 0 or data['ask'] <= 0:
                logger.warning(f"Invalid bid/ask in quotes data: bid={data.get('bid')}, ask={data.get('ask')}")
                return False
            
            # Validate spread
            if data['ask'] < data['bid']:
                logger.warning(f"Invalid spread in quotes data: bid={data['bid']}, ask={data['ask']}")
                return False
        
        elif data_type == 'bars' or data_type == 'aggregates':
            required_fields = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field in bars data: {field}")
                    return False
            
            # Validate OHLC values
            if not (data['low'] <= data['open'] <= data['high'] and 
                    data['low'] <= data['close'] <= data['high']):
                logger.warning(f"Invalid OHLC relationships in bars data: "
                              f"O={data['open']}, H={data['high']}, L={data['low']}, C={data['close']}")
                return False
        
        return True


# Module-level instance for convenience
_default_cleaner = None

def get_data_cleaner(config: Dict[str, Any] = None) -> DataCleaner:
    """
    Get or create the default data cleaner.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataCleaner instance
    """
    global _default_cleaner
    
    if _default_cleaner is None:
        _default_cleaner = DataCleaner(config)
    
    return _default_cleaner


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Cleaner')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--missing', type=str, default='ffill', 
                      choices=['drop', 'ffill', 'linear', 'time'], 
                      help='Missing value strategy')
    parser.add_argument('--outliers', type=str, default='zscore', 
                      choices=['zscore', 'iqr', 'winsorize'], 
                      help='Outlier detection method')
    parser.add_argument('--normalize', action='store_true', help='Normalize data')
    parser.add_argument('--norm-method', type=str, default='minmax', 
                      choices=['minmax', 'zscore', 'robust'], 
                      help='Normalization method')
    parser.add_argument('--websocket', action='store_true', help='Process as WebSocket data')
    
    args = parser.parse_args()
    
    # Load data
    try:
        data = pd.read_csv(args.input)
        print(f"Loaded data with shape {data.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)
    
    # Create config
    config = {
        'handle_missing': True,
        'missing_strategy': args.missing,
        'handle_outliers': True,
        'outlier_method': args.outliers,
        'normalize_data': args.normalize,
        'norm_method': args.norm_method,
        'ws_enable_smoothing': args.websocket
    }
    
    # Create cleaner
    cleaner = DataCleaner(config)
    
    # Clean data
    cleaned_data = cleaner.clean(data)
    
    # Save output
    if args.output:
        cleaned_data.to_csv(args.output, index=False)
        print(f"Saved cleaned data to {args.output}")
    else:
        print(cleaned_data.head())
