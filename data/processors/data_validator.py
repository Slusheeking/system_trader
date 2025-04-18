#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Validator Module
------------------
Provides data validation functionality for market data.
Ensures data quality and integrity before processing and model training.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('data_validator', category='data')


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """
    Data validation for market data.
    Provides methods for checking data quality and integrity.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure validation options
        self.check_required_columns = self.config.get('check_required_columns', True)
        self.check_data_types = self.config.get('check_data_types', True)
        self.check_value_ranges = self.config.get('check_value_ranges', True)
        self.check_duplicates = self.config.get('check_duplicates', True)
        self.check_continuity = self.config.get('check_continuity', True)
        
        # Configure validation thresholds
        self.max_missing_pct = self.config.get('max_missing_pct', 100.0)  # Allow 100% missing for some columns
        self.max_outlier_pct = self.config.get('max_outlier_pct', 5.0)
        self.min_data_points = self.config.get('min_data_points', 1)  # Allow single data point for testing
        
        # Configure required columns and data types
        self.required_columns = self.config.get('required_columns', 
                                             ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        
        self.expected_dtypes = self.config.get('expected_dtypes', {
            'timestamp': 'datetime64',
            'symbol': 'object',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        })
        
        # Configure value ranges
        self.value_ranges = self.config.get('value_ranges', {
            'open': (0, float('inf')),
            'high': (0, float('inf')),
            'low': (0, float('inf')),
            'close': (0, float('inf')),
            'volume': (0, float('inf'))
        })
        
        logger.info("DataValidator initialized")
    
    def validate(self, data: pd.DataFrame, raise_errors: bool = False) -> Dict[str, Any]:
        """
        Validate market data.
        
        Args:
            data: DataFrame with market data
            raise_errors: Whether to raise exceptions for validation errors
            
        Returns:
            Dictionary with validation results
        """
        if data.empty:
            msg = "Empty DataFrame provided, nothing to validate"
            logger.warning(msg)
            if raise_errors:
                raise ValidationError(msg)
            return {'valid': False, 'errors': [msg]}
        
        logger.info(f"Validating data with shape {data.shape}")
        
        # Initialize validation results
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'row_count': len(data),
                'column_count': len(data.columns),
                'symbols': list(data['symbol'].unique()) if 'symbol' in data.columns else [],
                'date_range': [
                    data['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in data.columns else None,
                    data['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in data.columns else None
                ]
            }
        }
        
        # Run validation checks
        if self.check_required_columns:
            self._validate_required_columns(data, results)
        
        if self.check_data_types:
            self._validate_data_types(data, results)
        
        if self.check_value_ranges:
            self._validate_value_ranges(data, results)
        
        if self.check_duplicates:
            self._validate_duplicates(data, results)
        
        if self.check_continuity:
            self._validate_continuity(data, results)
        
        # Additional checks
        self._validate_missing_values(data, results)
        self._validate_ohlc_integrity(data, results)
        self._validate_data_size(data, results)
        
        # Set overall validation status
        results['valid'] = len(results['errors']) == 0
        
        # Log validation results
        if results['valid']:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed with {len(results['errors'])} errors")
            for error in results['errors']:
                logger.error(f"Validation error: {error}")
        
        if results['warnings']:
            logger.info(f"Data validation generated {len(results['warnings'])} warnings")
            for warning in results['warnings']:
                logger.warning(f"Validation warning: {warning}")
        
        # Raise exception if requested
        if raise_errors and not results['valid']:
            raise ValidationError(f"Data validation failed: {results['errors']}")
        
        return results
    
    def _validate_required_columns(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate that required columns are present.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            results['errors'].append(error_msg)
    
    def _validate_data_types(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate that columns have the expected data types.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        for col, expected_dtype in self.expected_dtypes.items():
            if col in data.columns:
                # Check if column is datetime
                if expected_dtype.startswith('datetime'):
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        # Try to convert to datetime
                        try:
                            # Just check if conversion is possible, don't modify the data
                            pd.to_datetime(data[col])
                            results['warnings'].append(
                                f"Column '{col}' is not datetime type but can be converted"
                            )
                        except:
                            results['errors'].append(
                                f"Column '{col}' has incorrect data type. Expected {expected_dtype}, "
                                f"got {data[col].dtype}"
                            )
                # Check numeric types
                elif expected_dtype.startswith(('int', 'float')):
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        # Try to convert to numeric
                        try:
                            # Just check if conversion is possible, don't modify the data
                            pd.to_numeric(data[col])
                            results['warnings'].append(
                                f"Column '{col}' is not numeric type but can be converted"
                            )
                        except:
                            results['errors'].append(
                                f"Column '{col}' has incorrect data type. Expected {expected_dtype}, "
                                f"got {data[col].dtype}"
                            )
                # Check other types
                elif str(data[col].dtype) != expected_dtype:
                    results['warnings'].append(
                        f"Column '{col}' has unexpected data type. Expected {expected_dtype}, "
                        f"got {data[col].dtype}"
                    )
    
    def _validate_value_ranges(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate that values are within expected ranges.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in data.columns:
                # Check for values below minimum
                below_min = (data[col] < min_val).sum()
                if below_min > 0:
                    results['errors'].append(
                        f"Column '{col}' has {below_min} values below minimum {min_val}"
                    )
                
                # Check for values above maximum
                if max_val != float('inf'):
                    above_max = (data[col] > max_val).sum()
                    if above_max > 0:
                        results['errors'].append(
                            f"Column '{col}' has {above_max} values above maximum {max_val}"
                        )
                
                # Check for NaN values
                nan_count = data[col].isna().sum()
                if nan_count > 0:
                    nan_pct = (nan_count / len(data)) * 100
                    if nan_pct > self.max_missing_pct:
                        results['errors'].append(
                            f"Column '{col}' has {nan_count} NaN values ({nan_pct:.2f}%), "
                            f"exceeding threshold of {self.max_missing_pct}%"
                        )
                    else:
                        results['warnings'].append(
                            f"Column '{col}' has {nan_count} NaN values ({nan_pct:.2f}%)"
                        )
    
    def _validate_duplicates(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate that there are no duplicate rows.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        # Check for exact duplicates
        exact_dups = data.duplicated().sum()
        if exact_dups > 0:
            results['errors'].append(f"Found {exact_dups} exact duplicate rows")
        
        # Check for duplicates based on timestamp and symbol
        if 'timestamp' in data.columns and 'symbol' in data.columns:
            ts_sym_dups = data.duplicated(subset=['timestamp', 'symbol']).sum()
            if ts_sym_dups > 0:
                results['errors'].append(
                    f"Found {ts_sym_dups} duplicate timestamp-symbol combinations"
                )
    
    def _validate_continuity(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate time series continuity.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        if 'timestamp' not in data.columns:
            return
        
        # Process each symbol separately if 'symbol' column exists
        if 'symbol' in data.columns:
            symbols = data['symbol'].unique()
            
            for symbol in symbols:
                symbol_data = data[data['symbol'] == symbol].copy()
                
                # Sort by timestamp
                symbol_data = symbol_data.sort_values('timestamp')
                
                # Check for gaps in time series
                self._check_time_gaps(symbol_data, symbol, results)
        else:
            # Sort by timestamp
            sorted_data = data.sort_values('timestamp')
            
            # Check for gaps in time series
            self._check_time_gaps(sorted_data, None, results)
    
    def _check_time_gaps(self, data: pd.DataFrame, symbol: Optional[str], results: Dict[str, Any]) -> None:
        """
        Check for gaps in time series data.
        
        Args:
            data: DataFrame with sorted time series data
            symbol: Symbol name or None
            results: Validation results dictionary to update
        """
        if len(data) < 2:
            return
        
        # Calculate time differences
        time_diffs = data['timestamp'].diff().dropna()
        
        # Get unique time differences to identify the most common (expected) interval
        unique_diffs = time_diffs.value_counts().sort_values(ascending=False)
        
        if len(unique_diffs) == 0:
            return
        
        # Assume the most common difference is the expected interval
        expected_interval = unique_diffs.index[0]
        
        # Check for gaps (time differences greater than expected)
        gaps = time_diffs[time_diffs > expected_interval * 1.5]
        
        if len(gaps) > 0:
            symbol_str = f" for symbol {symbol}" if symbol else ""
            results['warnings'].append(
                f"Found {len(gaps)} time gaps{symbol_str}. Expected interval: {expected_interval}, "
                f"max gap: {gaps.max()}"
            )
    
    def _validate_missing_values(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate missing values.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        # Calculate missing values by column
        missing = data.isna().sum()
        missing_pct = (missing / len(data)) * 100
        
        # Add missing value stats to results
        results['stats']['missing_values'] = {
            col: {'count': int(count), 'percent': float(pct)}
            for col, count, pct in zip(missing.index, missing, missing_pct)
            if count > 0
        }
        
        # Check if any column exceeds the missing value threshold
        for col, pct in missing_pct.items():
            if pct > self.max_missing_pct:
                results['errors'].append(
                    f"Column '{col}' has {pct:.2f}% missing values, "
                    f"exceeding threshold of {self.max_missing_pct}%"
                )
    
    def _validate_ohlc_integrity(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate OHLC data integrity.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        # Check if OHLC columns exist
        ohlc_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in ohlc_cols):
            return
        
        # Check high >= low
        invalid_hl = (data['high'] < data['low']).sum()
        if invalid_hl > 0:
            results['errors'].append(f"Found {invalid_hl} rows where high < low")
        
        # Check high >= open
        invalid_ho = (data['high'] < data['open']).sum()
        if invalid_ho > 0:
            results['errors'].append(f"Found {invalid_ho} rows where high < open")
        
        # Check high >= close
        invalid_hc = (data['high'] < data['close']).sum()
        if invalid_hc > 0:
            results['errors'].append(f"Found {invalid_hc} rows where high < close")
        
        # Check low <= open
        invalid_lo = (data['low'] > data['open']).sum()
        if invalid_lo > 0:
            results['errors'].append(f"Found {invalid_lo} rows where low > open")
        
        # Check low <= close
        invalid_lc = (data['low'] > data['close']).sum()
        if invalid_lc > 0:
            results['errors'].append(f"Found {invalid_lc} rows where low > close")
    
    def _validate_data_size(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Validate data size.
        
        Args:
            data: DataFrame to validate
            results: Validation results dictionary to update
        """
        # Check if data has enough rows
        if len(data) < self.min_data_points:
            results['errors'].append(
                f"Data has only {len(data)} rows, minimum required is {self.min_data_points}"
            )
        
        # Check data size by symbol if symbol column exists
        if 'symbol' in data.columns:
            symbol_counts = data['symbol'].value_counts()
            
            # Add symbol counts to stats
            results['stats']['symbol_counts'] = symbol_counts.to_dict()
            
            # Check if any symbol has too few data points
            for symbol, count in symbol_counts.items():
                if count < self.min_data_points:
                    results['warnings'].append(
                        f"Symbol '{symbol}' has only {count} data points, "
                        f"minimum recommended is {self.min_data_points}"
                    )
    
    def generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'missing_values': {col: int(data[col].isna().sum()) for col in data.columns},
            'missing_pct': {col: float((data[col].isna().sum() / len(data)) * 100) for col in data.columns}
        }
        
        # Add numeric column statistics
        numeric_cols = data.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = {
                col: {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean()),
                    'median': float(data[col].median()),
                    'std': float(data[col].std())
                }
                for col in numeric_cols
            }
        
        # Add timestamp information if available
        if 'timestamp' in data.columns:
            timestamp_col = data['timestamp']
            if pd.api.types.is_datetime64_any_dtype(timestamp_col):
                summary['time_range'] = {
                    'start': timestamp_col.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': timestamp_col.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_days': (timestamp_col.max() - timestamp_col.min()).days
                }
        
        # Add symbol information if available
        if 'symbol' in data.columns:
            symbols = data['symbol'].unique()
            summary['symbols'] = {
                'count': len(symbols),
                'list': list(symbols),
                'data_points_per_symbol': {
                    symbol: int(data[data['symbol'] == symbol].shape[0])
                    for symbol in symbols
                }
            }
        
        return summary
    
    def check_for_outliers(self, data: pd.DataFrame, method: str = 'zscore', 
                         threshold: float = 3.0) -> Dict[str, Any]:
        """
        Check for outliers in numeric columns.
        
        Args:
            data: DataFrame to check
            method: Outlier detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        # Exclude certain columns
        exclude_cols = ['timestamp', 'symbol', 'volume']
        outlier_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        results = {
            'method': method,
            'threshold': threshold,
            'outliers_by_column': {}
        }
        
        for col in outlier_cols:
            # Detect outliers based on method
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
            elif method == 'iqr':
                # Interquartile Range method
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            # Count outliers
            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(data)) * 100
            
            # Store results
            results['outliers_by_column'][col] = {
                'count': int(outlier_count),
                'percent': float(outlier_pct),
                'indices': list(data.index[outliers])
            }
            
            # Check if outliers exceed threshold
            if outlier_pct > self.max_outlier_pct:
                logger.warning(
                    f"Column '{col}' has {outlier_pct:.2f}% outliers, "
                    f"exceeding threshold of {self.max_outlier_pct}%"
                )
        
        return results
    
    def check_stationarity(self, data: pd.DataFrame, column: str = 'close') -> Dict[str, Any]:
        """
        Check for stationarity in time series data.
        
        Args:
            data: DataFrame with time series data
            column: Column to check for stationarity
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore
        except ImportError:
            logger.warning("statsmodels not installed, cannot check stationarity")
            return {'error': 'statsmodels not installed'}
        
        if column not in data.columns:
            return {'error': f"Column '{column}' not found in data"}
        
        results = {}
        
        # Process each symbol separately if 'symbol' column exists
        if 'symbol' in data.columns:
            symbols = data['symbol'].unique()
            
            for symbol in symbols:
                symbol_data = data[data['symbol'] == symbol].copy()
                
                # Sort by timestamp if available
                if 'timestamp' in symbol_data.columns:
                    symbol_data = symbol_data.sort_values('timestamp')
                
                # Run ADF test
                try:
                    adf_result = adfuller(symbol_data[column].dropna())
                    results[symbol] = {
                        'adf_statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'critical_values': {str(key): float(val) for key, val in adf_result[4].items()},
                        'is_stationary': adf_result[1] < 0.05
                    }
                except Exception as e:
                    results[symbol] = {'error': str(e)}
        else:
            # Sort by timestamp if available
            sorted_data = data
            if 'timestamp' in data.columns:
                sorted_data = data.sort_values('timestamp')
            
            # Run ADF test
            try:
                adf_result = adfuller(sorted_data[column].dropna())
                results['all_data'] = {
                    'adf_statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {str(key): float(val) for key, val in adf_result[4].items()},
                    'is_stationary': adf_result[1] < 0.05
                }
            except Exception as e:
                results['all_data'] = {'error': str(e)}
        
        return results
    
    def check_correlation(self, data: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """
        Check correlation between columns.
        
        Args:
            data: DataFrame to check
            columns: Columns to include in correlation analysis (None for all numeric)
            
        Returns:
            Dictionary with correlation information
        """
        # Get numeric columns if not specified
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to only include columns that exist and are numeric
            numeric_cols = data.select_dtypes(include=np.number).columns
            columns = [col for col in columns if col in numeric_cols]
        
        if len(columns) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        results = {}
        
        # Process each symbol separately if 'symbol' column exists
        if 'symbol' in data.columns:
            symbols = data['symbol'].unique()
            
            for symbol in symbols:
                symbol_data = data[data['symbol'] == symbol].copy()
                
                # Calculate correlation matrix
                corr_matrix = symbol_data[columns].corr()
                
                # Convert to dictionary
                results[symbol] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'high_correlations': self._get_high_correlations(corr_matrix)
                }
        else:
            # Calculate correlation matrix
            corr_matrix = data[columns].corr()
            
            # Convert to dictionary
            results['all_data'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': self._get_high_correlations(corr_matrix)
            }
        
        return results
    
    def _get_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Get pairs of columns with high correlation.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold
            
        Returns:
            List of high correlation pairs
        """
        high_corrs = []
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs with correlation above threshold
        for col in upper.columns:
            for idx, val in upper[col].items():
                if abs(val) >= threshold:
                    high_corrs.append({
                        'column1': idx,
                        'column2': col,
                        'correlation': float(val)
                    })
        
        return high_corrs


# Module-level instance for convenience
_default_validator = None

def get_data_validator(config: Dict[str, Any] = None) -> DataValidator:
    """
    Get or create the default data validator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataValidator instance
    """
    global _default_validator
    
    if _default_validator is None:
        _default_validator = DataValidator(config)
    
    return _default_validator


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Data Validator')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output JSON file for validation results')
    parser.add_argument('--summary', action='store_true', help='Generate data summary')
    parser.add_argument('--outliers', action='store_true', help='Check for outliers')
    parser.add_argument('--correlation', action='store_true', help='Check correlation')
    
    args = parser.parse_args()
    
    # Load data
    try:
        data = pd.read_csv(args.input)
        print(f"Loaded data with shape {data.shape}")
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)
    
    # Create validator
    validator = DataValidator()
    
    # Validate data
    validation_results = validator.validate(data)
    
    # Print validation results
    if validation_results['valid']:
        print("Data validation passed")
    else:
        print(f"Data validation failed with {len(validation_results['errors'])} errors:")
        for error in validation_results['errors']:
            print(f"  - {error}")
    
    if validation_results['warnings']:
        print(f"Data validation generated {len(validation_results['warnings'])} warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    # Generate summary if requested
    if args.summary:
        summary = validator.generate_summary(data)
        print("\nData Summary:")
        print(f"  - Shape: {summary['shape']}")
        print(f"  - Columns: {len(summary['columns'])}")
        if 'time_range' in summary:
            print(f"  - Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        if 'symbols' in summary:
            print(f"  - Symbols: {summary['symbols']['count']}")
    
    # Check for outliers if requested
    if args.outliers:
        outlier_results = validator.check_for_outliers(data)
        print("\nOutlier Detection:")
        for col, info in outlier_results['outliers_by_column'].items():
            print(f"  - {col}: {info['count']} outliers ({info['percent']:.2f}%)")
    
    # Check correlation if requested
    if args.correlation:
        corr_results = validator.check_correlation(data)
        print("\nCorrelation Analysis:")
        for symbol, info in corr_results.items():
            if 'high_correlations' in info:
                print(f"  - {symbol}:")
                for corr in info['high_correlations']:
                    print(f"    - {corr['column1']} and {corr['column2']}: {corr['correlation']:.2f}")
    
    # Save results if output file specified
    if args.output:
        output = {
            'validation': validation_results,
        }
        
        if args.summary:
            output['summary'] = summary
        
        if args.outliers:
            output['outliers'] = outlier_results
        
        if args.correlation:
            output['correlation'] = corr_results
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {args.output}")
