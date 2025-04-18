#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Market Regime Detection Model
------------------------------------
This module implements a Hidden Markov Model and XGBoost hybrid for detecting
market regimes, enhanced with options flow data from Unusual Whales.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans

from utils.logging import setup_logger

# Setup logging
logger = setup_logger('market_regime_model')


class EnhancedMarketRegimeModel:
    """
    Enhanced Market Regime Model using HMM and XGBoost to identify different
    market states, enriched with options flow data from Unusual Whales.
    """

    def __init__(self, config: Dict):
        """
        Initialize the enhanced market regime model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model hyperparameters
        self.n_regimes = config.get('n_regimes', 4)
        self.lookback_window = config.get('lookback_window', 60)  # 60 days for regime detection
        self.smooth_window = config.get('smooth_window', 5)  # Smoothing window for regime transitions
        self.hmm_n_iter = config.get('hmm_n_iter', 100)
        
        # XGBoost parameters
        self.xgb_n_estimators = config.get('xgb_n_estimators', 100)
        self.xgb_learning_rate = config.get('xgb_learning_rate', 0.1)
        self.xgb_max_depth = config.get('xgb_max_depth', 5)
        
        # Feature parameters
        self.feature_groups = config.get('feature_groups', {
            'returns': True,
            'volatility': True,
            'trend': True,
            'breadth': True,
            'sentiment': True,
            'options_flow': True  # New feature group for options data
        })
        
        # Initialize models
        self.hmm_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        
        # For storing regime mappings
        self.regime_mapping = {}
        self.regime_names = ['trending_up', 'trending_down', 'high_volatility', 'low_volatility']
        
        # Track metrics
        self.metrics = {}
        
        # For storing regime transition probabilities
        self.transition_matrix = None
        
        logger.info("Enhanced Market Regime Model initialized")
    
    def _prepare_hmm_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for the HMM model, including options flow features.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            numpy array of features for HMM
        """
        df = data.copy()
        
        # Ensure we have enough data
        if len(df) < self.lookback_window:
            logger.error(f"Not enough data for HMM. Need {self.lookback_window}, got {len(df)}")
            return np.array([])
        
        # Required columns
        required_columns = ['close']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            return np.array([])
        
        # Calculate returns
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df = df.dropna()
        
        # Calculate volatility
        if 'volatility' not in df.columns:
            df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
            df = df.dropna()
        
        # Select features for HMM
        hmm_features = ['returns', 'volatility']
        
        # Optional features
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            hmm_features.append('volume_change')
        
        if 'vix' in df.columns:
            hmm_features.append('vix')
        
        # Add options flow features if available
        options_flow_features = [
            'put_call_ratio', 'call_premium_volume', 'put_premium_volume', 
            'smart_money_direction', 'unusual_activity_score', 'implied_volatility',
            'iv_skew', 'gamma_exposure'
        ]
        
        for feature in options_flow_features:
            if feature in df.columns:
                hmm_features.append(feature)
        
        # Scale features
        feature_array = df[hmm_features].values
        scaled_features = self.scaler.fit_transform(feature_array)
        
        return scaled_features
    
    def _prepare_xgb_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for the XGBoost classifier, enhanced with options flow data.
        
        Args:
            data: DataFrame with market data and HMM regime labels
            
        Returns:
            Tuple of (feature DataFrame, target array)
        """
        df = data.copy()
        
        # Ensure we have regime labels
        if 'regime' not in df.columns:
            logger.error("No regime labels found in data")
            return pd.DataFrame(), np.array([])
        
        # Create feature set
        features = []
        
        # Returns features
        if self.feature_groups.get('returns', True):
            # Calculate return features
            for window in [1, 5, 10, 21, 63]:
                df[f'return_{window}d'] = df['close'].pct_change(window)
            
            features.extend([f'return_{window}d' for window in [1, 5, 10, 21, 63]])
        
        # Volatility features
        if self.feature_groups.get('volatility', True):
            # Calculate volatility features
            for window in [10, 21, 63]:
                df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
            
            # Volatility ratios
            df['volatility_ratio_10_21'] = df['volatility_10d'] / df['volatility_21d']
            df['volatility_ratio_21_63'] = df['volatility_21d'] / df['volatility_63d']
            
            features.extend([f'volatility_{window}d' for window in [10, 21, 63]])
            features.extend(['volatility_ratio_10_21', 'volatility_ratio_21_63'])
        
        # Trend features
        if self.feature_groups.get('trend', True):
            # Calculate moving averages
            for window in [10, 21, 50, 200]:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            
            # Price relative to moving averages
            df['price_to_ma_50'] = df['close'] / df['ma_50']
            df['price_to_ma_200'] = df['close'] / df['ma_200']
            df['ma_50_200_ratio'] = df['ma_50'] / df['ma_200']
            
            # Trend strength (ADX-like)
            df['trend_strength'] = (df['ma_50_200_ratio'] - 1).abs() * 100
            
            features.extend(['price_to_ma_50', 'price_to_ma_200', 'ma_50_200_ratio', 'trend_strength'])
        
        # Market breadth features
        if self.feature_groups.get('breadth', True) and 'advance_decline' in df.columns:
            df['ad_ratio'] = df['advance_decline'].rolling(window=10).mean()
            df['ad_trend'] = df['ad_ratio'].pct_change(5)
            
            features.extend(['ad_ratio', 'ad_trend'])
        
        # Sentiment features
        if self.feature_groups.get('sentiment', True) and 'vix' in df.columns:
            df['vix_ma_10'] = df['vix'].rolling(window=10).mean()
            df['vix_ratio'] = df['vix'] / df['vix_ma_10']
            
            features.extend(['vix', 'vix_ma_10', 'vix_ratio'])
        
        # Options flow features
        if self.feature_groups.get('options_flow', True):
            options_features = [
                'put_call_ratio', 'call_premium_volume', 'put_premium_volume',
                'smart_money_direction', 'unusual_activity_score', 'implied_volatility',
                'iv_skew', 'gamma_exposure'
            ]
            
            # Add available options flow features
            for feature in options_features:
                if feature in df.columns:
                    # Apply rolling averages for smoothing
                    df[f'{feature}_ma5'] = df[feature].rolling(window=5).mean()
                    features.extend([feature, f'{feature}_ma5'])
            
            # Derived options metrics
            if 'put_call_ratio' in df.columns:
                # Put-call ratio trends
                df['pcr_change'] = df['put_call_ratio'].pct_change(5)
                features.append('pcr_change')
            
            if 'call_premium_volume' in df.columns and 'put_premium_volume' in df.columns:
                # Premium imbalance
                df['premium_imbalance'] = (df['call_premium_volume'] - df['put_premium_volume']) / (df['call_premium_volume'] + df['put_premium_volume'])
                features.append('premium_imbalance')
            
            if 'gamma_exposure' in df.columns:
                # Gamma exposure change
                df['gamma_exposure_change'] = df['gamma_exposure'].pct_change(3)
                features.append('gamma_exposure_change')
            
            if 'unusual_activity_score' in df.columns:
                # Unusual activity trend
                df['unusual_activity_trend'] = df['unusual_activity_score'].rolling(window=5).mean() / df['unusual_activity_score'].rolling(window=20).mean()
                features.append('unusual_activity_trend')
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Extract features and target
        X = df[features]
        y = df['regime']
        
        return X, y
    
    def _map_regimes_to_names(self, hmm_states: np.ndarray, data: pd.DataFrame) -> Dict:
        """
        Map HMM states to meaningful regime names.
        
        Args:
            hmm_states: Array of HMM state indices
            data: Original data with market metrics
            
        Returns:
            Dictionary mapping HMM states to regime names
        """
        # Create DataFrame with states and key metrics
        state_df = pd.DataFrame({
            'state': hmm_states,
            'returns': data['returns'].values,
            'volatility': data['volatility'].values
        })
        
        # Add options metrics if available
        if 'put_call_ratio' in data.columns:
            state_df['put_call_ratio'] = data['put_call_ratio'].values
        
        if 'unusual_activity_score' in data.columns:
            state_df['unusual_activity_score'] = data['unusual_activity_score'].values
            
        if 'smart_money_direction' in data.columns:
            state_df['smart_money_direction'] = data['smart_money_direction'].values
        
        # Calculate mean metrics for each state
        state_stats = state_df.groupby('state').agg({
            'returns': 'mean',
            'volatility': 'mean'
        })
        
        # Add options metrics if available
        if 'put_call_ratio' in state_stats.columns:
            state_stats['put_call_ratio'] = state_df.groupby('state')['put_call_ratio'].mean()
            
        if 'unusual_activity_score' in state_stats.columns:
            state_stats['unusual_activity_score'] = state_df.groupby('state')['unusual_activity_score'].mean()
            
        if 'smart_money_direction' in state_stats.columns:
            state_stats['smart_money_direction'] = state_df.groupby('state')['smart_money_direction'].mean()
        
        # Initialize mapping
        mapping = {}
        states = state_stats.index.values
        
        # Enhanced regime assignment using options data if available
        if 'smart_money_direction' in state_stats.columns and 'put_call_ratio' in state_stats.columns:
            # Find bullish state (high smart money direction, low put_call_ratio)
            bullish_score = state_stats['smart_money_direction'] - state_stats['put_call_ratio']
            trending_up_state = bullish_score.idxmax()
            mapping[trending_up_state] = 'trending_up'
            
            # Find bearish state (low smart money direction, high put_call_ratio)
            bearish_score = state_stats['put_call_ratio'] - state_stats['smart_money_direction']
            trending_down_state = bearish_score.idxmax()
            mapping[trending_down_state] = 'trending_down'
            
            # Remaining states based on volatility
            remaining_states = [s for s in states if s not in mapping]
            
            if len(remaining_states) >= 2:
                volatilities = state_stats.loc[remaining_states, 'volatility']
                high_vol_state = volatilities.idxmax()
                low_vol_state = volatilities.idxmin()
                
                mapping[high_vol_state] = 'high_volatility'
                mapping[low_vol_state] = 'low_volatility'
            elif len(remaining_states) == 1:
                # Only one state left, assign based on volatility
                if state_stats.loc[remaining_states[0], 'volatility'] > state_stats['volatility'].median():
                    mapping[remaining_states[0]] = 'high_volatility'
                else:
                    mapping[remaining_states[0]] = 'low_volatility'
        else:
            # Fallback to traditional mapping without options data
            # Get state with highest mean returns
            trending_up_state = state_stats['returns'].idxmax()
            mapping[trending_up_state] = 'trending_up'
            remaining_states = [s for s in states if s != trending_up_state]
            
            # Get state with lowest mean returns
            trending_down_state = state_stats.loc[remaining_states, 'returns'].idxmin()
            mapping[trending_down_state] = 'trending_down'
            remaining_states = [s for s in remaining_states if s != trending_down_state]
            
            # Assign remaining states based on volatility
            if len(remaining_states) >= 2:
                volatilities = state_stats.loc[remaining_states, 'volatility']
                high_vol_state = volatilities.idxmax()
                low_vol_state = volatilities.idxmin()
                
                mapping[high_vol_state] = 'high_volatility'
                mapping[low_vol_state] = 'low_volatility'
            elif len(remaining_states) == 1:
                # If only one state left, assign based on volatility
                if state_stats.loc[remaining_states[0], 'volatility'] > state_stats['volatility'].median():
                    mapping[remaining_states[0]] = 'high_volatility'
                else:
                    mapping[remaining_states[0]] = 'low_volatility'
        
        # Fill in any missing states
        for i in range(self.n_regimes):
            if i not in mapping:
                for name in self.regime_names:
                    if name not in mapping.values():
                        mapping[i] = name
                        break
                # If all names are used, create a generic name
                if i not in mapping:
                    mapping[i] = f'regime_{i}'
        
        return mapping
    
    def _smooth_regimes(self, regimes: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to regime predictions to prevent frequent switching.
        
        Args:
            regimes: Array of regime labels
            
        Returns:
            Smoothed regime labels
        """
        if len(regimes) <= self.smooth_window:
            return regimes
        
        smoothed = np.copy(regimes)
        
        for i in range(len(regimes)):
            # Define window boundaries
            start_idx = max(0, i - self.smooth_window // 2)
            end_idx = min(len(regimes), i + (self.smooth_window - self.smooth_window // 2))
            
            # Get window
            window = regimes[start_idx:end_idx]
            
            # Most frequent regime in window
            values, counts = np.unique(window, return_counts=True)
            smoothed[i] = values[np.argmax(counts)]
        
        return smoothed
    
    def train_hmm(self, market_data: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Train the Hidden Markov Model for unsupervised regime detection.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Tuple of (HMM metrics dict, DataFrame with regime labels)
        """
        logger.info("Training HMM for regime detection with options flow enhancement")
        
        # Prepare features
        X = self._prepare_hmm_features(market_data)
        
        if len(X) == 0:
            logger.error("Failed to prepare features for HMM")
            return {}, pd.DataFrame()
        
        # Initialize HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.hmm_n_iter,
            random_state=42
        )
        
        # Fit HMM model
        self.hmm_model.fit(X)
        
        # Predict states
        hmm_states = self.hmm_model.predict(X)
        
        # Map states to meaningful regime names
        self.regime_mapping = self._map_regimes_to_names(hmm_states, market_data)
        
        # Store transition matrix
        self.transition_matrix = self.hmm_model.transmat_
        
        # Map numeric states to named regimes
        named_regimes = np.array([self.regime_mapping[state] for state in hmm_states])
        
        # Smooth regimes
        smoothed_regimes = self._smooth_regimes(hmm_states)
        named_smoothed_regimes = np.array([self.regime_mapping[state] for state in smoothed_regimes])
        
        # Add to data
        regime_data = market_data.copy()
        regime_data = regime_data.iloc[-(len(named_regimes)):]
        regime_data['regime_raw'] = hmm_states
        regime_data['regime'] = named_regimes
        regime_data['regime_smoothed'] = named_smoothed_regimes
        
        # Calculate metrics
        regime_stats = {}
        for regime in self.regime_mapping.values():
            mask = regime_data['regime'] == regime
            if mask.any():
                regime_stats[regime] = {
                    'count': mask.sum(),
                    'pct': mask.mean() * 100,
                    'mean_return': regime_data.loc[mask, 'returns'].mean() * 100,
                    'volatility': regime_data.loc[mask, 'volatility'].mean() * 100,
                    'avg_duration': self._calculate_avg_duration(named_regimes, regime)
                }
                
                # Add options metrics if available
                if 'put_call_ratio' in regime_data.columns:
                    regime_stats[regime]['put_call_ratio'] = regime_data.loc[mask, 'put_call_ratio'].mean()
                
                if 'unusual_activity_score' in regime_data.columns:
                    regime_stats[regime]['unusual_activity_score'] = regime_data.loc[mask, 'unusual_activity_score'].mean()
                
                if 'implied_volatility' in regime_data.columns:
                    regime_stats[regime]['implied_volatility'] = regime_data.loc[mask, 'implied_volatility'].mean()
        
        # Calculate log likelihood
        hmm_metrics = {
            'log_likelihood': self.hmm_model.score(X),
            'aic': -2 * self.hmm_model.score(X) + 2 * (self.n_regimes * self.n_regimes + 2 * self.n_regimes * X.shape[1]),
            'regime_stats': regime_stats
        }
        
        logger.info(f"HMM training completed: {hmm_metrics}")
        
        return hmm_metrics, regime_data
    
    def _calculate_avg_duration(self, regimes: np.ndarray, regime_name: str) -> float:
        """
        Calculate average duration of a regime in days.
        
        Args:
            regimes: Array of regime labels
            regime_name: Name of regime to analyze
            
        Returns:
            Average duration in days
        """
        # Convert to binary array (1 where regime matches, 0 otherwise)
        binary = (regimes == regime_name).astype(int)
        
        # Find where regime changes
        changes = np.diff(np.concatenate([[0], binary, [0]]))
        
        # Start indices are where changes == 1
        starts = np.where(changes == 1)[0]
        
        # End indices are where changes == -1
        ends = np.where(changes == -1)[0]
        
        # Calculate durations
        durations = ends - starts
        
        # Return average duration
        return durations.mean() if len(durations) > 0 else 0
    
    def train_classifier(self, regime_data: pd.DataFrame) -> Dict:
        """
        Train XGBoost classifier for supervised regime prediction.
        
        Args:
            regime_data: DataFrame with market data and regime labels
            
        Returns:
            Dict with classifier metrics
        """
        logger.info("Training XGBoost classifier for regime prediction with options flow features")
        
        # Prepare features
        X, y = self._prepare_xgb_features(regime_data)
        
        if len(X) == 0 or len(y) == 0:
            logger.error("Failed to prepare features for XGBoost")
            return {}
        
        # Encode target
        y_encoded = self.encoder.fit_transform(y.values.reshape(-1, 1))
        
        # Initialize XGBoost model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            learning_rate=self.xgb_learning_rate,
            max_depth=self.xgb_max_depth,
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            random_state=42
        )
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Cross-validation
        fold_metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            self.xgb_model.fit(X_train, y_train)
            
            # Predict
            y_pred = self.xgb_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            fold_metrics.append(accuracy)
        
        # Final fit on all data
        self.xgb_model.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, self.xgb_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Check if options flow features are in top 10
        options_in_top10 = [feature for feature, _ in top_features if any(opt in feature for opt in [
            'put_call', 'premium', 'unusual', 'smart_money', 'gamma', 'iv_skew'
        ])]
        
        # Calculate metrics
        classifier_metrics = {
            'cv_accuracy_mean': np.mean(fold_metrics),
            'cv_accuracy_std': np.std(fold_metrics),
            'top_features': dict(top_features),
            'options_flow_importance': len(options_in_top10) / 10.0 if top_features else 0.0
        }
        
        logger.info(f"XGBoost classifier training completed: {classifier_metrics}")
        
        return classifier_metrics
    
    def train(self, market_data: pd.DataFrame) -> Dict:
        """
        Train the complete market regime model.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Dict with training metrics
        """
        logger.info("Training enhanced market regime model with options flow data")
        
        # Train HMM model
        hmm_metrics, regime_data = self.train_hmm(market_data)
        
        # Train XGBoost classifier
        classifier_metrics = self.train_classifier(regime_data)
        
        # Combine metrics
        self.metrics = {
            'hmm_metrics': hmm_metrics,
            'classifier_metrics': classifier_metrics
        }
        
        return self.metrics
    
    def predict_regime(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market regime for new data.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with regime predictions
        """
        logger.info("Predicting market regimes with options flow enhancement")
        
        if self.hmm_model is None or self.xgb_model is None:
            logger.error("Models not trained yet")
            return pd.DataFrame()
        
        # Prepare data for prediction
        result_data = market_data.copy()
        
        # Calculate returns if not present
        if 'returns' not in result_data.columns:
            result_data['returns'] = result_data['close'].pct_change()
            result_data = result_data.dropna()
        
        # Calculate volatility if not present
        if 'volatility' not in result_data.columns:
            result_data['volatility'] = result_data['returns'].rolling(window=21).std() * np.sqrt(252)
            result_data = result_data.dropna()
        
        # Prepare HMM features
        X_hmm = self._prepare_hmm_features(result_data)
        
        if len(X_hmm) == 0:
            logger.error("Failed to prepare features for HMM prediction")
            return pd.DataFrame()
        
        # Predict HMM states
        hmm_states = self.hmm_model.predict(X_hmm)
        
        # Map to named regimes
        named_regimes = np.array([self.regime_mapping[state] for state in hmm_states])
        
        # Smooth regimes
        smoothed_states = self._smooth_regimes(hmm_states)
        named_smoothed_regimes = np.array([self.regime_mapping[state] for state in smoothed_states])
        
        # Add to result data
        result_data = result_data.iloc[-(len(named_regimes)):]
        result_data['regime_hmm'] = named_regimes
        result_data['regime_hmm_smoothed'] = named_smoothed_regimes
        
        # Prepare features for XGBoost
        # Add dummy regime column for feature preparation
        result_data['regime'] = named_regimes
        X_xgb, _ = self._prepare_xgb_features(result_data)
        
        if not X_xgb.empty:
            # Predict with XGBoost
            xgb_pred = self.xgb_model.predict(X_xgb)
            xgb_pred_proba = self.xgb_model.predict_proba(X_xgb)
            
            # Add to result data
            result_data['regime_xgb'] = xgb_pred
            
            # Add probability for each regime
            regime_names = self.encoder.categories_[0]
            for i, regime in enumerate(regime_names):
                result_data[f'prob_{regime}'] = xgb_pred_proba[:, i]
            
            # Final regime prediction - enhanced decision logic
            # If options data is available, weigh XGBoost prediction higher
            if any(col in result_data.columns for col in [
                'put_call_ratio', 'unusual_activity_score', 'smart_money_direction'
            ]):
                # Use XGBoost prediction with high confidence
                result_data['regime'] = result_data['regime_xgb']
                logger.info("Using XGBoost with options data for final regime prediction")
            else:
                # Without options data, use a blend of HMM and XGBoost
                # Compare HMM and XGBoost predictions
                result_data['regime_match'] = result_data['regime_hmm_smoothed'] == result_data['regime_xgb']
                
                # Where they match, use that regime
                # Where they don't match, use the one with higher probability
                for idx, row in result_data.iterrows():
                    if not row['regime_match']:
                        hmm_regime = row['regime_hmm_smoothed']
                        xgb_regime = row['regime_xgb']
                        
                        # Get XGBoost probability for its prediction
                        xgb_prob = row[f'prob_{xgb_regime}']
                        
                        # If XGBoost is very confident, use it
                        if xgb_prob > 0.7:
                            result_data.at[idx, 'regime'] = xgb_regime
                        else:
                            # Otherwise use HMM's smoothed prediction
                            result_data.at[idx, 'regime'] = hmm_regime
                    else:
                        # They match, so use either one (they're the same)
                        result_data.at[idx, 'regime'] = row['regime_xgb']
        
        return result_data
    
    def get_regime_characteristics(self) -> Dict[str, Dict[str, float]]:
        """
        Get characteristics of each market regime.
        
        Returns:
            Dictionary mapping regime names to their characteristics
        """
        logger.info("Getting regime characteristics")
        
        # Initialize regime characteristics
        regime_characteristics = {}
        
        # If regime mapping is not available, return empty dict
        if not hasattr(self, 'regime_mapping') or not self.regime_mapping:
            logger.warning("No regime mapping available")
            return {}
        
        # Get unique regimes from mapping
        regimes = set(self.regime_mapping.values())
        
        # Initialize characteristics for each regime
        for regime in regimes:
            regime_characteristics[regime] = {
                'mean_return': 0.0,
                'volatility': 0.0,
                'avg_duration': 0.0,
                'frequency': 0.0
            }
            
            # Add options flow metrics if available
            for metric in ['put_call_ratio', 'unusual_activity_score', 'implied_volatility']:
                regime_characteristics[regime][metric] = 0.0
        
        # If we have regime data, calculate actual characteristics
        if hasattr(self, 'metrics') and 'hmm_metrics' in self.metrics:
            hmm_metrics = self.metrics['hmm_metrics']
            
            if 'regime_stats' in hmm_metrics:
                regime_stats = hmm_metrics['regime_stats']
                
                # Update characteristics with actual values
                for regime, stats in regime_stats.items():
                    if regime in regime_characteristics:
                        regime_characteristics[regime].update(stats)
        
        return regime_characteristics
    
    def predict_realtime(self, symbol: str = 'SPY') -> Dict[str, Any]:
        """
        Make a real-time market regime prediction using data from Redis.
        
        Args:
            symbol: Market index symbol to use for regime detection (default: SPY)
            
        Returns:
            Dictionary with regime prediction results
        """
        from data.processors.realtime_data_provider import RealtimeDataProvider
        
        logger.info(f"Making real-time market regime prediction using {symbol}")
        
        if self.hmm_model is None or self.xgb_model is None:
            logger.error("Models not trained yet")
            return {
                'timestamp': datetime.now(),
                'error': "Models not trained"
            }
        
        # Get real-time OHLCV data
        df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, limit=self.lookback_window)
        
        if df.empty:
            logger.warning(f"No real-time data available for {symbol}")
            return {
                'timestamp': datetime.now(),
                'error': "No real-time data available"
            }
        
        # Check if we have enough data
        if len(df) < self.lookback_window:
            logger.warning(f"Not enough real-time data (need {self.lookback_window}, got {len(df)})")
            return {
                'timestamp': datetime.now(),
                'error': f"Not enough data (need {self.lookback_window}, got {len(df)})"
            }
        
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df = df.dropna()
        
        # Calculate volatility if not present
        if 'volatility' not in df.columns:
            df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
            df = df.dropna()
        
        # Try to get options flow data from Unusual Whales if available
        try:
            options_data = RealtimeDataProvider.get_options_flow_data(symbol, limit=30)
            
            if not options_data.empty:
                logger.info(f"Found options flow data for {symbol}")
                
                # Calculate options flow metrics
                if 'premium' in options_data.columns and 'type' in options_data.columns:
                    # Calculate put/call ratio
                    calls = options_data[options_data['type'] == 'call']['premium'].sum()
                    puts = options_data[options_data['type'] == 'put']['premium'].sum()
                    
                    if calls > 0:
                        put_call_ratio = puts / calls
                    else:
                        put_call_ratio = 1.0
                    
                    # Add to dataframe
                    df['put_call_ratio'] = put_call_ratio
                
                # Add other options metrics if available
                if 'sentiment' in options_data.columns:
                    # Average sentiment (0-100 scale)
                    avg_sentiment = options_data['sentiment'].mean()
                    df['smart_money_direction'] = avg_sentiment / 100.0
                
                if 'iv' in options_data.columns:
                    # Average implied volatility
                    df['implied_volatility'] = options_data['iv'].mean()
                
                # Calculate unusual activity score based on volume/OI ratio
                if 'volumeopeninterest' in options_data.columns:
                    df['unusual_activity_score'] = options_data['volumeopeninterest'].mean()
        
        except Exception as e:
            logger.warning(f"Could not get options flow data: {str(e)}")
            # Continue without options data
        
        # Make prediction using the predict_regime method
        result = self.predict_regime(df)
        
        if result.empty:
            logger.error("Failed to predict regime")
            return {
                'timestamp': datetime.now(),
                'error': "Failed to predict regime"
            }
        
        # Get the latest regime prediction
        latest = result.iloc[-1]
        
        # Create result dictionary
        regime_result = {
            'timestamp': latest.get('timestamp', datetime.now()),
            'regime': latest.get('regime', 'unknown'),
            'regime_hmm': latest.get('regime_hmm', 'unknown'),
            'regime_xgb': latest.get('regime_xgb', 'unknown')
        }
        
        # Add probabilities for each regime if available
        for regime in self.regime_names:
            prob_col = f'prob_{regime}'
            if prob_col in latest:
                regime_result[prob_col] = float(latest[prob_col])
        
        # Add market metrics
        regime_result['market_return'] = float(latest.get('returns', 0))
        regime_result['market_volatility'] = float(latest.get('volatility', 0))
        
        # Add options flow metrics if available
        if 'put_call_ratio' in latest:
            regime_result['put_call_ratio'] = float(latest['put_call_ratio'])
        
        if 'smart_money_direction' in latest:
            regime_result['smart_money_direction'] = float(latest['smart_money_direction'])
        
        if 'implied_volatility' in latest:
            regime_result['implied_volatility'] = float(latest['implied_volatility'])
        
        if 'unusual_activity_score' in latest:
            regime_result['unusual_activity_score'] = float(latest['unusual_activity_score'])
        
        # Add trading implications based on regime
        regime = regime_result['regime']
        if regime == 'trending_up':
            regime_result['trading_bias'] = 'bullish'
            regime_result['volatility_expectation'] = 'low to moderate'
            regime_result['suggested_strategy'] = 'trend-following'
        elif regime == 'trending_down':
            regime_result['trading_bias'] = 'bearish'
            regime_result['volatility_expectation'] = 'moderate to high'
            regime_result['suggested_strategy'] = 'trend-following with hedges'
        elif regime == 'high_volatility':
            regime_result['trading_bias'] = 'neutral'
            regime_result['volatility_expectation'] = 'high'
            regime_result['suggested_strategy'] = 'mean-reversion with tight stops'
        elif regime == 'low_volatility':
            regime_result['trading_bias'] = 'neutral'
            regime_result['volatility_expectation'] = 'low'
            regime_result['suggested_strategy'] = 'range-bound trading'
        
        logger.info(f"Real-time regime prediction: {regime} with {regime_result.get('prob_' + regime, 0):.2f} confidence")
        
        return regime_result
