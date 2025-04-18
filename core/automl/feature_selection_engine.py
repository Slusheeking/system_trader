# -*- coding: utf-8 -*-
"""
Feature Selection Engine

Provides automated feature selection using RFE, L1-based selection, or Boruta.
Configuration is passed via the automl.feature_selection section of the config.
"""
import logging
from typing import Dict, Optional

import pandas as pd

from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try to import BorutaPy, set to None if not available
try:
    from boruta import BorutaPy
except ImportError:
    BorutaPy = None


logger = logging.getLogger(__name__)


class FeatureSelectionEngine:
    """
    Feature Selection Engine supporting multiple selection methods.
    Methods and parameters are configured via config dict:

    config = {
        'method': 'rfe',  # or 'l1', 'boruta'
        'selection_methods': {
            'rfe': {
                'n_features_to_select': 10,
                'step': 1,
                'estimator': LogisticRegression,  # class or callable
                'estimator_params': {}
            },
            'l1': {
                'C': 1.0,
                'solver': 'liblinear',
                'max_iter': 100
            },
            'boruta': {
                'n_estimators': 100,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    """

    def __init__(self, config: Dict):
        self.method = config.get('method', 'rfe').lower()
        self.methods_config = config.get('selection_methods', {}) or {}

    def select_features(self, data: pd.DataFrame, target: str) -> Optional[pd.DataFrame]:
        """
        Select features from the input dataframe.

        Args:
            data: DataFrame containing features and target column
            target: Name of the target column

        Returns:
            Reduced DataFrame containing only selected features and the target column
        """
        if target not in data.columns:
            logger.error(f"Target column '{target}' not found in data")
            return None

        X = data.drop(columns=[target])
        y = data[target]

        selected_features = []

        try:
            if self.method == 'rfe':
                params = self.methods_config.get('rfe', {})
                n_features = params.get('n_features_to_select', None)
                step = params.get('step', 1)
                estimator_cls = params.get('estimator', LogisticRegression)
                estimator_params = params.get('estimator_params', {})
                estimator = estimator_cls(**estimator_params)

                selector = RFE(
                    estimator=estimator,
                    n_features_to_select=n_features,
                    step=step
                )
                selector = selector.fit(X, y)
                mask = selector.support_
                selected_features = X.columns[mask].tolist()

            elif self.method == 'l1':
                params = self.methods_config.get('l1', {})
                C = params.get('C', 1.0)
                solver = params.get('solver', 'liblinear')
                max_iter = params.get('max_iter', 100)
                lr = LogisticRegression(
                    penalty='l1',
                    C=C,
                    solver=solver,
                    max_iter=max_iter
                )
                selector = SelectFromModel(estimator=lr)
                selector = selector.fit(X, y)
                mask = selector.get_support()
                selected_features = X.columns[mask].tolist()

            elif self.method == 'boruta':
                if BorutaPy is None:
                    logger.error("BorutaPy is not installed, cannot use 'boruta' method")
                    return data

                params = self.methods_config.get('boruta', {})
                n_estimators = params.get('n_estimators', 100)
                max_iter = params.get('max_iter', 100)
                random_state = params.get('random_state', None)

                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1
                )
                selector = BorutaPy(
                    estimator=rf,
                    n_estimators='auto',
                    max_iter=max_iter,
                    random_state=random_state
                )
                selector.fit(X.values, y.values)
                mask = selector.support_
                selected_features = X.columns[mask].tolist()

            else:
                logger.warning(f"Unknown feature selection method '{self.method}', returning original data")
                return data

        except Exception as e:
            logger.error(f"Error during feature selection ({self.method}): {e}")
            return data

        if not selected_features:
            logger.warning(f"No features selected using method '{self.method}', returning original data")
            return data

        # Return reduced DataFrame
        return data[selected_features + [target]]
