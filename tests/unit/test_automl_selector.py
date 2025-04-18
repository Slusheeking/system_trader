import pytest
import pandas as pd
from types import SimpleNamespace
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from system_trader.core.automl.feature_selection_engine import FeatureSelectionEngine
from system_trader.core.automl.automl_selector import AutoMLSelector
import system_trader.core.automl.automl_selector as automl_selector_module


def test_feature_selection_rfe():
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df['target'] = y

    # Configure RFE to select exactly 3 features
    config = {
        'method': 'rfe',
        'selection_methods': {
            'rfe': {
                'n_features_to_select': 3,
                'step': 1,
                'estimator': LogisticRegression,
                'estimator_params': {'random_state': 42}
            }
        }
    }
    engine = FeatureSelectionEngine(config)
    reduced = engine.select_features(df, 'target')

    # Should have reduced features plus the target column
    assert reduced is not None
    assert 'target' in reduced.columns
    assert reduced.shape[1] == 4  # 3 selected + 1 target


def test_feature_selection_l1():
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=4,
        n_redundant=0,
        random_state=0
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df['target'] = y

    # Configure L1-based selection
    config = {
        'method': 'l1',
        'selection_methods': {
            'l1': {
                'C': 0.5,
                'solver': 'liblinear',
                'max_iter': 200
            }
        }
    }
    engine = FeatureSelectionEngine(config)
    reduced = engine.select_features(df, 'target')

    # Ensure at least one feature is selected and less than original
    n_selected = reduced.shape[1] - 1
    assert reduced is not None
    assert n_selected > 0
    assert n_selected < 10


class DummyASClassifier:
    called_with = {}

    def __init__(self, time_left_for_this_task, per_run_time_limit):
        DummyASClassifier.called_with = {
            'time_left_for_this_task': time_left_for_this_task,
            'per_run_time_limit': per_run_time_limit
        }

    def fit(self, X, y):
        # No-op fit
        return self

    def get_params(self):
        # Return a dummy set of hyperparameters
        return {'dummy_param': 'dummy_value'}


def test_search_architecture_classical_with_explicit_limits(monkeypatch):
    # Stub the auto-sklearn module in automl_selector
    fake_asc = SimpleNamespace(AutoSklearnClassifier=DummyASClassifier)
    monkeypatch.setattr(automl_selector_module, 'asc', fake_asc)

    # Generate synthetic data
    X, y = make_classification(n_samples=50, n_features=5, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df['target'] = y

    # Config with explicit time limits
    config = {
        'classical': {
            'library': 'auto-sklearn',
            'time_limits': 10,
            'per_run_time_limit': 2
        }
    }
    selector = AutoMLSelector(config)
    result = selector.search_architecture(df, 'target', model_type='stock_selection')

    # Check that dummy classifier was called with explicit limits
    assert DummyASClassifier.called_with['time_left_for_this_task'] == 10
    assert DummyASClassifier.called_with['per_run_time_limit'] == 2

    # Result should be the dummy hyperparameters
    assert isinstance(result, dict)
    assert result == {'dummy_param': 'dummy_value'}


def test_search_architecture_classical_with_default_limits(monkeypatch):
    # Stub the auto-sklearn module again
    fake_asc = SimpleNamespace(AutoSklearnClassifier=DummyASClassifier)
    monkeypatch.setattr(automl_selector_module, 'asc', fake_asc)

    # Generate synthetic data
    X, y = make_classification(n_samples=60, n_features=8, random_state=2)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df['target'] = y

    # Config without per_run_time_limit to trigger default calculation
    config = {
        'classical': {
            'library': 'auto-sklearn',
            'time_limits': 100
        }
    }
    selector = AutoMLSelector(config)
    result = selector.search_architecture(df, 'target', model_type='stock_selection')

    # Default per_run_time_limit = max(60, time_limits // 10)
    expected_default = max(60, 100 // 10)
    assert DummyASClassifier.called_with['time_left_for_this_task'] == 100
    assert DummyASClassifier.called_with['per_run_time_limit'] == expected_default

    # Result should still be the dummy hyperparameters
    assert isinstance(result, dict)
    assert result == {'dummy_param': 'dummy_value'}
