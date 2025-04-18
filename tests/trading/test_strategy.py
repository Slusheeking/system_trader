import pandas as pd
import pytest

from trading.strategy import StrategyComposer


class DummyExtractor:
    """
    Dummy feature extractor that returns a preset DataFrame.
    """
    def __init__(self, features_df: pd.DataFrame):
        self._features_df = features_df

    def generate_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # Ignore raw_df and return the preset features DataFrame
        return self._features_df


class DummyModel:
    """
    Dummy market regime model that returns a preset DataFrame of regime predictions.
    """
    def __init__(self, regime_df: pd.DataFrame):
        self._regime_df = regime_df

    def predict_regime(self, features: pd.DataFrame) -> pd.DataFrame:
        # Ignore features and return the preset regime DataFrame
        return self._regime_df


def test_generate_signals_with_monkeypatched_components():
    # Prepare dummy raw market data (contents are irrelevant because extractor is mocked)
    raw_df = pd.DataFrame({'dummy_col': [1, 2, 3]})

    # Prepare dummy features DataFrame
    feat_df = pd.DataFrame({'feat1': [10, 20, 30]})

    # Prepare dummy regime DataFrame with index as timestamps
    idx = pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
    regime_df = pd.DataFrame({
        'regime': ['up', 'down', 'flat'],
        'regime_hmm': ['u_hmm', 'd_hmm', 'f_hmm'],
        'regime_xgb': ['u_xgb', 'd_xgb', 'f_xgb'],
        'prob_trending_up': [0.1, 0.2, 0.3],
        'prob_low_volatility': [0.4, 0.5, 0.6],
    }, index=idx)

    # Instantiate StrategyComposer without calling its __init__
    composer = StrategyComposer.__new__(StrategyComposer)

    # Monkeypatch the feature extractor and regime model
    composer.feature_extractor = DummyExtractor(feat_df)
    composer.market_regime_model = DummyModel(regime_df)

    # Initialize model_metrics for capturing updates
    composer.model_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'predictions_count': 0
    }

    # Call generate_signals and capture the output
    result = composer.generate_signals(raw_df)

    # Verify the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Expected columns in order
    expected_cols = [
        'timestamp',
        'regime',
        'regime_hmm',
        'regime_xgb',
        'prob_trending_up',
        'prob_low_volatility',
    ]
    assert list(result.columns) == expected_cols

    # Verify the timestamp column matches the regime DataFrame's index
    assert list(result['timestamp']) == list(idx)

    # Verify regime and regime variant columns
    assert result['regime'].tolist() == ['up', 'down', 'flat']
    assert result['regime_hmm'].tolist() == ['u_hmm', 'd_hmm', 'f_hmm']
    assert result['regime_xgb'].tolist() == ['u_xgb', 'd_xgb', 'f_xgb']

    # Verify probability columns
    assert result['prob_trending_up'].tolist() == [0.1, 0.2, 0.3]
    assert result['prob_low_volatility'].tolist() == [0.4, 0.5, 0.6]

    # Ensure model_metrics was updated correctly (predictions_count incremented by number of rows)
    assert composer.model_metrics['predictions_count'] == len(regime_df)
