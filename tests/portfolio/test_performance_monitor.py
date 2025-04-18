import pytest
import pandas as pd
import numpy as np
import pandas.testing as pdt

from portfolio.performance_monitor import PerformanceMonitor


def definitely_float_close(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) < tol


@pytest.fixture
def price_df() -> pd.DataFrame:
    return pd.DataFrame({'price': [100.0, 110.0, 105.0, 115.0]})


@pytest.fixture
def returns_series(price_df) -> pd.Series:
    # Synthetic returns from price series
    return price_df['price'].pct_change().dropna()


def test_compute_returns(price_df, returns_series):
    monitor = PerformanceMonitor()
    computed = monitor.compute_returns(price_df)
    # Expect returns equal to pct_change on the 'price' column
    pdt.assert_series_equal(computed, returns_series)
    # Ensure internal state is updated
    pdt.assert_series_equal(monitor.returns, returns_series)


def test_compute_sharpe(returns_series):
    monitor = PerformanceMonitor()
    # Compute expected Sharpe ratio
    mean_ret = returns_series.mean() - 0.0 / 252
    std_ret = returns_series.std(ddof=1)
    expected_sharpe = (mean_ret / std_ret) * np.sqrt(252)
    computed_sharpe = monitor.compute_sharpe(returns_series)
    assert definitely_float_close(computed_sharpe, expected_sharpe)
    assert definitely_float_close(monitor.sharpe_ratio, expected_sharpe)


def test_compute_max_drawdown(returns_series):
    monitor = PerformanceMonitor()
    # Manual drawdown calculation
    wealth_index = (1 + returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    expected_max_dd = drawdowns.min()
    computed_dd = monitor.compute_max_drawdown(returns_series)
    assert definitely_float_close(computed_dd, expected_max_dd)
    assert definitely_float_close(monitor.max_drawdown, expected_max_dd)


def test_generate_report(price_df, returns_series):
    monitor = PerformanceMonitor()
    # Setup metrics
    monitor.compute_returns(price_df)
    monitor.compute_sharpe(returns_series)
    monitor.compute_max_drawdown(returns_series)
    report = monitor.generate_report()

    # Validate keys
    assert set(report.keys()) == {'returns', 'sharpe_ratio', 'max_drawdown'}
    # Validate values
    pdt.assert_series_equal(report['returns'], monitor.returns)
    assert isinstance(report['sharpe_ratio'], float)
    assert isinstance(report['max_drawdown'], float)
    assert report['sharpe_ratio'] == monitor.sharpe_ratio
    assert report['max_drawdown'] == monitor.max_drawdown
