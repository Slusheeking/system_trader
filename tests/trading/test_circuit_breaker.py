import pytest
from trading.execution.circuit_breaker import CircuitBreaker, CircuitBreakerException

@pytest.fixture
def cb():
    """
    Provides a fresh CircuitBreaker instance for each test.
    """
    return CircuitBreaker()


def test_check_passes_within_thresholds(cb):
    """
    Should not raise any exception when metrics are within default thresholds.
    """
    metrics = {
        'max_drawdown': 0.10,  # below default 0.15
        'sharpe_ratio': 1.5,   # above default 0.0
        'error_rate': 0.02     # below default 0.05
    }
    # Expect no exception
    cb.check(metrics)


def test_raises_drawdown_exceeded(cb):
    """
    Should raise CircuitBreakerException for drawdown above default threshold.
    """
    metrics = {'max_drawdown': 0.20}  # above default 0.15
    with pytest.raises(CircuitBreakerException) as exc_info:
        cb.check(metrics)
    assert "Maximum drawdown threshold exceeded" in str(exc_info.value)


def test_raises_sharpe_ratio_below_threshold(cb):
    """
    Should raise CircuitBreakerException for sharpe ratio below default threshold.
    """
    metrics = {'sharpe_ratio': -0.5}  # below default 0.0
    with pytest.raises(CircuitBreakerException) as exc_info:
        cb.check(metrics)
    assert "Sharpe ratio below threshold" in str(exc_info.value)


def test_raises_error_rate_exceeded(cb):
    """
    Should raise CircuitBreakerException for error rate above default threshold.
    """
    metrics = {'error_rate': 0.10}  # above default 0.05
    with pytest.raises(CircuitBreakerException) as exc_info:
        cb.check(metrics)
    assert "Error rate threshold exceeded" in str(exc_info.value)


def test_drawdown_takes_priority_over_sharpe_and_error(cb):
    """
    When multiple metrics exceed thresholds, drawdown check occurs first and triggers the exception.
    """
    metrics = {
        'max_drawdown': 0.20,  # above default 0.15
        'sharpe_ratio': -1.0,  # below default 0.0
        'error_rate': 0.10     # above default 0.05
    }
    with pytest.raises(CircuitBreakerException) as exc_info:
        cb.check(metrics)
    assert "Maximum drawdown threshold exceeded" in str(exc_info.value)
