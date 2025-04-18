import pytest
import numpy as np
from portfolio.diversification_engine import DiversificationEngine


def test_optimize_weights_with_mock(monkeypatch):
    # Setup dummy expected returns and covariance matrix
    expected_returns = {'Asset1': 0.1, 'Asset2': 0.2}
    cov_matrix = {
        'Asset1': {'Asset1': 1.0, 'Asset2': 0.1},
        'Asset2': {'Asset1': 0.1, 'Asset2': 1.0}
    }
    engine = DiversificationEngine(expected_returns, cov_matrix)

    # Create a fake result with success and preset weights
    x = np.array([0.7, 0.3])
    DummyResult = type('DummyResult', (), {'success': True, 'x': x, 'message': ''})

    # Monkeypatch the minimize function in the module
    def fake_minimize(func, x0, method, bounds, constraints):
        return DummyResult

    monkeypatch.setattr(
        'portfolio.diversification_engine.minimize',
        fake_minimize
    )

    # Execute optimization
    weights = engine.optimize_weights(risk_aversion=1.0)

    # Assertions on output shape and constraints
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(expected_returns.keys())
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    for w in weights.values():
        assert 0.0 <= w <= 1.0
    assert weights['Asset1'] == pytest.approx(0.7)
    assert weights['Asset2'] == pytest.approx(0.3)


def test_optimize_weights_solver_failure(monkeypatch):
    # Setup minimal data
    expected_returns = {'A': 0.0, 'B': 0.0}
    cov_matrix = {
        'A': {'A': 1.0, 'B': 0.0},
        'B': {'A': 0.0, 'B': 1.0}
    }
    engine = DiversificationEngine(expected_returns, cov_matrix)

    # Create a fake result indicating failure
    FailureResult = type('FailureResult', (), {'success': False, 'x': None, 'message': 'optimization error'})

    def fake_minimize(func, x0, method, bounds, constraints):
        return FailureResult

    monkeypatch.setattr(
        'portfolio.diversification_engine.minimize',
        fake_minimize
    )

    # Expect a RuntimeError when optimization fails
    with pytest.raises(RuntimeError) as excinfo:
        engine.optimize_weights(risk_aversion=1.0)
    assert 'Portfolio optimization failed' in str(excinfo.value)


def test_rebalance_logic():
    current_positions = {'A': 0.4, 'B': 0.6}
    target_weights = {'A': 0.5, 'B': 0.5, 'C': 0.0}
    engine = DiversificationEngine({}, {})
    adjustments = engine.rebalance(current_positions, target_weights)

    # Expected adjustments: target - current
    assert adjustments == {'A': pytest.approx(0.1), 'B': pytest.approx(-0.1), 'C': pytest.approx(0.0)}
