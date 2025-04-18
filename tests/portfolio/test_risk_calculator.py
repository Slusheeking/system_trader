import pytest
import numpy as np
import pandas as pd

from portfolio.risk_calculator import RiskCalculator


def test_calculate_var_accuracy():
    # Known P&L values: losses = -pnl => [1,2,3,4], 50th percentile = 2.5
    positions = {
        'A': -1.0,
        'B': -2.0,
        'C': -3.0,
        'D': -4.0
    }
    var_value = RiskCalculator.calculate_var(positions, confidence=0.5)
    assert var_value == pytest.approx(2.5, rel=1e-6)


def test_calculate_es_accuracy():
    # For the same data, losses = [1,2,3,4], var_threshold = 2.5, tail_losses = [3,4], ES = 3.5
    positions = {
        'A': -1.0,
        'B': -2.0,
        'C': -3.0,
        'D': -4.0
    }
    es_value = RiskCalculator.calculate_es(positions, confidence=0.5)
    assert es_value == pytest.approx(3.5, rel=1e-6)


def test_calculate_var_empty_positions_raises():
    with pytest.raises(ValueError) as excinfo:
        RiskCalculator.calculate_var({}, confidence=0.99)
    assert "No position data provided for VaR calculation" in str(excinfo.value)


def test_calculate_es_empty_positions_raises():
    with pytest.raises(ValueError) as excinfo:
        RiskCalculator.calculate_es({}, confidence=0.99)
    assert "No position data provided for ES calculation" in str(excinfo.value)


def test_calculate_portfolio_risk_empty_market_data_raises():
    positions = {'A': 100.0}
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError) as excinfo:
        RiskCalculator.calculate_portfolio_risk(positions, empty_df)
    assert "Market data is empty for portfolio risk calculation" in str(excinfo.value)


def test_calculate_portfolio_risk_missing_position_raises():
    # DataFrame has symbol not in positions
    positions = {'A': 100.0}
    market_data = pd.DataFrame({
        'B': [100.0, 101.0, 102.0]
    })
    with pytest.raises(ValueError) as excinfo:
        RiskCalculator.calculate_portfolio_risk(positions, market_data)
    assert "Position for symbol 'B' not provided" in str(excinfo.value)
