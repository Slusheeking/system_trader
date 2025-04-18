import pandas as pd
import pytest

import backtesting.engine as engine_mod
from backtesting.engine import BacktestingEngine


class FakeStrategy:
    """
    Fake strategy that forwards positions to entries and exits to exits.
    """
    def process_signals(self, sig):
        return {"entries": sig.get("positions", []), "exits": sig.get("exits", [])}


class FakeComposer:
    """
    Fake strategy composer that returns FakeStrategy and emits predefined signals.
    """
    def __init__(self, config_path):
        pass

    def get_strategy(self, name):
        return FakeStrategy()

    def generate_signals(self, data):
        # Build a DataFrame with one entry signal and one exit signal
        return pd.DataFrame([
            {
                "positions": [{"symbol": "X", "qty": 1}],
                "exits": [],
                "regime": None,
                "timestamp": data["timestamp"].iloc[0]
            },
            {
                "positions": [],
                "exits": [{"symbol": "X", "qty": 1}],
                "regime": None,
                "timestamp": data["timestamp"].iloc[1]
            }
        ])


def test_run_backtest_creates_expected_trades(tmp_path, monkeypatch):
    # Prepare a minimal CSV file with timestamps
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"timestamp": ["2021-01-01", "2021-01-02"]})
    df.to_csv(csv_file, index=False)

    # Stub ConfigLoader.load to return our config dict
    config = {
        "data": {"path": str(csv_file)},
        "strategy": {"name": "test_strategy", "config_path": "unused"}
    }
    monkeypatch.setattr(
        engine_mod.ConfigLoader,
        "load",
        staticmethod(lambda path: config)
    )

    # Stub get_strategy_composer to return our fake composer
    monkeypatch.setattr(
        engine_mod,
        "get_strategy_composer",
        lambda config_path: FakeComposer(config_path)
    )

    # Instantiate engine and run backtest
    engine = BacktestingEngine(config_path="dummy_path")
    trades = engine.run_backtest()

    # Validate the output DataFrame
    assert isinstance(trades, pd.DataFrame)
    assert len(trades) == 2

    # First record should be an entry
    first = trades.iloc[0]
    assert first["type"] == "entry"
    assert first["symbol"] == "X"
    assert first["qty"] == 1
    assert pd.to_datetime(first["timestamp"]) == pd.to_datetime("2021-01-01")

    # Second record should be an exit
    second = trades.iloc[1]
    assert second["type"] == "exit"
    assert second["symbol"] == "X"
    assert second["qty"] == 1
    assert pd.to_datetime(second["timestamp"]) == pd.to_datetime("2021-01-02")
