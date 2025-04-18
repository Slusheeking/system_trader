import os
import sys
import json
import types
import pytest
import numpy as np
import pandas as pd
from backtesting.performance_analyzer import PerformanceAnalyzer


def sample_trades_df():
    timestamps = pd.date_range("2021-01-01", periods=5, freq="D")
    pnl = [10, -5, 15, -10, 20]
    df = pd.DataFrame({"timestamp": timestamps, "pnl": pnl})
    return df


def test_compute_metrics_basic():
    df = sample_trades_df()
    pa = PerformanceAnalyzer()
    metrics = pa.compute_metrics(df)

    # Basic counts
    assert metrics['total_trades'] == 5
    assert metrics['winning_trades'] == 3
    assert metrics['losing_trades'] == 2
    assert metrics['win_rate'] == pytest.approx(0.6)

    # Expected Sharpe ratio
    pnl = np.array([10, -5, 15, -10, 20])
    equity = 100000 + np.cumsum(pnl)
    returns = pd.Series(equity).pct_change().fillna(0)
    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std()
    assert metrics['sharpe_ratio'] == pytest.approx(expected_sharpe, rel=1e-6)

    # Expected maximum drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    expected_max_dd = float(drawdowns.min())
    assert metrics['max_drawdown'] == pytest.approx(expected_max_dd, rel=1e-6)


def test_generate_reports_creates_files(tmp_path):
    df = sample_trades_df()
    pa = PerformanceAnalyzer('test_strategy')
    # Compute metrics to populate trades_df and metrics
    pa.compute_metrics(df)
    output_dir = tmp_path / "reports"
    report_files = pa.generate_reports(str(output_dir))

    # Check that each reported file exists
    for key, filepath in report_files.items():
        fname = os.path.basename(filepath)
        full_path = output_dir / fname
        assert full_path.exists(), f"Expected report file {full_path} to exist"


def test_generate_reports_mlflow(monkeypatch, tmp_path):
    df = sample_trades_df()
    pa = PerformanceAnalyzer('mlflow_strategy')
    pa.compute_metrics(df)
    output_dir = tmp_path / "reports_mlflow"

    # Create a fake mlflow module
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.logged_metrics = []
    fake_mlflow.logged_artifacts = []

    class FakeRun:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            return False

    def start_run(run_name=None):
        fake_mlflow.run_name = run_name
        return FakeRun()

    def log_metric(key, value):
        fake_mlflow.logged_metrics.append((key, value))

    def log_artifact(path):
        fake_mlflow.logged_artifacts.append(path)

    fake_mlflow.start_run = start_run
    fake_mlflow.log_metric = log_metric
    fake_mlflow.log_artifact = log_artifact

    # Inject fake mlflow into sys.modules
    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)

    # Generate reports (should invoke mlflow logging)
    pa.generate_reports(str(output_dir))

    # Verify that metrics and artifacts were logged
    assert fake_mlflow.logged_metrics, "Expected metrics to be logged to MLflow"
    # There should be one artifact per generated report file
    assert len(fake_mlflow.logged_artifacts) == len(os.listdir(output_dir)), \
        "Expected all report files to be logged as artifacts"