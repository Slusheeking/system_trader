import logging

import pytest
import schedule

import scheduler.tasks.backtesting_task as bt_mod
from scheduler.tasks.backtesting_task import BacktestingTask


def setup_function(function):
    # Clear scheduled jobs before each test
    schedule.clear()


def test_schedule_daily_registers_daily_job():
    cfg = {'backtesting': {'schedule': {'frequency': 'daily', 'time': '10:20'}}}
    BacktestingTask.schedule(cfg)
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for daily frequency"
    job = jobs[0]
    assert job.interval == 1, "Daily job should have interval of 1"
    assert job.unit == 'days', "Daily job unit should be 'days'"
    assert job.at_time == '10:20', "Daily job time should match the configured time"


def test_schedule_hourly_registers_hourly_job():
    cfg = {'backtesting': {'schedule': {'frequency': 'hourly', 'time': ':15'}}}
    BacktestingTask.schedule(cfg)
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for hourly frequency"
    job = jobs[0]
    assert job.interval == 1, "Hourly job should have interval of 1"
    assert job.unit == 'hours', "Hourly job unit should be 'hours'"
    assert job.at_time == ':15', "Hourly job time should match the configured time"


def test_run_success_returns_and_logs_metrics(monkeypatch, caplog):
    # Prepare dummy metrics
    dummy_metrics = {'profit': 100, 'trades': 5}

    # Stub the BacktestingEngine to return dummy metrics
    class DummyEngine:
        def __init__(self, **kwargs):
            pass

        def run_backtest(self):
            return dummy_metrics

    monkeypatch.setattr(bt_mod, 'BacktestingEngine', DummyEngine)

    # Capture logs at INFO level
    caplog.set_level(logging.INFO)

    # Execute the backtesting task
    cfg = {'backtesting': {'params': {'foo': 'bar'}}}
    result = BacktestingTask().run(cfg)

    # Verify the returned metrics
    assert result == dummy_metrics, "run() should return the metrics from the engine"

    # Verify log messages
    assert "Backtesting completed successfully." in caplog.text
    assert f"Summary metrics: {dummy_metrics}" in caplog.text
