import logging
import schedule
import pytest

from scheduler.tasks.training_task import TrainingTask
import scheduler.tasks.training_task as training_task


def test_run_success(monkeypatch):
    """
    run() should return True when train_main executes without error.
    """
    monkeypatch.setattr(training_task, 'train_main', lambda: None)
    result = TrainingTask().run()
    assert result is True


def test_run_failure(monkeypatch, caplog):
    """
    run() should return False and log an error when train_main raises an exception.
    """
    def _fail():
        raise RuntimeError("failure")

    monkeypatch.setattr(training_task, 'train_main', _fail)
    caplog.set_level(logging.ERROR)
    result = TrainingTask().run()
    assert result is False
    assert "TrainingTask failed with error: failure" in caplog.text


def test_schedule_cron_fallback(caplog):
    """
    schedule() should warn about unsupported cron and schedule daily at 01:00.
    """
    schedule.clear()
    caplog.set_level(logging.WARNING)
    cfg = {'cron': '0 0 * * *'}

    TrainingTask.schedule(cfg)

    # Warning logged
    assert "Cron scheduling not supported: 0 0 * * *. Defaulting to daily at 01:00" in caplog.text

    # One job scheduled daily at 01:00
    jobs = schedule.jobs
    assert len(jobs) == 1
    job = jobs[0]
    assert job.unit == 'days'
    assert getattr(job, 'at_time', None) == '01:00'


def test_schedule_daily_with_time():
    """
    schedule() should schedule daily at the configured time.
    """
    schedule.clear()
    time_str = '05:30'
    cfg = {'frequency': 'daily', 'time': time_str}

    TrainingTask.schedule(cfg)

    jobs = schedule.jobs
    assert len(jobs) == 1
    job = jobs[0]
    assert job.unit == 'days'
    assert getattr(job, 'at_time', None) == time_str


def test_schedule_unsupported_frequency(caplog):
    """
    schedule() should warn on unsupported frequency and default to daily at 01:00.
    """
    schedule.clear()
    caplog.set_level(logging.WARNING)
    cfg = {'frequency': 'monthly'}

    TrainingTask.schedule(cfg)

    assert "Unsupported frequency 'monthly'. Defaulting to daily at 01:00" in caplog.text

    jobs = schedule.jobs
    assert len(jobs) == 1
    job = jobs[0]
    assert job.unit == 'days'
    assert getattr(job, 'at_time', None) == '01:00'
