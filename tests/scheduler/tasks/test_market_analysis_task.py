import logging
import pytest
import schedule

from scheduler.tasks.market_analysis_task import MarketAnalysisTask


def setup_function(function):
    """Clear scheduled jobs before each test."""
    schedule.clear()


class MockMarketRegimeModel:
    def detect_current_regime(self):
        return {
            'regime': 'bullish',
            'confidence': 85,
            'key_factors': ['momentum', 'volatility', 'breadth'],
            'probabilities': {'bullish': 0.85, 'neutral': 0.10, 'bearish': 0.05}
        }


def test_schedule_daily_registers_job():
    """Test that daily scheduling registers the correct job."""
    config = {
        'schedule': {
            'frequency': 'daily',
            'time': '05:30'
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    MarketAnalysisTask.schedule(schedule, config, worker_pool)
    
    # Check that a job was scheduled
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for daily frequency"
    job = jobs[0]
    assert job.interval == 1, "Daily job should have interval of 1"
    assert job.unit == 'days', "Daily job unit should be 'days'"
    assert job.at_time == '05:30', "Daily job time should match the configured time"


def test_schedule_weekly_registers_multiple_jobs():
    """Test that weekly scheduling registers jobs for each specified day."""
    config = {
        'schedule': {
            'frequency': 'weekly',
            'days': ['monday', 'wednesday', 'friday'],
            'time': '08:00'
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    MarketAnalysisTask.schedule(schedule, config, worker_pool)
    
    # Check that jobs were scheduled for each day
    jobs = schedule.jobs
    assert len(jobs) == 3, "Expected three scheduled jobs for weekly frequency with three days"
    
    # Check each job
    days = set()
    for job in jobs:
        assert job.unit == 'weeks', "Weekly job unit should be 'weeks'"
        assert job.at_time == '08:00', "Weekly job time should match the configured time"
        days.add(job.start_day)
    
    assert days == {'monday', 'wednesday', 'friday'}, "Jobs should be scheduled for the specified days"


def test_schedule_handles_invalid_weekday():
    """Test that scheduling handles invalid weekday gracefully."""
    config = {
        'schedule': {
            'frequency': 'weekly',
            'days': ['monday', 'invalid_day'],
            'time': '08:00'
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task (should not raise an exception)
    MarketAnalysisTask.schedule(schedule, config, worker_pool)
    
    # Check that only valid days were scheduled
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for weekly frequency with one valid day"
    assert jobs[0].start_day == 'monday', "Job should be scheduled for the valid day"


def test_run_performs_all_analyses(monkeypatch, caplog):
    """Test that run performs all configured analyses."""
    # Set up logging capture
    caplog.set_level(logging.INFO)
    
    # Mock the MarketRegimeModel
    monkeypatch.setattr(
        'models.market_regime.model.MarketRegimeModel',
        MockMarketRegimeModel
    )
    
    # Configure the task
    config = {
        'params': {
            'regime_analysis': True,
            'correlation_matrix': True,
            'volatility_surface': True
        }
    }
    
    # Run the task
    task = MarketAnalysisTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] != 'error', "Task should not have failed"
    assert 'summary' in result, "Result should include a summary"
    assert result['summary']['total_analyses'] == 3, "Should have performed 3 analyses"
    assert result['summary']['successful_analyses'] == 3, "All analyses should have succeeded"
    
    # Check that each analysis was performed
    assert 'regime_analysis' in result['details'], "Regime analysis should be in the results"
    assert 'correlation_matrix' in result['details'], "Correlation matrix should be in the results"
    assert 'volatility_surface' in result['details'], "Volatility surface should be in the results"
    
    # Check that each analysis succeeded
    assert result['details']['regime_analysis']['status'] == 'success', "Regime analysis should have succeeded"
    assert result['details']['correlation_matrix']['status'] == 'success', "Correlation matrix should have succeeded"
    assert result['details']['volatility_surface']['status'] == 'success', "Volatility surface should have succeeded"
    
    # Check logging
    assert "Starting market analysis task" in caplog.text
    assert "Performing market regime analysis" in caplog.text
    assert "Generating correlation matrix" in caplog.text
    assert "Analyzing volatility surface" in caplog.text
    assert "Market analysis completed" in caplog.text


def test_run_handles_analysis_failure(monkeypatch, caplog):
    """Test that run handles failures from individual analyses."""
    # Set up logging capture
    caplog.set_level(logging.ERROR)
    
    # Mock the MarketRegimeModel to raise an exception
    class FailingMarketRegimeModel:
        def detect_current_regime(self):
            raise ValueError("Test failure")
    
    monkeypatch.setattr(
        'models.market_regime.model.MarketRegimeModel',
        FailingMarketRegimeModel
    )
    
    # Configure the task
    config = {
        'params': {
            'regime_analysis': True,
            'correlation_matrix': True,
            'volatility_surface': True
        }
    }
    
    # Run the task
    task = MarketAnalysisTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] != 'error', "Task should not have failed completely"
    assert 'summary' in result, "Result should include a summary"
    assert result['summary']['total_analyses'] == 3, "Should have performed 3 analyses"
    assert result['summary']['successful_analyses'] == 2, "2 analyses should have succeeded"
    assert result['summary']['failed_analyses'] == 1, "1 analysis should have failed"
    
    # Check that each analysis was processed correctly
    assert result['details']['regime_analysis']['status'] == 'error', "Regime analysis should have failed"
    assert result['details']['correlation_matrix']['status'] == 'success', "Correlation matrix should have succeeded"
    assert result['details']['volatility_surface']['status'] == 'success', "Volatility surface should have succeeded"
    
    # Check error logging
    assert "Market regime analysis failed: Test failure" in caplog.text


def test_run_handles_complete_failure(monkeypatch, caplog):
    """Test that run handles a complete failure gracefully."""
    # Set up logging capture
    caplog.set_level(logging.ERROR)
    
    # Mock to force a complete failure
    def raise_error(*args, **kwargs):
        raise RuntimeError("Complete failure")
    
    # Apply the mock to the params access to trigger early failure
    monkeypatch.setattr(dict, 'get', raise_error)
    
    # Configure the task
    config = {}
    
    # Run the task
    task = MarketAnalysisTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] == 'error', "Task should have failed"
    assert 'error' in result, "Result should include an error message"
    
    # Check error logging
    assert "Market analysis task failed" in caplog.text
    assert "Complete failure" in caplog.text


def test_run_skips_disabled_analyses(monkeypatch, caplog):
    """Test that run skips analyses that are disabled in the configuration."""
    # Set up logging capture
    caplog.set_level(logging.INFO)
    
    # Mock the MarketRegimeModel
    monkeypatch.setattr(
        'models.market_regime.model.MarketRegimeModel',
        MockMarketRegimeModel
    )
    
    # Configure the task with only regime analysis enabled
    config = {
        'params': {
            'regime_analysis': True,
            'correlation_matrix': False,
            'volatility_surface': False
        }
    }
    
    # Run the task
    task = MarketAnalysisTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] != 'error', "Task should not have failed"
    assert 'summary' in result, "Result should include a summary"
    assert result['summary']['total_analyses'] == 1, "Should have performed 1 analysis"
    assert result['summary']['successful_analyses'] == 1, "The analysis should have succeeded"
    
    # Check that only regime analysis was performed
    assert 'regime_analysis' in result['details'], "Regime analysis should be in the results"
    assert 'correlation_matrix' not in result['details'], "Correlation matrix should not be in the results"
    assert 'volatility_surface' not in result['details'], "Volatility surface should not be in the results"
    
    # Check logging
    assert "Performing market regime analysis" in caplog.text
    assert "Generating correlation matrix" not in caplog.text
    assert "Analyzing volatility surface" not in caplog.text
