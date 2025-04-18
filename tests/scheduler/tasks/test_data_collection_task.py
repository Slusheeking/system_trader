import logging
import pytest
import schedule

from scheduler.tasks.data_collection_task import DataCollectionTask


def setup_function(function):
    """Clear scheduled jobs before each test."""
    schedule.clear()


class MockCollector:
    def collect(self, **kwargs):
        return {'summary': {'items': 10, 'status': 'success'}}


class MockCollectorFactory:
    @staticmethod
    def create(source):
        return MockCollector()


def test_schedule_hourly_registers_job(monkeypatch):
    """Test that hourly scheduling registers the correct job."""
    config = {
        'schedule': {
            'frequency': 'hourly',
            'time': ':05'
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    DataCollectionTask.schedule(schedule, config, worker_pool)
    
    # Check that a job was scheduled
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for hourly frequency"
    job = jobs[0]
    assert job.interval == 1, "Hourly job should have interval of 1"
    assert job.unit == 'hours', "Hourly job unit should be 'hours'"
    assert job.at_time == ':05', "Hourly job time should match the configured time"


def test_schedule_daily_registers_job():
    """Test that daily scheduling registers the correct job."""
    config = {
        'schedule': {
            'frequency': 'daily',
            'time': '04:30'
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    DataCollectionTask.schedule(schedule, config, worker_pool)
    
    # Check that a job was scheduled
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for daily frequency"
    job = jobs[0]
    assert job.interval == 1, "Daily job should have interval of 1"
    assert job.unit == 'days', "Daily job unit should be 'days'"
    assert job.at_time == '04:30', "Daily job time should match the configured time"


def test_schedule_minutes_registers_job():
    """Test that minutes scheduling registers the correct job."""
    config = {
        'schedule': {
            'frequency': 'minutes',
            'interval': 5
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    DataCollectionTask.schedule(schedule, config, worker_pool)
    
    # Check that a job was scheduled
    jobs = schedule.jobs
    assert len(jobs) == 1, "Expected one scheduled job for minutes frequency"
    job = jobs[0]
    assert job.interval == 5, "Minutes job should have the configured interval"
    assert job.unit == 'minutes', "Minutes job unit should be 'minutes'"


def test_run_collects_from_all_sources(monkeypatch, caplog):
    """Test that run collects data from all configured sources."""
    # Set up logging capture
    caplog.set_level(logging.INFO)
    
    # Mock the collector factory
    monkeypatch.setattr(
        'scheduler.tasks.data_collection_task.CollectorFactory',
        MockCollectorFactory
    )
    
    # Configure the task
    config = {
        'params': {
            'sources': ['alpaca', 'yahoo', 'polygon'],
            'asset_types': ['stocks', 'etfs']
        }
    }
    
    # Run the task
    task = DataCollectionTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] != 'error', "Task should not have failed"
    assert 'summary' in result, "Result should include a summary"
    assert result['summary']['total_sources'] == 3, "Should have processed 3 sources"
    assert result['summary']['successful_sources'] == 3, "All sources should have succeeded"
    
    # Check that each source was processed
    for source in config['params']['sources']:
        assert source in result['details'], f"Source {source} should be in the results"
        assert result['details'][source]['status'] == 'success', f"Source {source} should have succeeded"
    
    # Check logging
    assert "Starting data collection task" in caplog.text
    assert "Data collection completed" in caplog.text


def test_run_handles_source_failure(monkeypatch, caplog):
    """Test that run handles failures from individual sources."""
    # Set up logging capture
    caplog.set_level(logging.ERROR)
    
    # Create a factory that returns a failing collector for 'yahoo'
    class FailingCollectorFactory:
        @staticmethod
        def create(source):
            if source == 'yahoo':
                class FailingCollector:
                    def collect(self, **kwargs):
                        raise ValueError("Test failure")
                return FailingCollector()
            return MockCollector()
    
    # Mock the collector factory
    monkeypatch.setattr(
        'scheduler.tasks.data_collection_task.CollectorFactory',
        FailingCollectorFactory
    )
    
    # Configure the task
    config = {
        'params': {
            'sources': ['alpaca', 'yahoo', 'polygon'],
            'asset_types': ['stocks', 'etfs']
        }
    }
    
    # Run the task
    task = DataCollectionTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] != 'error', "Task should not have failed completely"
    assert 'summary' in result, "Result should include a summary"
    assert result['summary']['total_sources'] == 3, "Should have processed 3 sources"
    assert result['summary']['successful_sources'] == 2, "2 sources should have succeeded"
    assert result['summary']['failed_sources'] == 1, "1 source should have failed"
    
    # Check that each source was processed correctly
    assert result['details']['alpaca']['status'] == 'success', "alpaca should have succeeded"
    assert result['details']['yahoo']['status'] == 'error', "yahoo should have failed"
    assert result['details']['polygon']['status'] == 'success', "polygon should have succeeded"
    
    # Check error logging
    assert "Failed to collect data from yahoo: Test failure" in caplog.text


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
    task = DataCollectionTask()
    result = task.run(config)
    
    # Check the result
    assert result['status'] == 'error', "Task should have failed"
    assert 'error' in result, "Result should include an error message"
    
    # Check error logging
    assert "Data collection task failed" in caplog.text
    assert "Complete failure" in caplog.text
