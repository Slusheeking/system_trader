import logging
import pytest
import schedule
import datetime
from unittest.mock import patch

from scheduler.tasks.notification_task import NotificationTask


def setup_function(function):
    """Clear scheduled jobs before each test."""
    schedule.clear()


def test_schedule_critical_alerts_monitoring():
    """Test that critical alerts monitoring is scheduled correctly."""
    config = {
        'schedule': {
            'critical_alerts': True
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    NotificationTask.schedule(schedule, config, worker_pool)
    
    # Check that a job was scheduled for critical alerts
    jobs = [job for job in schedule.jobs if job.unit == 'minutes']
    assert len(jobs) == 1, "Expected one scheduled job for critical alerts monitoring"
    job = jobs[0]
    assert job.interval == 1, "Critical alerts job should have interval of 1"
    assert job.unit == 'minutes', "Critical alerts job unit should be 'minutes'"


def test_schedule_pre_market_notifications():
    """Test that pre-market notifications are scheduled correctly."""
    config = {
        'schedule': {
            'status_notifications': {
                'pre_market': True
            }
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    NotificationTask.schedule(schedule, config, worker_pool)
    
    # Check that jobs were scheduled for pre-market notifications
    jobs = [job for job in schedule.jobs if job.unit == 'days']
    assert len(jobs) == 6, "Expected six scheduled jobs for pre-market notifications"
    
    # Check that each pre-market notification time is scheduled
    times = sorted([job.at_time for job in jobs])
    expected_times = ['04:10', '04:30', '05:15', '06:15', '07:15', '09:00']
    assert times == expected_times, "Pre-market notification times should match expected times"


def test_schedule_market_hours_notifications():
    """Test that market hours notifications are scheduled correctly."""
    config = {
        'schedule': {
            'status_notifications': {
                'market_hours': True
            }
        }
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    NotificationTask.schedule(schedule, config, worker_pool)
    
    # Check that jobs were scheduled for market hours notifications
    jobs = [job for job in schedule.jobs if job.unit == 'days']
    
    # Should have 9 jobs: market open, 6 hourly reports (10-15), mid-day, EOD prep, close summary
    assert len(jobs) == 10, "Expected ten scheduled jobs for market hours notifications"
    
    # Check that market open and close are scheduled
    times = sorted([job.at_time for job in jobs])
    assert '09:35' in times, "Market open notification should be scheduled"
    assert '16:05' in times, "Market close notification should be scheduled"
    
    # Check that hourly reports are scheduled
    for hour in range(10, 16):
        assert f"{hour}:00" in times, f"Hourly report at {hour}:00 should be scheduled"


def test_schedule_weekly_notifications():
    """Test that weekly notifications are scheduled correctly."""
    config = {
        'schedule': {}  # Weekly notifications are always scheduled
    }
    
    # Create a mock worker pool
    class MockWorkerPool:
        def submit_task(self, *args, **kwargs):
            pass
    
    worker_pool = MockWorkerPool()
    
    # Schedule the task
    NotificationTask.schedule(schedule, config, worker_pool)
    
    # Check that jobs were scheduled for weekly notifications
    monday_jobs = [job for job in schedule.jobs if hasattr(job, 'start_day') and job.start_day == 'monday']
    friday_jobs = [job for job in schedule.jobs if hasattr(job, 'start_day') and job.start_day == 'friday']
    
    assert len(monday_jobs) == 1, "Expected one scheduled job for Monday"
    assert monday_jobs[0].at_time == '08:00', "Monday job should be scheduled at 08:00"
    
    assert len(friday_jobs) == 1, "Expected one scheduled job for Friday"
    assert friday_jobs[0].at_time == '16:30', "Friday job should be scheduled at 16:30"


def test_check_critical_alerts():
    """Test that check_critical_alerts returns the expected structure."""
    task = NotificationTask()
    result = task.check_critical_alerts({})
    
    assert result['status'] == 'success', "Critical alerts check should succeed"
    assert 'alerts_triggered' in result, "Result should include alerts_triggered"
    assert 'timestamp' in result, "Result should include timestamp"


def test_send_notification_with_all_delivery_methods():
    """Test that _send_notification handles all delivery methods."""
    config = {
        'schedule': {
            'delivery_methods': {
                'email': True,
                'slack': True,
                'dashboard': True,
                'mobile': True
            }
        }
    }
    
    task = NotificationTask()
    notification_type = 'test_notification'
    data = {'test': 'data'}
    
    result = task._send_notification(notification_type, data, config)
    
    assert result['status'] == 'success', "Notification should succeed"
    assert result['notification_type'] == notification_type, "Notification type should match"
    assert result['data'] == data, "Notification data should match"
    
    # Check that all delivery methods were used
    delivery_results = result['delivery_results']
    assert 'email' in delivery_results, "Email delivery should be included"
    assert 'slack' in delivery_results, "Slack delivery should be included"
    assert 'dashboard' in delivery_results, "Dashboard delivery should be included"
    assert 'mobile' in delivery_results, "Mobile delivery should be included"


def test_send_notification_handles_error():
    """Test that _send_notification handles errors gracefully."""
    config = {
        'schedule': {
            'delivery_methods': {
                'email': True
            }
        }
    }
    
    task = NotificationTask()
    
    # Mock the delivery method to raise an exception
    def mock_delivery_that_fails(*args, **kwargs):
        raise ValueError("Test failure")
    
    # Patch the dict.get method to raise an exception
    with patch.object(dict, 'get', side_effect=mock_delivery_that_fails):
        result = task._send_notification('test', {}, config)
    
    assert result['status'] == 'error', "Notification should fail"
    assert 'error' in result, "Result should include error message"
    assert 'timestamp' in result, "Result should include timestamp"


def test_monthly_strategy_overview_skips_non_first_day(monkeypatch):
    """Test that monthly strategy overview is skipped on non-first days of month."""
    # Mock datetime.now to return a non-first day of month
    class MockDateTime:
        @classmethod
        def now(cls):
            return datetime.datetime(2025, 4, 15)
    
    monkeypatch.setattr(datetime, 'datetime', MockDateTime)
    
    task = NotificationTask()
    result = task.send_monthly_strategy_overview({})
    
    assert result['status'] == 'skipped', "Task should be skipped on non-first day"
    assert result['reason'] == 'Not first day of month', "Skip reason should be specified"


def test_monthly_strategy_overview_runs_on_first_day(monkeypatch):
    """Test that monthly strategy overview runs on the first day of month."""
    # Mock datetime.now to return the first day of month
    class MockDateTime:
        @classmethod
        def now(cls):
            return datetime.datetime(2025, 4, 1)
        
        @classmethod
        def isoformat(cls):
            return "2025-04-01T00:00:00"
    
    # Mock the datetime class
    monkeypatch.setattr(datetime, 'datetime', MockDateTime)
    
    # Mock the _send_notification method to avoid actual sending
    def mock_send(*args, **kwargs):
        return {'status': 'success', 'timestamp': '2025-04-01T00:00:00'}
    
    task = NotificationTask()
    task._send_notification = mock_send
    
    result = task.send_monthly_strategy_overview({})
    
    assert result['status'] == 'success', "Task should run on first day of month"


def test_monthly_comprehensive_report_skips_non_last_day(monkeypatch):
    """Test that monthly comprehensive report is skipped on non-last days of month."""
    # Mock datetime to return a non-last day of month
    class MockDateTime:
        @classmethod
        def now(cls):
            return datetime.datetime(2025, 4, 15)
        
        @classmethod
        def __add__(cls, other):
            # Return a date in the same month
            return datetime.datetime(2025, 4, 16)
    
    monkeypatch.setattr(datetime, 'datetime', MockDateTime)
    monkeypatch.setattr(datetime.datetime, 'now', MockDateTime.now)
    
    # Mock timedelta
    class MockTimedelta:
        def __init__(self, days=0):
            self.days = days
    
    monkeypatch.setattr(datetime, 'timedelta', MockTimedelta)
    
    task = NotificationTask()
    result = task.send_monthly_comprehensive_report({})
    
    assert result['status'] == 'skipped', "Task should be skipped on non-last day"
    assert result['reason'] == 'Not last day of month', "Skip reason should be specified"


def test_notification_methods_return_expected_structure():
    """Test that all notification methods return the expected structure."""
    task = NotificationTask()
    
    # Mock the _send_notification method to avoid actual sending
    def mock_send(*args, **kwargs):
        return {'status': 'success', 'timestamp': datetime.datetime.now().isoformat()}
    
    task._send_notification = mock_send
    
    # Test a sample of notification methods
    methods = [
        task.send_system_startup_notification,
        task.send_market_regime_assessment,
        task.send_hourly_status_report,
        task.send_market_close_summary,
        task.send_weekly_outlook
    ]
    
    for method in methods:
        result = method({})
        assert result['status'] == 'success', f"{method.__name__} should succeed"
