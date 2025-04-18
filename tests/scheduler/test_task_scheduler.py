import pytest
import schedule
import time
import os
from scheduler import task_scheduler


def test_main_registers_tasks_and_exits(monkeypatch, capsys):
    # Prepare a fake configuration
    fake_config = {
        'worker_pool': {'max_workers': 2},
        'tasks': {
            'dummy_task': {'interval': 5},
            'data_collection_task': {'schedule': {'frequency': 'hourly'}},
            'market_analysis_task': {'schedule': {'frequency': 'daily'}},
            'notification_task': {'schedule': {'critical_alerts': True}}
        },
        'scheduler': {'check_interval': 1}
    }

    # Monkeypatch ConfigLoader.load to return the fake configuration
    monkeypatch.setattr(
        task_scheduler.ConfigLoader,
        'load',
        lambda path: fake_config
    )

    # Record state for assertions
    called = {}

    # Define dummy task modules with schedule functions
    class DummyTaskModule:
        @staticmethod
        def schedule(sched, config, worker_pool):
            called['sched'] = sched
            called['config'] = config
            called['worker_pool'] = worker_pool
            
    class DataCollectionTaskModule:
        @staticmethod
        def schedule(sched, config, worker_pool):
            called['data_collection_scheduled'] = True
            
    class MarketAnalysisTaskModule:
        @staticmethod
        def schedule(sched, config, worker_pool):
            called['market_analysis_scheduled'] = True
            
    class NotificationTaskModule:
        @staticmethod
        def schedule(sched, config, worker_pool):
            called['notification_scheduled'] = True

    # Monkeypatch discover_tasks to return our dummy modules
    monkeypatch.setattr(
        task_scheduler,
        'discover_tasks',
        lambda tasks_dir: {
            'dummy_task': DummyTaskModule,
            'data_collection_task': DataCollectionTaskModule,
            'market_analysis_task': MarketAnalysisTaskModule,
            'notification_task': NotificationTaskModule
        }
    )

    # Monkeypatch WorkerPool to record initialization
    class FakeWorkerPool:
        def __init__(self, config):
            called['worker_count'] = config.get('max_workers', 4)

    monkeypatch.setattr(task_scheduler, 'WorkerPool', FakeWorkerPool)

    # Monkeypatch schedule.run_pending to raise an exception to break the loop
    def fake_run_pending():
        raise Exception("stop_loop")
    monkeypatch.setattr(schedule, 'run_pending', fake_run_pending)

    # Monkeypatch time.sleep to no-op to avoid delays
    monkeypatch.setattr(time, 'sleep', lambda _interval: None)

    # Run main and expect a SystemExit due to the raised exception
    with pytest.raises(SystemExit) as sys_exit:
        task_scheduler.main()

    # Verify exit code from SystemExit
    assert sys_exit.value.code == 1

    # Capture logging output (logs are sent to stderr)
    captured = capsys.readouterr()
    stderr = captured.err

    # Assert that key log messages were output
    assert "Starting task scheduler" in stderr
    assert "Loading configuration from" in stderr
    assert "Initializing worker pool with 2 workers" in stderr
    assert "Registering task: dummy_task" in stderr
    assert "Starting scheduler loop with interval: 1 seconds" in stderr
    assert "Shutting down task scheduler" in stderr

    # Assert that our dummy schedule functions were called with correct arguments
    assert called.get('sched') is schedule
    assert called.get('config') == {'interval': 5}
    assert isinstance(called.get('worker_pool'), FakeWorkerPool)
    assert called.get('worker_count') == 2
    
    # Assert that all task modules were scheduled
    assert called.get('data_collection_scheduled') is True
    assert called.get('market_analysis_scheduled') is True
    assert called.get('notification_scheduled') is True
