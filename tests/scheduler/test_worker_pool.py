import pytest
import time
import threading
import logging
from scheduler.worker_pool import WorkerPool

def test_submit_task_returns_value():
    """Test that submit_task returns the expected value from the task."""
    pool = WorkerPool({'max_workers': 2})
    future = pool.submit_task(lambda x, y: x + y, 2, 3)
    assert future.result(timeout=1) == 5
    pool.shutdown()

def test_submit_task_raises_exception():
    """Test that submit_task properly propagates exceptions from the task."""
    pool = WorkerPool({'max_workers': 1})

    def bad_task():
        raise ValueError("failure")

    future = pool.submit_task(bad_task)
    with pytest.raises(ValueError) as excinfo:
        future.result(timeout=1)
    assert "failure" in str(excinfo.value)
    pool.shutdown()

def test_submit_task_logs_execution_events(caplog):
    """Test that submit_task logs task execution events."""
    caplog.set_level(logging.INFO)
    pool = WorkerPool({'max_workers': 1})
    
    def test_task():
        return "Task completed"
    
    future = pool.submit_task(test_task)
    result = future.result(timeout=1)
    
    assert result == "Task completed"
    assert "Submitting task: test_task" in caplog.text
    assert "Starting task: test_task" in caplog.text
    assert "Task completed successfully: test_task" in caplog.text
    
    pool.shutdown()

def test_submit_task_logs_failure(caplog):
    """Test that submit_task logs task failures."""
    caplog.set_level(logging.ERROR)
    pool = WorkerPool({'max_workers': 1})
    
    def failing_task():
        raise ValueError("Task failed")
    
    future = pool.submit_task(failing_task)
    
    with pytest.raises(ValueError) as excinfo:
        future.result(timeout=1)
    
    assert "Task failed" in str(excinfo.value)
    assert "Task failed: failing_task - Task failed" in caplog.text
    
    pool.shutdown()

def test_shutdown_wait_true_blocks_until_tasks_complete():
    """Test that shutdown with wait=True blocks until all tasks complete."""
    event = threading.Event()

    def long_task():
        event.wait()
        return "completed"

    pool = WorkerPool({'max_workers': 1})
    future = pool.submit_task(long_task)

    shutdown_event = threading.Event()

    def do_shutdown():
        pool.shutdown(wait=True)
        shutdown_event.set()

    shutdown_thread = threading.Thread(target=do_shutdown)
    shutdown_thread.start()

    # Ensure shutdown is blocking while task is not complete
    time.sleep(0.1)
    assert not shutdown_event.is_set(), (
        "Shutdown returned before task completion with wait=True"
    )

    # Complete the task and allow shutdown to finish
    event.set()
    assert shutdown_event.wait(timeout=1), (
        "Shutdown did not complete after task finished with wait=True"
    )
    assert future.result(timeout=1) == "completed"

def test_shutdown_wait_false_returns_immediately():
    """Test that shutdown with wait=False returns immediately without waiting for tasks."""
    event = threading.Event()

    def long_task():
        event.wait()
        return "completed"

    pool = WorkerPool({'max_workers': 1})
    future = pool.submit_task(long_task)

    shutdown_event = threading.Event()

    def do_shutdown():
        pool.shutdown(wait=False)
        shutdown_event.set()

    shutdown_thread = threading.Thread(target=do_shutdown)
    shutdown_thread.start()

    # Shutdown should return immediately
    assert shutdown_event.wait(timeout=1), (
        "Shutdown did not return immediately with wait=False"
    )

    # Task should still be pending
    assert not future.done(), (
        "Task completed too early before event set with wait=False"
    )

    # Now complete the task
    event.set()
    assert future.result(timeout=1) == "completed"

def test_worker_pool_handles_multiple_concurrent_tasks():
    """Test that worker pool can handle multiple concurrent tasks."""
    pool = WorkerPool({'max_workers': 4})
    
    results = []
    def task(value):
        time.sleep(0.1)  # Simulate some work
        return value * 2
    
    # Submit multiple tasks
    futures = [pool.submit_task(task, i) for i in range(5)]
    
    # Collect results
    for future in futures:
        results.append(future.result(timeout=1))
    
    # Check results
    assert sorted(results) == [0, 2, 4, 6, 8]
    pool.shutdown()

def test_worker_pool_respects_max_workers():
    """Test that worker pool respects the max_workers configuration."""
    # Create a pool with just 1 worker
    pool = WorkerPool({'max_workers': 1})
    
    # Use an event to control task execution
    start_event = threading.Event()
    finish_event = threading.Event()
    
    def controlled_task():
        start_event.set()  # Signal that task has started
        finish_event.wait()  # Wait for signal to finish
        return "done"
    
    # Submit first task (should start immediately)
    future1 = pool.submit_task(controlled_task)
    
    # Wait for first task to start
    assert start_event.wait(timeout=1), "First task should have started"
    
    # Reset event for second task
    start_event.clear()
    
    # Submit second task (should be queued)
    future2 = pool.submit_task(controlled_task)
    
    # Second task should not start yet (only 1 worker)
    time.sleep(0.2)
    assert not start_event.is_set(), "Second task should not have started yet"
    
    # Allow first task to complete
    finish_event.set()
    assert future1.result(timeout=1) == "done"
    
    # Now second task should start
    assert start_event.wait(timeout=1), "Second task should have started after first completed"
    
    # Allow second task to complete
    assert future2.result(timeout=1) == "done"
    
    pool.shutdown()
