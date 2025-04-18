import os
import pytest
import yaml
from utils.config_loader import ConfigLoader


@pytest.fixture
def scheduler_config_path():
    """Return the path to the test scheduler configuration file."""
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'test_scheduler.yaml')


def test_scheduler_config_file_exists(scheduler_config_path):
    """Test that the scheduler configuration file exists."""
    assert os.path.exists(scheduler_config_path), "Scheduler config file should exist"


def test_scheduler_config_loads_correctly(scheduler_config_path):
    """Test that the scheduler configuration loads correctly."""
    config = ConfigLoader.load(scheduler_config_path)
    
    # Check top-level sections
    assert 'worker_pool' in config, "Config should have worker_pool section"
    assert 'scheduler' in config, "Config should have scheduler section"
    assert 'tasks' in config, "Config should have tasks section"
    
    # Check worker pool configuration
    assert config['worker_pool']['max_workers'] == 4, "Worker pool should have 4 max workers"
    
    # Check scheduler configuration
    assert config['scheduler']['check_interval'] == 10, "Scheduler check interval should be 10 seconds"
    
    # Check task configurations
    tasks = config['tasks']
    assert 'training_task' in tasks, "Config should include training_task"
    assert 'backtesting_task' in tasks, "Config should include backtesting_task"
    assert 'data_collection_task' in tasks, "Config should include data_collection_task"
    assert 'market_analysis_task' in tasks, "Config should include market_analysis_task"
    assert 'notification_task' in tasks, "Config should include notification_task"


def test_training_task_config(scheduler_config_path):
    """Test that the training task configuration is correct."""
    config = ConfigLoader.load(scheduler_config_path)
    training_config = config['tasks']['training_task']
    
    assert training_config['frequency'] == 'daily', "Training task should have daily frequency"
    assert training_config['time'] == '01:00', "Training task should run at 01:00"


def test_backtesting_task_config(scheduler_config_path):
    """Test that the backtesting task configuration is correct."""
    config = ConfigLoader.load(scheduler_config_path)
    backtesting_config = config['tasks']['backtesting_task']['backtesting']
    
    assert backtesting_config['schedule']['frequency'] == 'daily', "Backtesting task should have daily frequency"
    assert backtesting_config['schedule']['time'] == '02:00', "Backtesting task should run at 02:00"
    
    # Check parameters
    params = backtesting_config['params']
    assert params['lookback_days'] == 30, "Backtesting should use 30 lookback days"
    assert 'AAPL' in params['symbols'], "Backtesting should include AAPL in symbols"
    assert 'momentum' in params['strategies'], "Backtesting should include momentum strategy"


def test_data_collection_task_config(scheduler_config_path):
    """Test that the data collection task configuration is correct."""
    config = ConfigLoader.load(scheduler_config_path)
    data_collection_config = config['tasks']['data_collection_task']
    
    assert data_collection_config['schedule']['frequency'] == 'hourly', "Data collection should have hourly frequency"
    assert data_collection_config['schedule']['time'] == ':05', "Data collection should run at :05 minutes"
    
    # Check parameters
    params = data_collection_config['params']
    assert 'alpaca' in params['sources'], "Data collection should include alpaca source"
    assert 'stocks' in params['asset_types'], "Data collection should include stocks asset type"


def test_market_analysis_task_config(scheduler_config_path):
    """Test that the market analysis task configuration is correct."""
    config = ConfigLoader.load(scheduler_config_path)
    market_analysis_config = config['tasks']['market_analysis_task']
    
    assert market_analysis_config['schedule']['frequency'] == 'daily', "Market analysis should have daily frequency"
    assert market_analysis_config['schedule']['time'] == '05:30', "Market analysis should run at 05:30"
    
    # Check parameters
    params = market_analysis_config['params']
    assert params['regime_analysis'] is True, "Market analysis should include regime analysis"
    assert params['correlation_matrix'] is True, "Market analysis should include correlation matrix"
    assert params['volatility_surface'] is True, "Market analysis should include volatility surface"


def test_notification_task_config(scheduler_config_path):
    """Test that the notification task configuration is correct."""
    config = ConfigLoader.load(scheduler_config_path)
    notification_config = config['tasks']['notification_task']
    
    # Check schedule configuration
    schedule = notification_config['schedule']
    assert schedule['critical_alerts'] is True, "Notifications should include critical alerts"
    assert schedule['status_notifications']['pre_market'] is True, "Notifications should include pre-market status"
    assert schedule['status_notifications']['market_hours'] is True, "Notifications should include market hours status"
    assert schedule['status_notifications']['post_market'] is True, "Notifications should include post-market status"
    
    # Check delivery methods
    delivery_methods = schedule['delivery_methods']
    assert delivery_methods['email'] is True, "Notifications should include email delivery"
    assert delivery_methods['slack'] is True, "Notifications should include slack delivery"
    assert delivery_methods['dashboard'] is True, "Notifications should include dashboard delivery"


def test_config_yaml_syntax(scheduler_config_path):
    """Test that the YAML syntax in the config file is valid."""
    with open(scheduler_config_path, 'r') as f:
        try:
            yaml_content = yaml.safe_load(f)
            assert yaml_content is not None, "YAML content should not be None"
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax in config file: {e}")


def test_config_structure_matches_task_requirements(scheduler_config_path):
    """Test that the config structure matches the requirements of each task."""
    config = ConfigLoader.load(scheduler_config_path)
    
    # Training task expects frequency and time
    training_config = config['tasks']['training_task']
    assert 'frequency' in training_config, "Training task config should have frequency"
    assert 'time' in training_config, "Training task config should have time"
    
    # Backtesting task expects backtesting.schedule.frequency and backtesting.schedule.time
    backtesting_config = config['tasks']['backtesting_task']
    assert 'backtesting' in backtesting_config, "Backtesting task config should have backtesting section"
    assert 'schedule' in backtesting_config['backtesting'], "Backtesting config should have schedule section"
    assert 'frequency' in backtesting_config['backtesting']['schedule'], "Backtesting schedule should have frequency"
    assert 'time' in backtesting_config['backtesting']['schedule'], "Backtesting schedule should have time"
    
    # Data collection task expects schedule.frequency
    data_collection_config = config['tasks']['data_collection_task']
    assert 'schedule' in data_collection_config, "Data collection task config should have schedule section"
    assert 'frequency' in data_collection_config['schedule'], "Data collection schedule should have frequency"
    
    # Market analysis task expects schedule.frequency and schedule.time
    market_analysis_config = config['tasks']['market_analysis_task']
    assert 'schedule' in market_analysis_config, "Market analysis task config should have schedule section"
    assert 'frequency' in market_analysis_config['schedule'], "Market analysis schedule should have frequency"
    assert 'time' in market_analysis_config['schedule'], "Market analysis schedule should have time"
    
    # Notification task expects schedule with various notification settings
    notification_config = config['tasks']['notification_task']
    assert 'schedule' in notification_config, "Notification task config should have schedule section"
    assert 'critical_alerts' in notification_config['schedule'], "Notification schedule should have critical_alerts"
    assert 'status_notifications' in notification_config['schedule'], "Notification schedule should have status_notifications"
    assert 'delivery_methods' in notification_config['schedule'], "Notification schedule should have delivery_methods"
