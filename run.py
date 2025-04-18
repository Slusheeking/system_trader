#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
System Trader Runner
-------------------
Command-line interface for running the trading system in various modes.
Provides access to all system components and operations.
"""

# Suppress TensorFlow/CUDA warnings - must be set before any TensorFlow imports
import os
import sys

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from allocating all GPU memory

# Create a custom stderr filter to suppress specific CUDA warnings
class CUDAWarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, message):
        # Filter out CUDA factory registration warnings and other TensorFlow warnings
        if ("Unable to register" in message and "factory for plugin" in message) or \
           "cuDNN" in message or \
           "cuBLAS" in message or \
           "Attempting to register factory" in message:
            # Skip these messages
            pass
        else:
            # Write all other messages
            self.original_stderr.write(message)
            
    def flush(self):
        self.original_stderr.flush()
        
    def fileno(self):
        return self.original_stderr.fileno()

# Apply the filter to stderr
sys.stderr = CUDAWarningFilter(sys.stderr)

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

# Import other modules
import argparse
import logging
import json
import time
import signal
import subprocess
import shutil
import socket
from datetime import datetime, timedelta
import threading
import yaml

# Add the project root to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Fix pandas_ta issue with numpy.NaN
try:
    import pandas_ta
    import os
    
    # Fix the squeeze_pro.py file
    squeeze_pro_path = os.path.join(os.path.dirname(pandas_ta.__file__), 'momentum', 'squeeze_pro.py')
    if os.path.exists(squeeze_pro_path):
        with open(squeeze_pro_path, 'r') as f:
            content = f.read()
        
        if 'from numpy import NaN as npNaN' in content:
            fixed_content = content.replace('from numpy import NaN as npNaN', 'from numpy import nan as npNaN')
            with open(squeeze_pro_path, 'w') as f:
                f.write(fixed_content)
            print(f"Fixed pandas_ta squeeze_pro.py")
except Exception as e:
    print(f"Warning: Could not fix pandas_ta: {str(e)}")

# Import system components
from utils.logging import setup_logger
from utils.config_loader import ConfigLoader
from data.database.timeseries_db import get_timescale_client
from data.database.redis_client import get_redis_client
from data.collectors.factory import CollectorFactory
from scheduler.task_scheduler import discover_tasks
from scheduler.worker_pool import WorkerPool
from orchestration.workflow_manager import get_workflow_manager
# Import backtesting engine
try:
    from backtesting.engine import BacktestingEngine
except ImportError as e:
    print(f"Warning: Could not import BacktestingEngine: {str(e)}")
    # Create a dummy BacktestingEngine class
    class BacktestingEngine:
        def __init__(self, config_path=None):
            pass
from monitoring.monitor_manager import get_monitor_manager
from main import SystemTrader
from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.finbert_integration import analyzer as rule_based_analyzer

# Setup logging
logger = setup_logger('system_runner')


class ServiceManager:
    """
    Manages external services required by the trading system.
    """
    
    def __init__(self, config_path: str = 'config/system_config.yaml'):
        """Initialize the service manager."""
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load(config_path)
        self.services = {}
        self.processes = {}
        
        # Load service configurations
        self._load_service_configs()
        
        logger.info("Service Manager initialized")
    
    def _load_service_configs(self):
        """Load service configurations from config file."""
        services_config = self.config.get('services', {})
        
        # PostgreSQL/TimescaleDB
        self.services['timescaledb'] = services_config.get('timescaledb', {
            'enabled': True,
            'host': 'localhost',
            'port': 5432,
            'user': 'timescaleuser',
            'password': 'password',
            'dbname': 'timescaledb_test',
            'service_file': '/etc/systemd/system/postgresql.service',
            'docker': {
                'enabled': False,
                'image': 'timescale/timescaledb:latest-pg14',
                'container_name': 'timescaledb',
                'ports': ['5432:5432'],
                'volumes': ['timescaledb_data:/var/lib/postgresql/data'],
                'environment': [
                    'POSTGRES_USER=timescaleuser',
                    'POSTGRES_PASSWORD=password',
                    'POSTGRES_DB=timescaledb_test'
                ]
            }
        })
        
        # Redis
        self.services['redis'] = services_config.get('redis', {
            'enabled': True,
            'host': 'localhost',
            'port': 6379,
            'service_file': '/etc/systemd/system/redis.service',
            'docker': {
                'enabled': False,
                'image': 'redis:latest',
                'container_name': 'redis',
                'ports': ['6379:6379'],
                'volumes': ['redis_data:/data']
            }
        })
        
        # MLflow
        self.services['mlflow'] = services_config.get('mlflow', {
            'enabled': True,
            'host': 'localhost',
            'port': 5000,
            'tracking_uri': 'http://localhost:5000',
            'artifact_root': './mlruns',
            'backend_store_uri': 'sqlite:///mlflow.db',
            'service_file': None,  # No systemd service, run as process
            'docker': {
                'enabled': False,
                'image': 'ghcr.io/mlflow/mlflow:latest',
                'container_name': 'mlflow',
                'ports': ['5000:5000'],
                'volumes': ['./mlruns:/mlruns', './mlflow.db:/mlflow.db'],
                'command': 'mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns'
            }
        })
        
        # Prometheus
        self.services['prometheus'] = services_config.get('prometheus', {
            'enabled': True,
            'host': 'localhost',
            'port': 9090,
            'config_path': 'config/prometheus/prometheus.yml',
            'service_file': 'config/prometheus/prometheus.service',
            'docker': {
                'enabled': False,
                'image': 'prom/prometheus:latest',
                'container_name': 'prometheus',
                'ports': ['9090:9090'],
                'volumes': [
                    './config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml',
                    './config/prometheus/rules:/etc/prometheus/rules',
                    'prometheus_data:/prometheus'
                ],
                'command': '--config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus'
            }
        })
        
        # Alertmanager - disabled, using Grafana for alerts
        self.services['alertmanager'] = services_config.get('alertmanager', {
            'enabled': False,  # Disabled
            'host': 'localhost',
            'port': 9093,
            'config_path': 'config/prometheus/alertmanager.yml',
            'service_file': None,  # No systemd service, run as process
            'docker': {
                'enabled': False,
                'image': 'prom/alertmanager:latest',
                'container_name': 'alertmanager',
                'ports': ['9093:9093'],
                'volumes': [
                    './config/prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml',
                    'alertmanager_data:/alertmanager'
                ]
            }
        })
        
        # Grafana
        self.services['grafana'] = services_config.get('grafana', {
            'enabled': True,
            'host': 'localhost',
            'port': 3000,
            'config_path': 'config/grafana/grafana.ini',
            'service_file': 'config/grafana/grafana.service',
            'docker': {
                'enabled': False,
                'image': 'grafana/grafana:latest',
                'container_name': 'grafana',
                'ports': ['3000:3000'],
                'volumes': [
                    './config/grafana:/etc/grafana',
                    './config/grafana/provisioning:/etc/grafana/provisioning',
                    './monitoring/grafana:/var/lib/grafana/dashboards',
                    'grafana_data:/var/lib/grafana'
                ],
                'environment': [
                    'GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource'
                ]
            }
        })
    
    def start_service(self, service_name: str) -> bool:
        """
        Start a specific service.
        
        Args:
            service_name: Name of the service to start
            
        Returns:
            Boolean indicating success
        """
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service_config = self.services[service_name]
        if not service_config.get('enabled', True):
            logger.info(f"Service {service_name} is disabled, skipping")
            return False
        
        logger.info(f"Starting service: {service_name}")
        
        # Check if service is already running
        if self._is_service_running(service_name):
            logger.info(f"Service {service_name} is already running")
            return True
        
        # Check if we should use Docker
        if service_config.get('docker', {}).get('enabled', False):
            return self._start_docker_service(service_name)
        
        # Check if we should use systemd
        service_file = service_config.get('service_file')
        if service_file and os.path.exists(service_file):
            return self._start_systemd_service(service_name)
        
        # Otherwise start as a process
        return self._start_process_service(service_name)
    
    def _is_service_running(self, service_name: str) -> bool:
        """
        Check if a service is running.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating if service is running
        """
        # Check if we have a process for this service
        if service_name in self.processes and self.processes[service_name].poll() is None:
            return True
        
        # Check if port is open
        service_config = self.services[service_name]
        host = service_config.get('host', 'localhost')
        port = service_config.get('port')
        
        if port:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    return s.connect_ex((host, port)) == 0
            except Exception:
                pass
        
        # For systemd services, check systemctl status
        service_file = service_config.get('service_file')
        if service_file and os.path.exists(service_file):
            try:
                service_name = os.path.basename(service_file)
                result = subprocess.run(['systemctl', 'is-active', service_name], 
                                      capture_output=True, text=True)
                return result.stdout.strip() == 'active'
            except Exception:
                pass
        
        # For Docker services, check if container is running
        if service_config.get('docker', {}).get('enabled', False):
            try:
                container_name = service_config['docker']['container_name']
                result = subprocess.run(['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                                      capture_output=True, text=True)
                return container_name in result.stdout
            except Exception:
                pass
        
        return False
    
    def _start_systemd_service(self, service_name: str) -> bool:
        """
        Start a service using systemd.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating success
        """
        service_file = self.services[service_name]['service_file']
        service_unit = os.path.basename(service_file)
        
        try:
            # Start service
            result = subprocess.run(['sudo', 'systemctl', 'start', service_unit], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error starting {service_name} service: {result.stderr}")
                return False
            
            logger.info(f"Started {service_name} service")
            return True
        except Exception as e:
            logger.error(f"Error starting {service_name} service: {str(e)}")
            return False
    
    def _start_docker_service(self, service_name: str) -> bool:
        """
        Start a service using Docker.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating success
        """
        docker_config = self.services[service_name]['docker']
        container_name = docker_config['container_name']
        
        try:
            # Check if container exists
            result = subprocess.run(['docker', 'ps', '-a', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                                  capture_output=True, text=True)
            
            if container_name in result.stdout:
                # Container exists, start it
                result = subprocess.run(['docker', 'start', container_name], 
                                      capture_output=True, text=True)
            else:
                # Container doesn't exist, create and start it
                cmd = ['docker', 'run', '-d', '--name', container_name]
                
                # Add ports
                for port_mapping in docker_config.get('ports', []):
                    cmd.extend(['-p', port_mapping])
                
                # Add volumes
                for volume_mapping in docker_config.get('volumes', []):
                    cmd.extend(['-v', volume_mapping])
                
                # Add environment variables
                for env_var in docker_config.get('environment', []):
                    cmd.extend(['-e', env_var])
                
                # Add image
                cmd.append(docker_config['image'])
                
                # Add command if specified
                if 'command' in docker_config:
                    cmd.extend(docker_config['command'].split())
                
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error starting {service_name} Docker container: {result.stderr}")
                return False
            
            logger.info(f"Started {service_name} Docker container")
            return True
        except Exception as e:
            logger.error(f"Error starting {service_name} Docker container: {str(e)}")
            return False
    
    def _start_process_service(self, service_name: str) -> bool:
        """
        Start a service as a subprocess.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating success
        """
        try:
            if service_name == 'mlflow':
                # Start MLflow server using Python directly instead of CLI
                mlflow_config = self.services['mlflow']
                host = mlflow_config.get('host', 'localhost')
                port = str(mlflow_config.get('port', 5000))
                backend_store_uri = mlflow_config.get('backend_store_uri', 'sqlite:///mlflow.db')
                artifact_root = mlflow_config.get('artifact_root', './mlruns')
                
                # Create a Python script to start MLflow
                script_content = f"""
import sys
import os
import subprocess

# Ensure we're using the system MLflow, not the local one
sys.path = [p for p in sys.path if not p.endswith('system_trader')]

try:
    # Try to import mlflow directly
    import mlflow.server
    from mlflow.server import app
    
    # Start the server
    app.run(host='{host}', port={port}, backend_store_uri='{backend_store_uri}', default_artifact_root='{artifact_root}')
except ImportError:
    # Fall back to subprocess if direct import fails
    subprocess.run([
        'python', '-m', 'mlflow', 'server',
        '--host', '{host}',
        '--port', '{port}',
        '--backend-store-uri', '{backend_store_uri}',
        '--default-artifact-root', '{artifact_root}'
    ])
"""
                
                # Write the script to a temporary file
                with open('start_mlflow_server.py', 'w') as f:
                    f.write(script_content)
                
                # Make it executable
                os.chmod('start_mlflow_server.py', 0o755)
                
                # Run the script
                cmd = ['python', 'start_mlflow_server.py']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.processes['mlflow'] = process
                
                logger.info(f"Started MLflow server (PID: {process.pid})")
                return True
                
            elif service_name == 'alertmanager':
                # Start Alertmanager
                alertmanager_config = self.services['alertmanager']
                config_path = alertmanager_config.get('config_path', 'config/prometheus/alertmanager.yml')
                
                cmd = [
                    'alertmanager',
                    '--config.file', config_path,
                    '--web.listen-address', f"{alertmanager_config.get('host', 'localhost')}:{alertmanager_config.get('port', 9093)}"
                ]
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.processes['alertmanager'] = process
                
                logger.info(f"Started Alertmanager (PID: {process.pid})")
                return True
                
            else:
                logger.error(f"No process configuration for service: {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting {service_name} process: {str(e)}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """
        Stop a specific service.
        
        Args:
            service_name: Name of the service to stop
            
        Returns:
            Boolean indicating success
        """
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service_config = self.services[service_name]
        
        logger.info(f"Stopping service: {service_name}")
        
        # Check if we have a process for this service
        if service_name in self.processes:
            try:
                process = self.processes[service_name]
                process.terminate()
                process.wait(timeout=10)
                del self.processes[service_name]
                logger.info(f"Stopped {service_name} process")
                return True
            except Exception as e:
                logger.error(f"Error stopping {service_name} process: {str(e)}")
                return False
        
        # Check if we should use Docker
        if service_config.get('docker', {}).get('enabled', False):
            return self._stop_docker_service(service_name)
        
        # Check if we should use systemd
        service_file = service_config.get('service_file')
        if service_file and os.path.exists(service_file):
            return self._stop_systemd_service(service_name)
        
        logger.warning(f"No running instance found for {service_name}")
        return False
    
    def _stop_systemd_service(self, service_name: str) -> bool:
        """
        Stop a service using systemd.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating success
        """
        service_file = self.services[service_name]['service_file']
        service_unit = os.path.basename(service_file)
        
        try:
            # Stop service
            result = subprocess.run(['sudo', 'systemctl', 'stop', service_unit], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error stopping {service_name} service: {result.stderr}")
                return False
            
            logger.info(f"Stopped {service_name} service")
            return True
        except Exception as e:
            logger.error(f"Error stopping {service_name} service: {str(e)}")
            return False
    
    def _stop_docker_service(self, service_name: str) -> bool:
        """
        Stop a service using Docker.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating success
        """
        docker_config = self.services[service_name]['docker']
        container_name = docker_config['container_name']
        
        try:
            # Stop container
            result = subprocess.run(['docker', 'stop', container_name], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error stopping {service_name} Docker container: {result.stderr}")
                return False
            
            logger.info(f"Stopped {service_name} Docker container")
            return True
        except Exception as e:
            logger.error(f"Error stopping {service_name} Docker container: {str(e)}")
            return False
    
    def start_all_services(self) -> dict:
        """
        Start all enabled services.
        
        Returns:
            Dictionary of service names and success status
        """
        results = {}
        
        # Start services in the correct order
        service_order = [
            'timescaledb',  # Database first
            'redis',        # Cache next
            'mlflow',       # ML tracking
            'prometheus',   # Monitoring
            'grafana'       # Visualization
        ]
        
        for service_name in service_order:
            if service_name in self.services and self.services[service_name].get('enabled', True):
                results[service_name] = self.start_service(service_name)
                
                # If database failed to start, don't continue
                if service_name == 'timescaledb' and not results[service_name]:
                    logger.error("Database failed to start, aborting service startup")
                    break
                
                # Give services time to start
                time.sleep(2)
        
        return results
    
    def stop_all_services(self) -> dict:
        """
        Stop all running services.
        
        Returns:
            Dictionary of service names and success status
        """
        results = {}
        
        # Stop services in reverse order
        service_order = [
            'grafana',      # Visualization first
            'prometheus',   # Monitoring
            'mlflow',       # ML tracking
            'redis',        # Cache
            'timescaledb'   # Database last
        ]
        
        for service_name in service_order:
            if service_name in self.services and self._is_service_running(service_name):
                results[service_name] = self.stop_service(service_name)
                
                # Give services time to stop
                time.sleep(1)
        
        return results
    
    def get_service_status(self) -> dict:
        """
        Get status of all services.
        
        Returns:
            Dictionary of service names and status
        """
        status = {}
        
        for service_name in self.services:
            if not self.services[service_name].get('enabled', True):
                status[service_name] = 'disabled'
            elif self._is_service_running(service_name):
                status[service_name] = 'running'
            else:
                status[service_name] = 'stopped'
        
        return status
    
    def cleanup(self):
        """Clean up resources."""
        # Stop any running processes
        for service_name, process in list(self.processes.items()):
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                    logger.info(f"Terminated {service_name} process")
            except Exception as e:
                logger.error(f"Error terminating {service_name} process: {str(e)}")


class SystemRunner:
    """
    Command-line interface for running the trading system in various modes.
    """
    
    def __init__(self):
        """Initialize the system runner."""
        self.config_loader = ConfigLoader()
        self.components = {}
        self.service_manager = ServiceManager()
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("System Runner initialized")
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down...")
        self._shutdown()
        sys.exit(0)
    
    def _shutdown(self):
        """Shutdown all running components."""
        # Shutdown components
        for name, component in self.components.items():
            if hasattr(component, 'shutdown') and callable(component.shutdown):
                try:
                    logger.info(f"Shutting down {name}...")
                    component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {name}: {str(e)}")
        
        # Cleanup service manager
        if hasattr(self, 'service_manager'):
            self.service_manager.cleanup()
    
    def _load_config(self, config_path, overrides=None):
        """
        Load configuration with optional overrides.

        Args:
            config_path: Path to configuration file
            overrides: Optional dictionary of configuration overrides

        Returns:
            Loaded configuration
        """
        # Check if CONFIG_PATH environment variable is set
        env_config_path = os.environ.get('CONFIG_PATH')
        if env_config_path:
            logger.info(f"Using configuration from CONFIG_PATH environment variable: {env_config_path}")
            config_path = env_config_path
        
        config = self.config_loader.load(config_path)
        
        # Apply overrides if provided
        if overrides:
            for key_path, value in overrides.items():
                keys = key_path.split('.')
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
        
        return config
    
    def run_full_system(self, config_path, test_mode=False, start_services=True):
        """
        Run the full trading system.
        
        Args:
            config_path: Path to system configuration
            test_mode: Whether to run in test mode
            start_services: Whether to start required services
        """
        logger.info(f"Starting full system with config: {config_path}")
        
        try:
            # Start required services if requested
            if start_services:
                logger.info("Starting required services...")
                service_results = self.service_manager.start_all_services()
                
                # Check if critical services started
                if 'timescaledb' in service_results and not service_results['timescaledb']:
                    logger.error("Failed to start TimescaleDB, aborting system startup")
                    return
                
                # Log service status
                status = self.service_manager.get_service_status()
                for service, state in status.items():
                    logger.info(f"Service {service}: {state}")
            
            # Create and start trader
            trader = SystemTrader(config_path)
            self.components['trader'] = trader
            
            if test_mode:
                logger.info("Running in test mode")
                # Set test mode for components
                if 'circuit_breaker' in trader.components:
                    trader.components['circuit_breaker'].set_test_mode(True)
            
            # Start the trader
            trader.start()
            
        except Exception as e:
            logger.error(f"Error running full system: {str(e)}")
            self._shutdown()
            sys.exit(1)
    
    def run_scheduler(self, config_path, start_services=True):
        """
        Run the task scheduler.
        
        Args:
            config_path: Path to scheduler configuration
            start_services: Whether to start required services
        """
        logger.info(f"Starting task scheduler with config: {config_path}")
        
        try:
            # Start required services if requested
            if start_services:
                logger.info("Starting required services...")
                required_services = ['timescaledb', 'redis']
                for service in required_services:
                    if not self.service_manager.start_service(service):
                        logger.warning(f"Failed to start {service}, scheduler may not function correctly")
            
            # Load scheduler configuration
            config = self._load_config(config_path)
            
            # Initialize worker pool
            worker_count = config.get('worker_pool', {}).get('max_workers', 4)
            logger.info(f"Initializing worker pool with {worker_count} workers")
            worker_pool = WorkerPool({'max_workers': worker_count})
            self.components['worker_pool'] = worker_pool
            
            # Discover and register tasks
            tasks_dir = os.path.join(os.path.dirname(__file__), 'scheduler', 'tasks')
            tasks = discover_tasks(tasks_dir)
            
            if not tasks:
                logger.warning("No tasks discovered")
            
            # Register tasks with the scheduler
            import schedule
            for task_name, task_module in tasks.items():
                if task_name in config.get('tasks', {}):
                    task_config = config['tasks'][task_name]
                    logger.info(f"Registering task: {task_name} with config: {task_config}")
                    task_module.schedule(schedule, task_config, worker_pool)
                else:
                    logger.warning(f"No configuration found for task: {task_name}")
            
            # Start the scheduler loop
            interval = config.get('scheduler', {}).get('check_interval', 60)
            logger.info(f"Starting scheduler loop with interval: {interval} seconds")
            
            while True:
                schedule.run_pending()
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"Error running scheduler: {str(e)}")
            self._shutdown()
            sys.exit(1)
    
    def run_data_collection(self, collector_names=None, symbols=None, days=1, config_path=None, start_services=True):
        """
        Run data collection for specified collectors and symbols.
        
        Args:
            collector_names: List of collector names to run (None for all)
            symbols: List of symbols to collect (None for all configured)
            days: Number of days of historical data to collect
            config_path: Path to collector configuration
            start_services: Whether to start required services
        """
        if config_path is None:
            config_path = 'config/collector_config.yaml'
        
        logger.info(f"Starting data collection with config: {config_path}")
        
        try:
            # Start required services if requested
            if start_services:
                logger.info("Starting required services...")
                if not self.service_manager.start_service('timescaledb'):
                    logger.error("Failed to start TimescaleDB, aborting data collection")
                    return
            
            # Load collector configuration
            collector_config = self._load_config(config_path)
            
            # Determine which collectors to run
            if collector_names is None:
                # Use all enabled collectors
                collectors_to_run = [name for name, config in collector_config.items() 
                                   if config.get('enabled', True)]
            else:
                # Use specified collectors if they exist and are enabled
                collectors_to_run = []
                for name in collector_names:
                    if name in collector_config:
                        if collector_config[name].get('enabled', True):
                            collectors_to_run.append(name)
                        else:
                            logger.warning(f"Collector {name} is disabled, skipping")
                    else:
                        logger.warning(f"Unknown collector: {name}")
            
            if not collectors_to_run:
                logger.error("No collectors to run")
                return
            
            # Initialize database
            db_client = get_timescale_client()
            self.components['db_client'] = db_client
            
            # Run each collector
            for collector_name in collectors_to_run:
                logger.info(f"Running collector: {collector_name}")
                
                try:
                    # Create collector
                    collector = CollectorFactory.create(collector_name, collector_config[collector_name])
                    
                    # Set symbols if provided
                    if symbols:
                        collector.set_symbols(symbols)
                    
                    # Set date range if days > 1
                    if days > 1:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days)
                        collector.set_date_range(start_date, end_date)
                    
                    # Run collector
                    collector.run()
                    
                    logger.info(f"Collector {collector_name} completed")
                    
                except Exception as e:
                    logger.error(f"Error running collector {collector_name}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
    
    # This system is only focused on trading functionality, not ML training
    
    def run_backtesting(self, strategy_name, symbols, start_date, end_date, config_path=None, start_services=True):
        """
        Run backtesting for a strategy.
        
        Args:
            strategy_name: Name of the strategy to backtest
            symbols: List of symbols to backtest
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            config_path: Path to backtesting configuration
            start_services: Whether to start required services
        """
        if config_path is None:
            config_path = 'config/backtesting_config.yaml'
        
        logger.info(f"Starting backtesting for strategy: {strategy_name}")
        
        try:
            # Start required services if requested
            if start_services:
                logger.info("Starting required services...")
                if not self.service_manager.start_service('timescaledb'):
                    logger.error("Failed to start TimescaleDB, aborting backtesting")
                    return
            
            # Initialize backtest engine
            engine = BacktestingEngine(config_path)
            self.components['backtest_engine'] = engine
            
            # Run backtest
            results = engine.run_backtest(
                strategy_name=strategy_name,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Print summary
            print("\nBacktesting Results:")
            print(f"  Strategy: {strategy_name}")
            print(f"  Symbols: {', '.join(symbols)}")
            print(f"  Period: {start_date} to {end_date}")
            print(f"  Total Return: {results['total_return']:.2%}")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"  Win Rate: {results['win_rate']:.2%}")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = 'results/backtests'
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, f"{strategy_name}_{timestamp}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_path}")
            
            logger.info("Backtesting completed")
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
    
    def run_workflow(self, model_names=None, symbols=None, config_path=None, start_services=True):
        """
        Run the model workflow.
        
        Args:
            model_names: List of models to execute (None for all)
            symbols: List of symbols to analyze
            config_path: Path to workflow configuration
            start_services: Whether to start required services
        """
        if config_path is None:
            config_path = 'config/workflow.json'
        
        logger.info(f"Starting workflow execution with config: {config_path}")
        
        try:
            # Start required services if requested
            if start_services:
                logger.info("Starting required services...")
                required_services = ['timescaledb', 'redis']
                for service in required_services:
                    if not self.service_manager.start_service(service):
                        logger.warning(f"Failed to start {service}, workflow may not function correctly")
            
            # Initialize workflow manager
            workflow_manager = get_workflow_manager(config_path)
            self.components['workflow_manager'] = workflow_manager
            
            # Fetch data for symbols
            if symbols:
                # Determine date range (e.g., last 45 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=45)
                
                # Format dates
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Fetch data
                logger.info(f"Fetching data for {len(symbols)} symbols from {start_str} to {end_str}")
                
                # This is a placeholder - you would need to implement data fetching
                # based on your system's data access patterns
                db_client = get_timescale_client()
                data = db_client.query_market_data(
                    symbols=symbols,
                    start_time=start_date,
                    end_time=end_date
                )
                
                # Execute workflow
                results = workflow_manager.execute_workflow(data, model_names)
                
                # Print summary
                print("\nWorkflow Results:")
                for model_name, result in results.items():
                    if model_name == 'combined':
                        continue
                    
                    status = 'success' if 'error' not in result else 'error'
                    message = result.get('error', '')
                    
                    print(f"  {model_name}: {status}")
                    if message:
                        print(f"    Error: {message}")
                
                # Print combined results
                if 'combined' in results:
                    combined = results['combined']
                    print("\nCombined Results:")
                    
                    if 'signals' in combined:
                        signals = combined['signals']
                        print(f"  Total Signals: {len(signals)}")
                        
                        for signal in signals[:5]:  # Show first 5 signals
                            print(f"    {signal['symbol']}: {signal['signal_type']} at {signal['timestamp']}")
                        
                        if len(signals) > 5:
                            print(f"    ... and {len(signals) - 5} more")
                
                logger.info("Workflow execution completed")
            
            else:
                logger.error("No symbols provided for workflow execution")
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
    
    def run_sentiment_analysis(self, text=None, file_path=None, output_path=None, use_rule_based=True, config_path=None, start_services=True):
        """
        Run sentiment analysis on text or file.
        
        Args:
            text: Text to analyze
            file_path: Path to file with texts to analyze
            output_path: Path to save results
            use_rule_based: Whether to use rule-based implementation
            config_path: Path to sentiment analysis configuration
            start_services: Whether to start required services
        """
        if config_path is None:
            config_path = 'config/collector_config.yaml'
        
        logger.info(f"Starting sentiment analysis with config: {config_path}")
        
        try:
            # Load configuration
            config = self._load_config(config_path)
            sentiment_config = config.get('sentiment_analysis', {})
            
            # Initialize sentiment analyzer
            logger.info("Initializing sentiment analyzer...")
            analyzer = SentimentAnalyzer(
                use_finbert=sentiment_config.get('use_finbert', False),
                use_rule_based=use_rule_based or sentiment_config.get('use_rule_based', True)
            )
            
            # Analyze text or file
            if text:
                logger.info(f"Analyzing text: {text[:50]}...")
                result = analyzer.analyze(text)
                
                # Print result
                print("\nSentiment Analysis Result:")
                print(f"Text: {text}")
                print(f"Sentiment: {result['classification']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Magnitude: {result['magnitude']:.4f}")
                
                # Save result if output path provided
                if output_path:
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"Result saved to {output_path}")
                
            elif file_path:
                logger.info(f"Analyzing file: {file_path}")
                
                # Read file
                with open(file_path, 'r') as f:
                    texts = [line.strip() for line in f if line.strip()]
                
                # Analyze each text
                results = []
                for text in texts:
                    result = analyzer.analyze(text)
                    results.append({
                        'text': text,
                        'sentiment': result['classification'],
                        'score': result['score'],
                        'magnitude': result['magnitude']
                    })
                
                # Print summary
                print("\nSentiment Analysis Results:")
                print(f"Analyzed {len(results)} texts")
                
                # Count by sentiment
                sentiment_counts = {}
                for result in results:
                    sentiment = result['sentiment']
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                print("\nSentiment Distribution:")
                for sentiment, count in sentiment_counts.items():
                    print(f"  {sentiment}: {count} ({count/len(results):.1%})")
                
                # Print first 5 results
                print("\nSample Results:")
                for i, result in enumerate(results[:5]):
                    print(f"  {i+1}. {result['text'][:50]}...")
                    print(f"     Sentiment: {result['sentiment']}")
                    print(f"     Score: {result['score']:.4f}")
                
                # Save results if output path provided
                if output_path:
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Results saved to {output_path}")
            
            else:
                logger.error("No text or file provided for analysis")
                print("Error: Please provide either --text or --file for analysis")
            
            logger.info("Sentiment analysis completed")
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            print(f"Error: {str(e)}")
    
    def run_monitoring(self, config_path=None, start_services=True):
        """
        Run the monitoring system.
        
        Args:
            config_path: Path to monitoring configuration
            start_services: Whether to start required services
        """
        if config_path is None:
            config_path = 'config/system_config.yaml'
        
        logger.info(f"Starting monitoring system with config: {config_path}")
        
        try:
            # Start required services if requested
            if start_services:
                logger.info("Starting required services...")
                
                # Check if services are installed before trying to start them
                required_services = ['prometheus', 'grafana']
                for service in required_services:
                    # Check if the service executable exists
                    service_executable = service
                    if service == 'grafana':
                        service_executable = 'grafana-server'
                    
                    if shutil.which(service_executable):
                        if not self.service_manager.start_service(service):
                            logger.warning(f"Failed to start {service}, monitoring may not function correctly")
                    else:
                        logger.warning(f"{service} is not installed, skipping")
            
            # Initialize monitor manager
            monitor_manager = get_monitor_manager(config_path)
            self.components['monitor_manager'] = monitor_manager
            
            # Start monitoring
            monitor_manager.start()
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in monitoring system: {str(e)}")
            self._shutdown()
    
    def initialize_database(self, reset=False):
        """
        Initialize the database.
        
        Args:
            reset: Whether to reset existing tables
        """
        logger.info("Initializing database")
        
        try:
            # Start TimescaleDB if not running
            if not self.service_manager._is_service_running('timescaledb'):
                logger.info("Starting TimescaleDB...")
                if not self.service_manager.start_service('timescaledb'):
                    logger.error("Failed to start TimescaleDB, aborting database initialization")
                    return
            
            # Get database client
            db_client = get_timescale_client()
            
            # Create tables
            db_client.create_market_data_table(reset=reset)
            db_client.create_trade_data_table(reset=reset)
            db_client.create_analytics_table(reset=reset)
            
            # Create continuous aggregates
            db_client.create_continuous_aggregate('1 hour', 'market_data_hourly')
            db_client.create_continuous_aggregate('1 day', 'market_data_daily')
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def run_diagnostics(self):
        """Run system diagnostics."""
        logger.info("Running system diagnostics")
        
        try:
            # Check database connection
            logger.info("Checking database connection...")
            db_client = get_timescale_client()
            health = db_client.health_check()
            
            print("\nDatabase Health:")
            print(f"  Status: {health['status']}")
            print(f"  Message: {health['message']}")
            
            if 'details' in health:
                details = health['details']
                print("  Details:")
                for key, value in details.items():
                    print(f"    {key}: {value}")
            
            # Check Redis connection
            logger.info("Checking Redis connection...")
            redis_client = get_redis_client()
            redis_health = redis_client.health_check()
            
            print("\nRedis Health:")
            print(f"  Status: {redis_health['status']}")
            print(f"  Connected: {redis_health['connected']}")
            if 'memory_used' in redis_health:
                print(f"  Memory Used: {redis_health['memory_used']}")
            
            # Check model status
            logger.info("Checking model status...")
            try:
                workflow_manager = get_workflow_manager()
                model_status = workflow_manager.get_model_status()
                
                print("\nModel Status:")
                for model_name, status in model_status.items():
                    enabled = status.get('enabled', False)
                    initialized = status.get('initialized', False)
                    last_execution = status.get('last_execution', 'Never')
                    
                    print(f"  {model_name}:")
                    print(f"    Enabled: {enabled}")
                    print(f"    Initialized: {initialized}")
                    print(f"    Last Execution: {last_execution}")
            except Exception as e:
                logger.warning(f"Could not check model status: {str(e)}")
            
            # Check service status
            logger.info("Checking service status...")
            service_status = self.service_manager.get_service_status()
            
            print("\nService Status:")
            for service, status in service_status.items():
                print(f"  {service}: {status}")
            
            # Check system resources
            logger.info("Checking system resources...")
            try:
                import psutil
                
                print("\nSystem Resources:")
                print(f"  CPU Usage: {psutil.cpu_percent()}%")
                print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
                print(f"  Disk Usage: {psutil.disk_usage('/').percent}%")
            except ImportError:
                logger.warning("psutil not installed, skipping system resource check")
            
            logger.info("Diagnostics completed")
            
        except Exception as e:
            logger.error(f"Error in diagnostics: {str(e)}")


def run_docker_services():
    """
    Start all Docker services using docker-compose.
    This function is a wrapper around the start_docker_services.sh script.
    """
    try:
        # Check if Docker is installed
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Docker is not installed. Please install Docker first.")
            return False
        
        # Check if Docker Compose is installed
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Docker Compose is not installed. Please install Docker Compose first.")
            return False
        
        # Start Docker services using the script
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'start_docker_services.sh')
        if not os.path.exists(script_path):
            print(f"Docker services script not found at {script_path}")
            return False
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Run the script
        print("Starting Docker services...")
        process = subprocess.Popen([script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Return the process for monitoring
        return process
    
    except Exception as e:
        print(f"Error starting Docker services: {str(e)}")
        return False


def main():
    """Main entry point for the system runner."""
    parser = argparse.ArgumentParser(description='System Trader Runner')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Full system command
    system_parser = subparsers.add_parser('system', help='Run the full trading system')
    system_parser.add_argument('--config', type=str, default='config/system_config.yaml', 
                             help='Path to system configuration')
    system_parser.add_argument('--test', action='store_true', help='Run in test mode')
    system_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # Scheduler command
    scheduler_parser = subparsers.add_parser('scheduler', help='Run the task scheduler')
    scheduler_parser.add_argument('--config', type=str, default='config/scheduler.yaml',
                                help='Path to scheduler configuration')
    scheduler_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # Data collection command
    data_parser = subparsers.add_parser('collect', help='Run data collection')
    data_parser.add_argument('--collectors', type=str, nargs='+', 
                           help='Collectors to run (default: all enabled)')
    data_parser.add_argument('--symbols', type=str, nargs='+',
                           help='Symbols to collect (default: all configured)')
    data_parser.add_argument('--days', type=int, default=1,
                           help='Number of days of historical data to collect')
    data_parser.add_argument('--config', type=str, default='config/collector_config.yaml',
                           help='Path to collector configuration')
    data_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # No training functionality in this system, focusing only on trading
    
    # Backtesting command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--strategy', type=str, required=True,
                               help='Strategy to backtest')
    backtest_parser.add_argument('--symbols', type=str, nargs='+', required=True,
                               help='Symbols to backtest')
    backtest_parser.add_argument('--start', type=str, required=True,
                               help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', type=str, required=True,
                               help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--config', type=str, default='config/backtesting_config.yaml',
                               help='Path to backtesting configuration')
    backtest_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run the model workflow')
    workflow_parser.add_argument('--models', type=str, nargs='+',
                               help='Models to execute (default: all enabled)')
    workflow_parser.add_argument('--symbols', type=str, nargs='+', required=True,
                               help='Symbols to analyze')
    workflow_parser.add_argument('--config', type=str, default='config/workflow.json',
                               help='Path to workflow configuration')
    workflow_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # Sentiment analysis command
    sentiment_parser = subparsers.add_parser('sentiment', help='Run sentiment analysis')
    sentiment_parser.add_argument('--text', type=str, help='Text to analyze')
    sentiment_parser.add_argument('--file', type=str, help='File with texts to analyze')
    sentiment_parser.add_argument('--output', type=str, help='Path to save results')
    sentiment_parser.add_argument('--rule-based', action='store_true', help='Use rule-based implementation')
    sentiment_parser.add_argument('--config', type=str, default='config/collector_config.yaml',
                                help='Path to sentiment analysis configuration')
    sentiment_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # Monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Run the monitoring system')
    monitor_parser.add_argument('--config', type=str, default='config/system_config.yaml',
                              help='Path to monitoring configuration')
    monitor_parser.add_argument('--no-services', action='store_true', help='Do not start services')
    
    # Database command
    db_parser = subparsers.add_parser('db', help='Database operations')
    db_parser.add_argument('--init', action='store_true', help='Initialize database')
    db_parser.add_argument('--reset', action='store_true', help='Reset database')
    
    # Services command
    services_parser = subparsers.add_parser('services', help='Service management')
    services_parser.add_argument('--start', action='store_true', help='Start all services')
    services_parser.add_argument('--stop', action='store_true', help='Stop all services')
    services_parser.add_argument('--status', action='store_true', help='Show service status')
    services_parser.add_argument('--service', type=str, help='Specific service to manage')
    
    # Docker command
    docker_parser = subparsers.add_parser('docker', help='Docker services management')
    docker_parser.add_argument('--start', action='store_true', help='Start all Docker services')
    docker_parser.add_argument('--stop', action='store_true', help='Stop all Docker services')
    docker_parser.add_argument('--status', action='store_true', help='Show Docker services status')
    
    # Diagnostics command
    subparsers.add_parser('diagnostics', help='Run system diagnostics')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create system runner
    runner = SystemRunner()
    
    # Execute command
    if args.command == 'system':
        runner.run_full_system(args.config, args.test, not args.no_services)
    
    elif args.command == 'scheduler':
        runner.run_scheduler(args.config, not args.no_services)
    
    elif args.command == 'collect':
        runner.run_data_collection(args.collectors, args.symbols, args.days, args.config, not args.no_services)
    
    # No training functionality in this system
    
    elif args.command == 'backtest':
        runner.run_backtesting(args.strategy, args.symbols, args.start, args.end, args.config, not args.no_services)
    
    elif args.command == 'workflow':
        runner.run_workflow(args.models, args.symbols, args.config, not args.no_services)
    
    elif args.command == 'sentiment':
        runner.run_sentiment_analysis(args.text, args.file, args.output, args.rule_based, args.config, not args.no_services)
    
    elif args.command == 'monitor':
        runner.run_monitoring(args.config, not args.no_services)
    
    elif args.command == 'db':
        if args.init:
            runner.initialize_database(args.reset)
        else:
            print("No database operation specified. Use --init to initialize the database.")
    
    elif args.command == 'services':
        if args.service:
            # Manage specific service
            if args.start:
                success = runner.service_manager.start_service(args.service)
                print(f"Starting service {args.service}: {'Success' if success else 'Failed'}")
            elif args.stop:
                success = runner.service_manager.stop_service(args.service)
                print(f"Stopping service {args.service}: {'Success' if success else 'Failed'}")
            else:
                status = runner.service_manager._is_service_running(args.service)
                print(f"Service {args.service} is {'running' if status else 'stopped'}")
        else:
            # Manage all services
            if args.start:
                results = runner.service_manager.start_all_services()
                print("Service startup results:")
                for service, success in results.items():
                    print(f"  {service}: {'Success' if success else 'Failed'}")
            elif args.stop:
                results = runner.service_manager.stop_all_services()
                print("Service shutdown results:")
                for service, success in results.items():
                    print(f"  {service}: {'Success' if success else 'Failed'}")
            elif args.status or not (args.start or args.stop):
                status = runner.service_manager.get_service_status()
                print("Service status:")
                for service, state in status.items():
                    print(f"  {service}: {state}")
    
    elif args.command == 'diagnostics':
        runner.run_diagnostics()
    
    elif args.command == 'docker':
        if args.start:
            process = run_docker_services()
            if process:
                print("Docker services are starting. Use 'docker-compose ps' to check status.")
                print("Press Ctrl+C to exit (services will continue running in the background)")
                try:
                    # Stream output until user interrupts
                    for line in iter(process.stdout.readline, b''):
                        print(line.decode('utf-8').rstrip())
                except KeyboardInterrupt:
                    print("\nExiting... Docker services will continue running.")
            else:
                print("Failed to start Docker services")
        elif args.stop:
            try:
                print("Stopping Docker services...")
                result = subprocess.run(['docker-compose', 'down'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("Docker services stopped successfully")
                else:
                    print(f"Error stopping Docker services: {result.stderr}")
            except Exception as e:
                print(f"Error stopping Docker services: {str(e)}")
        elif args.status:
            try:
                print("Docker services status:")
                subprocess.run(['docker-compose', 'ps'])
            except Exception as e:
                print(f"Error getting Docker services status: {str(e)}")
        else:
            # Default to showing status
            try:
                print("Docker services status:")
                subprocess.run(['docker-compose', 'ps'])
            except Exception as e:
                print(f"Error getting Docker services status: {str(e)}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

