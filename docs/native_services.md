# Native Services Setup

This document describes how to set up and manage the system_trader services as native systemd services instead of using Docker containers.

## Overview

The system_trader project has been migrated from Docker containers to native systemd services. This provides several advantages:

1. Reduced overhead - No Docker layer means better performance
2. Simplified management - Direct systemd integration for service management
3. Better integration with the host system - Services run directly on the host
4. Improved reliability - Native systemd services with automatic restart capabilities

## Services

The following services are now managed as native systemd services:

1. **MLflow** - Model tracking and registry service
2. **Prometheus** - Metrics collection and storage
3. **Grafana** - Visualization and dashboards
4. **Scheduler** - Task scheduling and execution
5. **Monitoring** - System monitoring and alerting

## Setup

To set up the native services, follow these steps:

1. Clean up Docker-related files (if migrating from Docker):

```bash
sudo ./scripts/cleanup_docker_files.sh
```

2. Set up the native services:

```bash
sudo ./scripts/setup_native_services.sh
```

This script will:
- Create systemd service files for each service
- Create necessary data directories
- Set appropriate permissions
- Install the services in systemd
- Enable the services to start on boot
- Start all services

## Managing Services

### Starting Services

To start all services:

```bash
sudo ./scripts/start_native_services.sh
```

To start a specific service:

```bash
sudo systemctl start system-trader-<service>.service
```

Where `<service>` is one of: `mlflow`, `prometheus`, `grafana`, `scheduler`, or `monitoring`.

### Stopping Services

To stop all services:

```bash
sudo ./scripts/stop_native_services.sh
```

To stop a specific service:

```bash
sudo systemctl stop system-trader-<service>.service
```

### Checking Service Status

To check the status of all services:

```bash
sudo systemctl status system-trader-*
```

To check the status of a specific service:

```bash
sudo systemctl status system-trader-<service>.service
```

### Viewing Service Logs

To view the logs for a service:

```bash
sudo journalctl -u system-trader-<service>.service
```

To follow the logs in real-time:

```bash
sudo journalctl -u system-trader-<service>.service -f
```

## Service URLs

The services are available at the following URLs:

- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- System Metrics: http://localhost:8001/metrics

## Troubleshooting

If a service fails to start, check the logs:

```bash
sudo journalctl -u system-trader-<service>.service -n 50
```

Common issues:

1. **Port conflicts**: Make sure no other services are using the same ports
2. **Permission issues**: Ensure the data directories have the correct permissions
3. **Missing dependencies**: Install any required dependencies
4. **Configuration errors**: Check the service configuration files

## Customization

The service configuration files are located in `/etc/systemd/system/` and can be customized as needed. After making changes, reload systemd:

```bash
sudo systemctl daemon-reload
```

Then restart the affected service:

```bash
sudo systemctl restart system-trader-<service>.service
```
