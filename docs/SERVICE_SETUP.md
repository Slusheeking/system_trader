# System Trader Service Setup

This document explains how to set up the System Trader to run 24/7 as a system service, ensuring all components (including monitoring) are always running.

## Components Included

The System Trader service includes the following components:

- **TimescaleDB**: Time-series database for storing market data
- **Redis**: In-memory cache for real-time data
- **MLflow**: Model tracking and registry
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboards, visualization, and alert management
- **Task Scheduler**: Schedules and runs trading tasks
- **System Monitoring**: Monitors system health and performance

## Installation Options

There are two main ways to run the System Trader 24/7:

### Option 1: Systemd Service (Recommended)

This is the recommended approach for production environments. It ensures all components start automatically on system boot and are restarted if they crash.

1. Make the installation script executable:
   ```bash
   chmod +x scripts/install_system_trader_service.sh
   ```

2. Run the installation script as root:
   ```bash
   sudo ./scripts/install_system_trader_service.sh
   ```

3. Start the service:
   ```bash
   sudo systemctl start system_trader
   ```

4. Check the service status:
   ```bash
   sudo systemctl status system_trader
   ```

5. View the logs:
   ```bash
   sudo journalctl -u system_trader -f
   ```

### Option 2: Manual Start

For development or testing environments, you can start the services manually:

1. Make the script executable:
   ```bash
   chmod +x scripts/start_all_services.sh
   ```

2. Run the script:
   ```bash
   ./scripts/start_all_services.sh
   ```

## Monitoring the System

Once the system is running, you can monitor it through:

1. **Grafana Dashboards**: Access at http://localhost:3000
   - Default credentials: admin/admin

2. **Prometheus**: Access at http://localhost:9090
   - View metrics and alerts

3. **MLflow UI**: Access at http://localhost:5000
   - View model training history and performance

4. **Service Status**: Check the status of all services
   ```bash
   ./run.py services --status
   ```

## Troubleshooting

If any component fails to start:

1. Check the logs:
   ```bash
   tail -f logs/*.log
   ```

2. Restart the service:
   ```bash
   sudo systemctl restart system_trader
   ```

3. Start individual components:
   ```bash
   ./run.py services --service <service_name> --start
   ```
   Where `<service_name>` can be: timescaledb, redis, mlflow, prometheus, grafana

4. Check if required ports are available:
   - TimescaleDB: 5432
   - Redis: 6379
   - MLflow: 5000
   - Prometheus: 9090
   - Grafana (with alerts): 3000
   - System Metrics Exporter: 8001
   - Trading Metrics Exporter: 8002
   - Model Metrics Exporter: 8003

## Security Considerations

For production deployments:

1. Configure proper authentication for all services
2. Use HTTPS for web interfaces
3. Set up proper firewall rules
4. Use secure passwords and API keys
5. Consider running services in containers or with restricted permissions