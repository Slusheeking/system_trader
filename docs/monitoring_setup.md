# Monitoring Setup

This document describes the monitoring setup for the System Trader application. The monitoring system consists of Prometheus for metrics collection, Grafana for visualization, and Redis for caching and data storage.

## Components

### Prometheus

Prometheus is a time series database that collects and stores metrics from various sources. It is used to monitor the performance and health of the System Trader application.

- **Port**: 9090
- **Configuration**: `/home/ubuntu/system_trader/config/prometheus/prometheus.yml`
- **Data Directory**: `/home/ubuntu/system_trader/data/prometheus`
- **Service**: `system-trader-prometheus.service`

### Grafana

Grafana is a visualization tool that is used to create dashboards for the metrics collected by Prometheus. It provides a user-friendly interface for monitoring the System Trader application.

- **Port**: 3001
- **Configuration**: `/home/ubuntu/system_trader/config/grafana/grafana.ini`
- **Data Directory**: `/home/ubuntu/system_trader/data/grafana`
- **Service**: `system-trader-grafana.service`
- **Default Credentials**: admin/admin

### Redis

Redis is an in-memory data structure store that is used as a database, cache, and message broker. It is used for caching model predictions, storing temporary data, and sharing data between different components of the system.

- **Port**: 6379
- **Configuration**: `/home/ubuntu/system_trader/config/redis/redis.conf`
- **Data Directory**: `/home/ubuntu/system_trader/data/redis`
- **Service**: `system-trader-redis.service`

## Dashboards

The following dashboards are available in Grafana:

- **Portfolio Dashboard**: View portfolio performance metrics
- **Model Performance Dashboard**: Monitor ML model performance
- **System Health Dashboard**: Monitor system health metrics
- **Data Quality Dashboard**: Monitor data quality metrics

## Starting and Stopping

### Starting the Monitoring Services

To start the monitoring services, run the following command:

```bash
./scripts/start_monitoring_services.sh
```

This will start both Prometheus and Grafana.

### Stopping the Monitoring Services

To stop the monitoring services, run the following command:

```bash
./scripts/stop_monitoring_services.sh
```

This will stop both Grafana and Prometheus.

## Accessing the Monitoring Services

### Prometheus

Prometheus can be accessed at: http://localhost:9090

### Grafana

Grafana can be accessed at: http://localhost:3001

Default credentials: admin/admin

### Redis

Redis can be accessed at: localhost:6379

Redis CLI: `redis-cli`

## Troubleshooting

### Port Conflicts

If there are port conflicts, the start scripts will attempt to resolve them automatically. If they cannot be resolved, you may need to manually stop the conflicting processes or change the port configuration.

### Service Failures

If a service fails to start, check the logs for more information:

```bash
sudo journalctl -u system-trader-prometheus.service -n 50
sudo journalctl -u system-trader-grafana.service -n 50
```

## Configuration

### Prometheus Configuration

The Prometheus configuration is located at `/home/ubuntu/system_trader/config/prometheus/prometheus.yml`. This file defines the scrape configurations and alerting rules.

### Grafana Configuration

The Grafana configuration is located at `/home/ubuntu/system_trader/config/grafana/grafana.ini`. This file defines the server settings, database settings, and other Grafana-specific settings.

## Adding New Metrics

To add new metrics to Prometheus, you need to:

1. Create a new exporter or modify an existing one
2. Update the Prometheus configuration to scrape the new metrics
3. Create a new dashboard or update an existing one in Grafana to visualize the new metrics

## Adding New Dashboards

To add a new dashboard to Grafana:

1. Create a new JSON file in the `/home/ubuntu/system_trader/monitoring/grafana` directory
2. Update the `/home/ubuntu/system_trader/config/grafana/provisioning/dashboards/dashboards.yaml` file to include the new dashboard
3. Restart Grafana to load the new dashboard
