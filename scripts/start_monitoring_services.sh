#!/bin/bash

# Script to start all monitoring services
# This script will:
# 1. Start Prometheus
# 2. Start Grafana
# 3. Start Redis

echo "Starting monitoring services..."

# Start Prometheus
echo "Starting Prometheus..."
sudo systemctl restart system-trader-prometheus.service
if [ $? -eq 0 ]; then
    echo "Prometheus started successfully."
else
    echo "Failed to start Prometheus."
    exit 1
fi

# Start Grafana
echo "Starting Grafana..."
sudo systemctl restart system-trader-grafana.service
if [ $? -eq 0 ]; then
    echo "Grafana started successfully."
else
    echo "Failed to start Grafana."
    exit 1
fi

# Start Redis
echo "Starting Redis..."
sudo systemctl restart system-trader-redis.service
if [ $? -eq 0 ]; then
    echo "Redis started successfully."
else
    echo "Failed to start Redis."
    exit 1
fi

# Start Redis Exporter
echo "Starting Redis Exporter..."
/home/ubuntu/system_trader/scripts/monitoring/start_redis_exporter.sh
if [ $? -eq 0 ]; then
    echo "Redis Exporter started successfully."
else
    echo "Failed to start Redis Exporter."
    exit 1
fi

echo "All monitoring services started successfully."
echo "Prometheus is available at: http://localhost:9090"
echo "Grafana is available at: http://localhost:3001"
echo "Redis is available at: localhost:6379"
echo "Grafana default credentials: admin/admin"
