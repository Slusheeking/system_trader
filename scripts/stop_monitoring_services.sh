#!/bin/bash

# Script to stop all monitoring services
# This script will:
# 1. Stop Grafana
# 2. Stop Prometheus
# 3. Stop Redis

echo "Stopping monitoring services..."

# Stop Grafana
echo "Stopping Grafana..."
sudo systemctl stop system-trader-grafana.service
if [ $? -eq 0 ]; then
    echo "Grafana stopped successfully."
else
    echo "Failed to stop Grafana."
    exit 1
fi

# Stop Prometheus
echo "Stopping Prometheus..."
sudo systemctl stop system-trader-prometheus.service
if [ $? -eq 0 ]; then
    echo "Prometheus stopped successfully."
else
    echo "Failed to stop Prometheus."
    exit 1
fi

# Stop Redis
echo "Stopping Redis..."
sudo systemctl stop system-trader-redis.service
if [ $? -eq 0 ]; then
    echo "Redis stopped successfully."
else
    echo "Failed to stop Redis."
    exit 1
fi

echo "All monitoring services stopped successfully."
