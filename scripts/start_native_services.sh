#!/bin/bash

# Script to start all native systemd services for the trading system
# This replaces the Docker-based approach with direct systemd services

LOG_FILE="logs/start_native_services.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log "Please run as root (use sudo)"
    exit 1
fi

log "Starting all native systemd services for the trading system..."

# Start all services
log "Starting all services..."
systemctl start system-trader-mlflow.service
systemctl start system-trader-prometheus.service
systemctl start system-trader-grafana.service
systemctl start system-trader-scheduler.service
systemctl start system-trader-monitoring.service

# Check service status
log "Checking service status..."
systemctl status system-trader-*

# Display access information
log "Services are available at:"
log "- MLflow: http://localhost:5000"
log "- Prometheus: http://localhost:9090"
log "- Grafana: http://localhost:3000"
log "- System Metrics: http://localhost:8001/metrics"

log "All services are now running. Use 'sudo systemctl stop system-trader-*' to stop all services."
