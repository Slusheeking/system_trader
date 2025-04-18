#!/bin/bash

# Script to stop all native systemd services for the trading system

LOG_FILE="logs/stop_native_services.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log "Please run as root (use sudo)"
    exit 1
fi

log "Stopping all native systemd services for the trading system..."

# Stop all services
log "Stopping all services..."
systemctl stop system-trader-monitoring.service
systemctl stop system-trader-scheduler.service
systemctl stop system-trader-grafana.service
systemctl stop system-trader-prometheus.service
systemctl stop system-trader-mlflow.service

# Check service status
log "Checking service status..."
systemctl status system-trader-*

log "All services have been stopped."
