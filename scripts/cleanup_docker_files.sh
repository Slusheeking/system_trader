#!/bin/bash

# Script to remove all Docker-related files from the system_trader project
# This is part of the migration from Docker containers to native services

LOG_FILE="logs/cleanup_docker.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "Starting cleanup of Docker-related files..."

# Docker Compose files
log "Removing Docker Compose files..."
rm -f docker-compose.monitoring.yml
rm -f docker-compose.yml

# Dockerfile files
log "Removing Dockerfile files..."
rm -f Dockerfile.exporter
rm -f Dockerfile.exporter.fixed
rm -f Dockerfile.monitoring.minimal
rm -f test.Dockerfile
rm -f finbert/Dockerfile
rm -f finbert/Dockerfile.arm64

# Docker-related configuration files
log "Removing Docker-related configuration files..."
rm -f config/docker_system_config_linux.yaml
rm -f config/docker_system_config.yaml
rm -f config/system_trader_docker.service

# Docker-related scripts
log "Removing Docker-related scripts..."
rm -f scripts/install_docker_service.sh
rm -f scripts/start_docker_services.sh
rm -f scripts/start_monitoring_services.sh

# Check if any systemd service is installed
if [ -f "/etc/systemd/system/system_trader_docker.service" ]; then
    log "Removing Docker systemd service..."
    sudo systemctl stop system_trader_docker.service
    sudo systemctl disable system_trader_docker.service
    sudo rm -f /etc/systemd/system/system_trader_docker.service
    sudo systemctl daemon-reload
fi

log "Cleanup complete. All Docker-related files have been removed."
