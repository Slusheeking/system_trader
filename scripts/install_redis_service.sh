#!/bin/bash

# Script to install Redis service
# This script will:
# 1. Check if Redis is installed
# 2. Install Redis if needed
# 3. Copy the service file to systemd
# 4. Enable and start the service

# Log file
LOG_FILE="/home/ubuntu/system_trader/logs/install_redis.log"
mkdir -p /home/ubuntu/system_trader/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Check if Redis is installed
check_redis() {
    if command -v redis-server &> /dev/null; then
        log "Redis is already installed"
        return 0
    else
        log "Redis is not installed"
        return 1
    fi
}

# Install Redis
install_redis() {
    log "Installing Redis..."
    sudo apt-get update
    sudo apt-get install -y redis-server
    
    if [ $? -eq 0 ]; then
        log "Redis installed successfully"
        return 0
    else
        log "Failed to install Redis"
        return 1
    fi
}

# Install Redis service
install_service() {
    log "Installing Redis service..."
    
    # Create data directory
    mkdir -p /home/ubuntu/system_trader/data/redis
    
    # Copy service file
    sudo cp /home/ubuntu/system_trader/config/redis/system-trader-redis.service /etc/systemd/system/
    
    if [ $? -eq 0 ]; then
        log "Service file copied successfully"
    else
        log "Failed to copy service file"
        return 1
    fi
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    if [ $? -eq 0 ]; then
        log "Systemd reloaded successfully"
    else
        log "Failed to reload systemd"
        return 1
    fi
    
    # Enable service
    sudo systemctl enable system-trader-redis.service
    
    if [ $? -eq 0 ]; then
        log "Service enabled successfully"
    else
        log "Failed to enable service"
        return 1
    fi
    
    # Start service
    sudo systemctl start system-trader-redis.service
    
    if [ $? -eq 0 ]; then
        log "Service started successfully"
    else
        log "Failed to start service"
        return 1
    fi
    
    return 0
}

# Check service status
check_service() {
    log "Checking Redis service status..."
    
    sudo systemctl status system-trader-redis.service
    
    if [ $? -eq 0 ]; then
        log "Redis service is running"
        return 0
    else
        log "Redis service is not running"
        return 1
    fi
}

# Main logic
log "Starting Redis service installation..."

# Check if Redis is installed
if ! check_redis; then
    # Install Redis
    if ! install_redis; then
        log "Failed to install Redis. Exiting."
        exit 1
    fi
fi

# Install service
if ! install_service; then
    log "Failed to install Redis service. Exiting."
    exit 1
fi

# Check service status
if ! check_service; then
    log "Redis service is not running. Please check the logs."
    exit 1
fi

log "Redis service installation completed successfully."
