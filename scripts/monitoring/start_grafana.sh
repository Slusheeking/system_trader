#!/bin/bash

# Script to start Grafana with proper configuration
# This script will:
# 1. Check if port 3000 is already in use
# 2. If it is, try to stop any processes using it
# 3. Start Grafana with the appropriate configuration

LOG_FILE="/home/ubuntu/system_trader/logs/grafana.log"
mkdir -p /home/ubuntu/system_trader/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Default port
DEFAULT_PORT=3001

# Grafana configuration
CONFIG_FILE="/home/ubuntu/system_trader/config/grafana/grafana.ini"
HOME_PATH="/usr/share/grafana"
LOGS_PATH="/home/ubuntu/system_trader/logs/grafana"
DATA_PATH="/home/ubuntu/system_trader/data/grafana"
PLUGINS_PATH="/home/ubuntu/system_trader/data/grafana/plugins"
PROVISIONING_PATH="/home/ubuntu/system_trader/config/grafana/provisioning"

# Check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Try to stop any process using the port
stop_process_on_port() {
    local port=$1
    log "Attempting to stop process using port $port..."
    
    # Find process IDs using the port
    local pids=$(lsof -t -i:$port 2>/dev/null)
    
    if [ -n "$pids" ]; then
        log "Found processes $pids using port $port. Stopping them..."
        kill $pids 2>/dev/null
        sleep 2
        
        # Check if port is now free
        if ! check_port $port; then
            log "Successfully freed port $port"
            return 0
        else
            # Try with SIGKILL if normal kill didn't work
            log "First attempt to free port $port failed, trying with SIGKILL..."
            kill -9 $pids 2>/dev/null
            sleep 2
            
            if ! check_port $port; then
                log "Successfully freed port $port with SIGKILL"
                return 0
            else
                log "Failed to free port $port even after SIGKILL"
                return 1
            fi
        fi
    else
        log "No processes found using port $port"
        return 1
    fi
}

# Try to stop Docker container using the port
stop_docker_container() {
    local port=$1
    log "Attempting to stop Docker container using port $port..."
    
    # Check if docker command is available and we have permission
    if ! command -v docker &> /dev/null || ! docker ps &> /dev/null; then
        log "Docker command not available or no permission to access Docker"
        return 1
    fi
    
    # Find container ID using the port
    local container_id=$(docker ps | grep $port | awk '{print $1}')
    
    if [ -n "$container_id" ]; then
        log "Found Docker container $container_id using port $port. Stopping it..."
        docker stop $container_id
        sleep 2
        
        # Check if port is now free
        if ! check_port $port; then
            log "Successfully stopped container and freed port $port"
            return 0
        else
            log "Failed to free port $port even after stopping container"
            return 1
        fi
    else
        log "No Docker container found using port $port"
        return 1
    fi
}

# Start Grafana
start_grafana() {
    log "Starting Grafana on port $DEFAULT_PORT..."
    
    # Ensure data directories exist
    mkdir -p $DATA_PATH
    mkdir -p $LOGS_PATH
    mkdir -p $PLUGINS_PATH
    
    # Start Grafana
    /usr/sbin/grafana-server \
        --config=$CONFIG_FILE \
        --homepath=$HOME_PATH \
        cfg:default.paths.logs=$LOGS_PATH \
        cfg:default.paths.data=$DATA_PATH \
        cfg:default.paths.plugins=$PLUGINS_PATH \
        cfg:default.paths.provisioning=$PROVISIONING_PATH
}

# Main logic
log "Starting Grafana service..."

# Check if default port is in use
if check_port $DEFAULT_PORT; then
    log "Port $DEFAULT_PORT is already in use"
    
    # Try to stop any process using the port
    if stop_process_on_port $DEFAULT_PORT; then
        log "Successfully freed port $DEFAULT_PORT"
        start_grafana
    # Try to stop Docker container if process stop failed
    elif stop_docker_container $DEFAULT_PORT; then
        log "Successfully freed port $DEFAULT_PORT"
        start_grafana
    else
        log "Could not free port $DEFAULT_PORT. Exiting."
        exit 1
    fi
else
    # Default port is free, use it
    log "Port $DEFAULT_PORT is available"
    start_grafana
fi
