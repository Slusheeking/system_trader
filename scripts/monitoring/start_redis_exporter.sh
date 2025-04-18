#!/bin/bash

# Script to start Redis metrics exporter
# This script will:
# 1. Check if port 9121 is already in use
# 2. If it is, try to stop any processes using it
# 3. Start the Redis metrics exporter

LOG_FILE="/home/ubuntu/system_trader/logs/redis_exporter.log"
mkdir -p /home/ubuntu/system_trader/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Default port
DEFAULT_PORT=9121

# Redis metrics exporter
EXPORTER_SCRIPT="/home/ubuntu/system_trader/monitoring/prometheus/exporters/redis_metrics_exporter.py"

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

# Start Redis metrics exporter
start_exporter() {
    log "Starting Redis metrics exporter on port $DEFAULT_PORT..."
    
    # Start exporter in background
    nohup python3 $EXPORTER_SCRIPT --exporter-port $DEFAULT_PORT > $LOG_FILE 2>&1 &
    
    # Check if exporter started successfully
    sleep 2
    if check_port $DEFAULT_PORT; then
        log "Redis metrics exporter started successfully"
        return 0
    else
        log "Failed to start Redis metrics exporter"
        return 1
    fi
}

# Main logic
log "Starting Redis metrics exporter..."

# Check if default port is in use
if check_port $DEFAULT_PORT; then
    log "Port $DEFAULT_PORT is already in use"
    
    # Try to stop any process using the port
    if stop_process_on_port $DEFAULT_PORT; then
        log "Successfully freed port $DEFAULT_PORT"
        start_exporter
    else
        log "Could not free port $DEFAULT_PORT. Exiting."
        exit 1
    fi
else
    # Default port is free, use it
    log "Port $DEFAULT_PORT is available"
    start_exporter
fi
