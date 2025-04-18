#!/bin/bash

# Script to stop Redis metrics exporter
# This script will:
# 1. Find the Redis metrics exporter process
# 2. Stop the process

LOG_FILE="/home/ubuntu/system_trader/logs/redis_exporter.log"
mkdir -p /home/ubuntu/system_trader/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Default port
DEFAULT_PORT=9121

# Find and stop the Redis metrics exporter process
stop_exporter() {
    log "Stopping Redis metrics exporter..."
    
    # Find process IDs using the port
    local pids=$(lsof -t -i:$DEFAULT_PORT 2>/dev/null)
    
    if [ -n "$pids" ]; then
        log "Found processes $pids using port $DEFAULT_PORT. Stopping them..."
        kill $pids 2>/dev/null
        sleep 2
        
        # Check if processes are still running
        if [ -z "$(lsof -t -i:$DEFAULT_PORT 2>/dev/null)" ]; then
            log "Redis metrics exporter stopped successfully."
            return 0
        else
            # Try with SIGKILL if normal kill didn't work
            log "First attempt to stop Redis metrics exporter failed, trying with SIGKILL..."
            kill -9 $pids 2>/dev/null
            sleep 2
            
            if [ -z "$(lsof -t -i:$DEFAULT_PORT 2>/dev/null)" ]; then
                log "Redis metrics exporter stopped successfully with SIGKILL."
                return 0
            else
                log "Failed to stop Redis metrics exporter even after SIGKILL."
                return 1
            fi
        fi
    else
        log "No Redis metrics exporter process found."
        return 0
    fi
}

# Main logic
stop_exporter
exit $?
