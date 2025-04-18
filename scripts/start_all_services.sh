#!/bin/bash

# Comprehensive script to start all System Trader services
# This includes database, scheduler, and all monitoring components

LOG_FILE="logs/all_services.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "Starting all System Trader services..."

# Create a PID file to indicate the service is running
echo $$ > /tmp/system_trader_services.pid

# Function to check if a port is in use
is_port_in_use() {
    netstat -tuln | grep -q ":$1 "
    return $?
}

# Function to wait for a service to be available on a port
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=$3
    local attempt=1
    
    log "Waiting for $service_name to be available on port $port..."
    while ! is_port_in_use $port; do
        if [ $attempt -gt $max_attempts ]; then
            log "Timed out waiting for $service_name on port $port"
            return 1
        fi
        log "Attempt $attempt/$max_attempts: $service_name not yet available on port $port, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    log "$service_name is available on port $port"
    return 0
}

# Check if TimescaleDB is running
if systemctl is-active postgresql.service > /dev/null 2>&1; then
    log "TimescaleDB is already running"
else
    log "Starting TimescaleDB..."
    sudo systemctl start postgresql.service
    if [ $? -eq 0 ]; then
        log "TimescaleDB started successfully"
        wait_for_service "TimescaleDB" 5432 12
    else
        log "Failed to start TimescaleDB"
    fi
fi

# Check if Redis is running
if systemctl is-active redis-server.service > /dev/null 2>&1; then
    log "Redis is already running"
else
    log "Starting Redis..."
    sudo systemctl start redis-server.service
    if [ $? -eq 0 ]; then
        log "Redis started successfully"
        wait_for_service "Redis" 6379 12
    else
        log "Failed to start Redis"
    fi
fi

# Start MLflow as a background process
log "Starting MLflow..."
MLFLOW_PID=$(pgrep -f "mlflow server")
if [ -n "$MLFLOW_PID" ]; then
    log "MLflow is already running (PID: $MLFLOW_PID)"
else
    # Create a Python script to start MLflow
    cat > start_mlflow.py << 'EOF'
import sys
import os
import subprocess
import time

# Ensure we're using the system MLflow, not the local one
sys.path = [p for p in sys.path if not p.endswith('system_trader')]

try:
    # Try to import mlflow directly
    import mlflow.server
    from mlflow.server import app
    
    # Start the server
    app.run(host='localhost', port=5000, backend_store_uri='sqlite:///mlflow.db', default_artifact_root='./mlruns')
except ImportError:
    # Fall back to subprocess if direct import fails
    subprocess.run([
        'python', '-m', 'mlflow', 'server',
        '--host', 'localhost',
        '--port', '5000',
        '--backend-store-uri', 'sqlite:///mlflow.db',
        '--default-artifact-root', './mlruns'
    ])
EOF

    # Run the script
    nohup python start_mlflow.py > logs/mlflow.log 2>&1 &
    
    MLFLOW_PID=$!
    log "MLflow started with PID: $MLFLOW_PID"
    
    # Verify MLflow is running
    wait_for_service "MLflow" 5000 12
fi

# Check for Prometheus installation and start it
if command -v prometheus > /dev/null 2>&1; then
    log "Starting Prometheus..."
    PROMETHEUS_PID=$(pgrep -f "prometheus")
    if [ -n "$PROMETHEUS_PID" ]; then
        log "Prometheus is already running (PID: $PROMETHEUS_PID)"
    else
        # Start Prometheus with the configuration file
        if [ -f "config/prometheus/prometheus.yml" ]; then
            nohup prometheus \
                --config.file=config/prometheus/prometheus.yml \
                --storage.tsdb.path=/tmp/prometheus \
                > logs/prometheus.log 2>&1 &
            PROMETHEUS_PID=$!
            log "Prometheus started with PID: $PROMETHEUS_PID"
            wait_for_service "Prometheus" 9090 12
        else
            log "Prometheus config file not found"
        fi
    fi
else
    log "Prometheus is not installed. Install with: sudo apt-get install prometheus"
fi

# Note: Alertmanager is disabled - using Grafana for alerts directly
log "Alertmanager is disabled - using Grafana for alerts directly"

# Check for Grafana installation and start it
if systemctl list-unit-files | grep -q grafana-server.service; then
    log "Starting Grafana..."
    if systemctl is-active grafana-server.service > /dev/null 2>&1; then
        log "Grafana is already running"
    else
        sudo systemctl start grafana-server.service
        if [ $? -eq 0 ]; then
            log "Grafana started successfully"
            wait_for_service "Grafana" 3000 12
        else
            log "Failed to start Grafana"
        fi
    fi
else
    log "Grafana is not installed. Install with: sudo apt-get install grafana"
fi

# Start the scheduler as a background process
log "Starting Task Scheduler..."
SCHEDULER_PID=$(pgrep -f "./run.py scheduler")
if [ -n "$SCHEDULER_PID" ]; then
    log "Task Scheduler is already running (PID: $SCHEDULER_PID)"
else
    nohup ./run.py scheduler > logs/scheduler.log 2>&1 &
    SCHEDULER_PID=$!
    log "Task Scheduler started with PID: $SCHEDULER_PID"
    
    # Verify scheduler is running
    sleep 5
    if ps -p $SCHEDULER_PID > /dev/null; then
        log "Task Scheduler is running properly"
    else
        log "Task Scheduler failed to start or terminated"
    fi
fi

# Start system monitoring exporters
log "Starting system monitoring exporters..."
MONITOR_PID=$(pgrep -f "./run.py monitor")
if [ -n "$MONITOR_PID" ]; then
    log "System Monitoring is already running (PID: $MONITOR_PID)"
else
    nohup ./run.py monitor > logs/monitor.log 2>&1 &
    MONITOR_PID=$!
    log "System Monitoring started with PID: $MONITOR_PID"
    
    # Verify monitoring is running
    sleep 5
    if ps -p $MONITOR_PID > /dev/null; then
        log "System Monitoring is running properly"
    else
        log "System Monitoring failed to start or terminated"
    fi
fi
log "System monitoring started with PID: $MONITOR_PID"

# Start the main trading system
log "Starting Trading System..."
SYSTEM_PID=$(pgrep -f "./run.py system")
if [ -n "$SYSTEM_PID" ]; then
    log "Trading System is already running (PID: $SYSTEM_PID)"
else
    nohup ./run.py system --test > logs/system.log 2>&1 &
    SYSTEM_PID=$!
    log "Trading System started with PID: $SYSTEM_PID"
    
    # Verify system is running
    sleep 5
    if ps -p $SYSTEM_PID > /dev/null; then
        log "Trading System is running properly"
    else
        log "Trading System failed to start or terminated"
    fi
fi

log "All services startup completed"

# Display service status
echo ""
echo "Service Status:"
echo "==============="

# Check TimescaleDB
if systemctl is-active postgresql.service > /dev/null 2>&1; then
    echo "TimescaleDB: running"
else
    echo "TimescaleDB: stopped"
fi

# Check Redis
if systemctl is-active redis-server.service > /dev/null 2>&1; then
    echo "Redis: running"
else
    echo "Redis: stopped"
fi

# Check MLflow
if pgrep -f "mlflow server" > /dev/null; then
    echo "MLflow: running"
else
    echo "MLflow: stopped"
fi

# Check Prometheus
if pgrep -f "prometheus" > /dev/null; then
    echo "Prometheus: running"
else
    echo "Prometheus: stopped"
fi


# Check Alertmanager
echo "Alertmanager: disabled (using Grafana for alerts)"

# Check Grafana
if systemctl is-active grafana-server.service > /dev/null 2>&1; then
    echo "Grafana: running"
else
    echo "Grafana: stopped"
fi

# Check Scheduler
if pgrep -f "./run.py scheduler" > /dev/null; then
    echo "Task Scheduler: running"
else
    echo "Task Scheduler: stopped"
fi

# Check System Monitoring
if pgrep -f "./run.py monitor" > /dev/null; then
    echo "System Monitoring: running"
else
    echo "System Monitoring: stopped"
fi

# Check Trading System
if pgrep -f "./run.py system" > /dev/null; then
    echo "Trading System: running"
else
    echo "Trading System: stopped"
fi

# Keep the script running to maintain the service
log "Service is now running. Use 'systemctl stop system_trader_full' to stop."
tail -f /dev/null