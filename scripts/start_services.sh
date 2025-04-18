#!/bin/bash

# Startup script for system_trader services
# This script ensures all required services are running

LOG_FILE="logs/services.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "Starting system_trader services..."

# Check if TimescaleDB is running
if systemctl is-active postgresql.service > /dev/null 2>&1; then
    log "TimescaleDB is already running"
else
    log "Starting TimescaleDB..."
    sudo systemctl start postgresql.service
    if [ $? -eq 0 ]; then
        log "TimescaleDB started successfully"
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
    nohup mlflow server \
        --host localhost \
        --port 5000 \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        > logs/mlflow.log 2>&1 &
    
    MLFLOW_PID=$!
    log "MLflow started with PID: $MLFLOW_PID"
    
    # Verify MLflow is running
    sleep 5
    if ps -p $MLFLOW_PID > /dev/null; then
        log "MLflow is running properly"
    else
        log "MLflow failed to start or terminated"
    fi
fi

# Check for Prometheus installation
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
            log "Prometheus started with PID: $!"
        else
            log "Prometheus config file not found"
        fi
    fi
else
    log "Prometheus is not installed. Install with: sudo apt-get install prometheus"
fi

# Alertmanager is disabled - using Grafana for alerts directly
log "Alertmanager is disabled - using Grafana for alerts directly"

# Check for Grafana installation
if systemctl list-unit-files | grep -q grafana-server.service; then
    log "Starting Grafana..."
    if systemctl is-active grafana-server.service > /dev/null 2>&1; then
        log "Grafana is already running"
    else
        sudo systemctl start grafana-server.service
        if [ $? -eq 0 ]; then
            log "Grafana started successfully"
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

log "Service startup completed"

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

# Alertmanager disabled - using Grafana for alerts
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