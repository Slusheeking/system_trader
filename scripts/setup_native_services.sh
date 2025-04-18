#!/bin/bash

# Script to set up native systemd services for the trading system
# This replaces the Docker-based approach with direct systemd services

LOG_FILE="logs/setup_native_services.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log "Please run as root (use sudo)"
    exit 1
fi

# Base directory
BASE_DIR="/home/ubuntu/system_trader"
cd $BASE_DIR

log "Setting up native systemd services for the trading system..."

# Create systemd service files directory if it doesn't exist
mkdir -p $BASE_DIR/config/systemd

# Create MLflow service file
log "Creating MLflow service file..."
cat > $BASE_DIR/config/systemd/system-trader-mlflow.service << 'EOF'
[Unit]
Description=MLflow Tracking Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/system_trader
Environment="PYTHONPATH=/usr/local/lib/python3.10/dist-packages"
ExecStart=/usr/bin/python3 -c "import mlflow.server; mlflow.server.app.run(host='0.0.0.0', port=5000, backend_store_uri='sqlite:///mlflow.db', default_artifact_root='./mlruns')"
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create Prometheus service file
log "Creating Prometheus service file..."
cat > $BASE_DIR/config/systemd/system-trader-prometheus.service << 'EOF'
[Unit]
Description=Prometheus Time Series Database
Documentation=https://prometheus.io/docs/introduction/overview/
Wants=network-online.target
After=network-online.target

[Service]
User=ubuntu
Group=ubuntu
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file=/home/ubuntu/system_trader/config/prometheus/prometheus.yml \
    --storage.tsdb.path=/home/ubuntu/system_trader/data/prometheus \
    --storage.tsdb.retention.time=30d \
    --web.console.templates=/usr/share/prometheus/consoles \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.listen-address=0.0.0.0:9090 \
    --web.external-url=http://localhost:9090

# Store data in /home/ubuntu/system_trader/data/prometheus
WorkingDirectory=/home/ubuntu/system_trader

# Restart on failure but with exponential backoff
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create Grafana service file
log "Creating Grafana service file..."
cat > $BASE_DIR/config/systemd/system-trader-grafana.service << 'EOF'
[Unit]
Description=Grafana instance
Documentation=http://docs.grafana.org
Wants=network-online.target
After=network-online.target

[Service]
User=ubuntu
Group=ubuntu
Type=simple
Restart=on-failure
WorkingDirectory=/home/ubuntu/system_trader
ExecStart=/usr/sbin/grafana-server \
          --config=/home/ubuntu/system_trader/config/grafana/grafana.ini \
          --homepath=/usr/share/grafana \
          cfg:default.paths.logs=/home/ubuntu/system_trader/logs/grafana \
          cfg:default.paths.data=/home/ubuntu/system_trader/data/grafana \
          cfg:default.paths.plugins=/home/ubuntu/system_trader/data/grafana/plugins \
          cfg:default.paths.provisioning=/home/ubuntu/system_trader/config/grafana/provisioning

[Install]
WantedBy=multi-user.target
EOF

# Create Scheduler service file
log "Creating Scheduler service file..."
cat > $BASE_DIR/config/systemd/system-trader-scheduler.service << 'EOF'
[Unit]
Description=System Trader Scheduler
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/system_trader
Environment="PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/home/ubuntu/system_trader"
ExecStart=/usr/bin/python3 run.py scheduler
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create Monitoring service file
log "Creating Monitoring service file..."
cat > $BASE_DIR/config/systemd/system-trader-monitoring.service << 'EOF'
[Unit]
Description=System Trader Monitoring
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/system_trader
Environment="PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/home/ubuntu/system_trader"
ExecStart=/usr/bin/python3 run.py monitor --no-services
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create directories for data
log "Creating data directories..."
mkdir -p $BASE_DIR/data/prometheus
mkdir -p $BASE_DIR/data/grafana
mkdir -p $BASE_DIR/data/grafana/plugins
mkdir -p $BASE_DIR/logs/grafana

# Set permissions
log "Setting permissions..."
chown -R ubuntu:ubuntu $BASE_DIR/data
chown -R ubuntu:ubuntu $BASE_DIR/logs

# Install services
log "Installing systemd services..."
cp $BASE_DIR/config/systemd/system-trader-mlflow.service /etc/systemd/system/
cp $BASE_DIR/config/systemd/system-trader-prometheus.service /etc/systemd/system/
cp $BASE_DIR/config/systemd/system-trader-grafana.service /etc/systemd/system/
cp $BASE_DIR/config/systemd/system-trader-scheduler.service /etc/systemd/system/
cp $BASE_DIR/config/systemd/system-trader-monitoring.service /etc/systemd/system/

# Reload systemd
log "Reloading systemd..."
systemctl daemon-reload

# Enable services
log "Enabling services..."
systemctl enable system-trader-mlflow.service
systemctl enable system-trader-prometheus.service
systemctl enable system-trader-grafana.service
systemctl enable system-trader-scheduler.service
systemctl enable system-trader-monitoring.service

# Start services
log "Starting services..."
systemctl start system-trader-mlflow.service
systemctl start system-trader-prometheus.service
systemctl start system-trader-grafana.service
systemctl start system-trader-scheduler.service
systemctl start system-trader-monitoring.service

# Check service status
log "Checking service status..."
systemctl status system-trader-mlflow.service
systemctl status system-trader-prometheus.service
systemctl status system-trader-grafana.service
systemctl status system-trader-scheduler.service
systemctl status system-trader-monitoring.service

log "Setup complete. All services are now running as native systemd services."
log "You can check the status of all services with: sudo systemctl status system-trader-*"
