#!/bin/bash

# Script to download and install Prometheus and Grafana
# Alertmanager is disabled - using Grafana for alerts directly
# This script will download the binaries and set them up in the local directory

LOG_FILE="logs/install_monitoring.log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

INSTALL_DIR="$HOME/monitoring_tools"
mkdir -p $INSTALL_DIR

# Function to download and extract a tar.gz file
download_and_extract() {
    local url=$1
    local output_file=$2
    local extract_dir=$3
    
    log "Downloading $url to $output_file"
    curl -L $url -o $output_file
    
    log "Extracting $output_file to $extract_dir"
    mkdir -p $extract_dir
    tar -xzf $output_file -C $extract_dir --strip-components=1
    
    log "Cleaning up $output_file"
    rm $output_file
}

# Install Prometheus
install_prometheus() {
    log "Installing Prometheus..."
    
    # Download and extract Prometheus
    PROMETHEUS_VERSION="2.45.0"
    PROMETHEUS_URL="https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-arm64.tar.gz"
    PROMETHEUS_TAR="/tmp/prometheus.tar.gz"
    PROMETHEUS_DIR="$INSTALL_DIR/prometheus"
    
    download_and_extract $PROMETHEUS_URL $PROMETHEUS_TAR $PROMETHEUS_DIR
    
    # Create symbolic links
    ln -sf $PROMETHEUS_DIR/prometheus $HOME/bin/prometheus
    ln -sf $PROMETHEUS_DIR/promtool $HOME/bin/promtool
    
    # Copy configuration files
    mkdir -p $PROMETHEUS_DIR/config
    cp -f config/prometheus/prometheus.yml $PROMETHEUS_DIR/config/
    cp -rf config/prometheus/rules $PROMETHEUS_DIR/config/
    
    log "Prometheus installed successfully"
}

# Install Alertmanager - DISABLED
# Using Grafana directly for alerts instead
# Keeping this function commented out for reference if needed in the future
# install_alertmanager() {
#     log "Installing Alertmanager..."
#
#     # Download and extract Alertmanager
#     ALERTMANAGER_VERSION="0.26.0"
#     ALERTMANAGER_URL="https://github.com/prometheus/alertmanager/releases/download/v${ALERTMANAGER_VERSION}/alertmanager-${ALERTMANAGER_VERSION}.linux-arm64.tar.gz"
#     ALERTMANAGER_TAR="/tmp/alertmanager.tar.gz"
#     ALERTMANAGER_DIR="$INSTALL_DIR/alertmanager"
#
#     download_and_extract $ALERTMANAGER_URL $ALERTMANAGER_TAR $ALERTMANAGER_DIR
#
#     # Create symbolic links
#     ln -sf $ALERTMANAGER_DIR/alertmanager $HOME/bin/alertmanager
#     ln -sf $ALERTMANAGER_DIR/amtool $HOME/bin/amtool
#
#     # Copy configuration files
#     mkdir -p $ALERTMANAGER_DIR/config
#     cp -f config/prometheus/alertmanager.yml $ALERTMANAGER_DIR/config/
#
#     log "Alertmanager installed successfully"
# }

# Install Grafana
install_grafana() {
    log "Installing Grafana..."
    
    # Download and extract Grafana
    GRAFANA_VERSION="10.2.0"
    GRAFANA_URL="https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-arm64.tar.gz"
    GRAFANA_TAR="/tmp/grafana.tar.gz"
    GRAFANA_DIR="$INSTALL_DIR/grafana"
    
    download_and_extract $GRAFANA_URL $GRAFANA_TAR $GRAFANA_DIR
    
    # Create symbolic links
    ln -sf $GRAFANA_DIR/bin/grafana $HOME/bin/grafana-server
    ln -sf $GRAFANA_DIR/bin/grafana-cli $HOME/bin/grafana-cli
    
    # Copy configuration files
    mkdir -p $GRAFANA_DIR/conf
    cp -f config/grafana/grafana.ini $GRAFANA_DIR/conf/
    cp -rf config/grafana/provisioning $GRAFANA_DIR/conf/
    
    log "Grafana installed successfully"
}

# Create bin directory if it doesn't exist
mkdir -p $HOME/bin

# Install all tools
log "Starting installation of monitoring tools..."
install_prometheus
# Alertmanager disabled - using Grafana for alerts directly
log "Alertmanager is disabled - using Grafana for alerts directly"
install_grafana

log "All monitoring tools installed successfully"
log "Add $HOME/bin to your PATH to use the tools"

# Update the PATH for the current session
export PATH="$HOME/bin:$PATH"

# Check if tools are available
log "Checking if tools are available..."
which prometheus && log "Prometheus is available" || log "Prometheus is not available"
log "Alertmanager is disabled - using Grafana for alerts directly"
which grafana-server && log "Grafana is available" || log "Grafana is not available"

log "Installation complete"