#!/bin/bash

# Script to install the System Trader Monitoring Service
# This will set up a systemd service to run the monitoring components

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

# Get the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Copy the service file to systemd
echo "Installing system_trader_monitoring.service..."
cp "$PROJECT_DIR/config/system_trader_monitoring.service" /etc/systemd/system/

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable system_trader_monitoring.service

echo "System Trader Monitoring Service has been installed and enabled."
echo "You can start it with: sudo systemctl start system_trader_monitoring"
echo "You can check its status with: sudo systemctl status system_trader_monitoring"
echo "You can stop it with: sudo systemctl stop system_trader_monitoring"
