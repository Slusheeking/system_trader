#!/bin/bash

# Script to install the System Trader as a systemd service
# This will enable all components to run 24/7 and restart automatically

echo "Installing System Trader as a systemd service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Get the absolute path to the system_trader directory
SYSTEM_TRADER_DIR=$(cd "$(dirname "$(dirname "$0")")" && pwd)
echo "System Trader directory: $SYSTEM_TRADER_DIR"

# Make run.py executable
chmod +x "$SYSTEM_TRADER_DIR/run.py"

# Make start_all_services.sh executable
chmod +x "$SYSTEM_TRADER_DIR/scripts/start_all_services.sh"

# Copy the service file to systemd directory
cp "$SYSTEM_TRADER_DIR/config/system_trader_full.service" /etc/systemd/system/system_trader.service

# Update the WorkingDirectory and ExecStart paths in the service file
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$SYSTEM_TRADER_DIR|" /etc/systemd/system/system_trader.service
sed -i "s|ExecStart=.*|ExecStart=$SYSTEM_TRADER_DIR/scripts/start_all_services.sh|" /etc/systemd/system/system_trader.service

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable system_trader.service

echo "Service installed and enabled to start on boot"
echo "To start the service now, run: sudo systemctl start system_trader.service"
echo "To check status, run: sudo systemctl status system_trader.service"
echo "To view logs, run: sudo journalctl -u system_trader.service -f"
echo ""
echo "This service will ensure all components are running 24/7:"
echo "- TimescaleDB database"
echo "- Redis cache"
echo "- MLflow tracking server"
echo "- Prometheus monitoring"
echo "- Alertmanager for alerts"
echo "- Grafana dashboards"
echo "- Task Scheduler"
echo "- System Monitoring"