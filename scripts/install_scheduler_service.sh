#!/bin/bash

# Script to install the System Trader Scheduler as a systemd service
# This will enable the scheduler to run 24/7 and restart automatically

echo "Installing System Trader Scheduler as a systemd service..."

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

# Copy the service file to systemd directory
cp "$SYSTEM_TRADER_DIR/config/system_trader_scheduler.service" /etc/systemd/system/

# Make sure run.py is properly executable
chmod +x "$SYSTEM_TRADER_DIR/run.py"

# Update the WorkingDirectory and ExecStart paths in the service file
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$SYSTEM_TRADER_DIR|" /etc/systemd/system/system_trader_scheduler.service
sed -i "s|ExecStart=.*|ExecStart=/usr/bin/python3 $SYSTEM_TRADER_DIR/run.py scheduler --config $SYSTEM_TRADER_DIR/config/scheduler.yaml|" /etc/systemd/system/system_trader_scheduler.service

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable system_trader_scheduler.service

# Start the service immediately
systemctl start system_trader_scheduler.service

# Check service status
systemctl status system_trader_scheduler.service

echo ""
echo "Service installed, enabled, and started!"
echo "The scheduler will now start automatically when the system boots."
echo ""
echo "Common commands:"
echo "  - Check status:      sudo systemctl status system_trader_scheduler.service"
echo "  - Stop service:      sudo systemctl stop system_trader_scheduler.service"
echo "  - Restart service:   sudo systemctl restart system_trader_scheduler.service"
echo "  - View logs:         sudo journalctl -u system_trader_scheduler.service -f"