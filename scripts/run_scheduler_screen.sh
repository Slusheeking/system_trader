#!/bin/bash

# Script to run the System Trader Scheduler in a screen session
# This allows the scheduler to run in the background and persist even if you disconnect

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "Screen is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y screen
fi

# Get the absolute path to the system_trader directory
SYSTEM_TRADER_DIR=$(cd "$(dirname "$(dirname "$0")")" && pwd)
echo "System Trader directory: $SYSTEM_TRADER_DIR"

# Make run.py executable
chmod +x "$SYSTEM_TRADER_DIR/run.py"

# Create logs directory if it doesn't exist
mkdir -p "$SYSTEM_TRADER_DIR/logs"

# Check if a screen session for the scheduler already exists
if screen -list | grep -q "scheduler"; then
    echo "A scheduler screen session is already running."
    echo "To attach to it, use: screen -r scheduler"
    echo "To detach from it (once attached), press Ctrl+A followed by D"
    exit 0
fi

# Start a new screen session for the scheduler
echo "Starting scheduler in a new screen session..."
cd "$SYSTEM_TRADER_DIR"
screen -dmS scheduler -L -Logfile "$SYSTEM_TRADER_DIR/logs/scheduler_screen.log" ./run.py scheduler

# Check if the screen session was created successfully
if screen -list | grep -q "scheduler"; then
    echo "Scheduler started successfully in a screen session named 'scheduler'"
    echo "To attach to it, use: screen -r scheduler"
    echo "To detach from it (once attached), press Ctrl+A followed by D"
    echo "Logs are being saved to: $SYSTEM_TRADER_DIR/logs/scheduler_screen.log"
else
    echo "Failed to start scheduler in a screen session"
    exit 1
fi