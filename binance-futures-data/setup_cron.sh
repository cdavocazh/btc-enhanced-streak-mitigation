#!/bin/bash
#
# Setup Cron Job for Automated Data Refresh
# ==========================================
# This script sets up a cron job to run data refresh every 5 minutes.
#
# Usage:
#   ./setup_cron.sh           # Install the cron job
#   ./setup_cron.sh --remove  # Remove the cron job
#   ./setup_cron.sh --status  # Show current cron status
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFRESH_SCRIPT="${SCRIPT_DIR}/refresh_data.py"
LOG_FILE="${SCRIPT_DIR}/refresh.log"
CRON_ID="btc_data_refresh"

# Python path (use the system Python or a specific venv)
PYTHON_PATH="python"

# Check if running with --remove flag
if [[ "$1" == "--remove" ]]; then
    echo "Removing cron job..."
    crontab -l 2>/dev/null | grep -v "$CRON_ID" | crontab -
    echo "Cron job removed."
    exit 0
fi

# Check if running with --status flag
if [[ "$1" == "--status" ]]; then
    echo "Current cron jobs:"
    echo "=================="
    crontab -l 2>/dev/null | grep "$CRON_ID" || echo "No $CRON_ID cron job found."
    echo ""
    echo "Last 10 lines of refresh log:"
    echo "=============================="
    tail -10 "$LOG_FILE" 2>/dev/null || echo "No log file found."
    exit 0
fi

# Verify refresh script exists
if [[ ! -f "$REFRESH_SCRIPT" ]]; then
    echo "Error: refresh_data.py not found at $REFRESH_SCRIPT"
    exit 1
fi

# Verify Python is available
if ! command -v $PYTHON_PATH &> /dev/null; then
    echo "Error: Python not found. Please update PYTHON_PATH in this script."
    exit 1
fi

echo "Setting up cron job for data refresh..."
echo "  Script: $REFRESH_SCRIPT"
echo "  Log: $LOG_FILE"
echo "  Interval: Every 5 minutes"
echo ""

# Create the cron entry
# Format: minute hour day month weekday command
CRON_ENTRY="*/5 * * * * cd $SCRIPT_DIR && $PYTHON_PATH $REFRESH_SCRIPT >> $LOG_FILE 2>&1 # $CRON_ID"

# Remove existing entry (if any) and add new one
(crontab -l 2>/dev/null | grep -v "$CRON_ID"; echo "$CRON_ENTRY") | crontab -

echo "Cron job installed successfully!"
echo ""
echo "To verify:"
echo "  crontab -l | grep $CRON_ID"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To remove:"
echo "  $0 --remove"
