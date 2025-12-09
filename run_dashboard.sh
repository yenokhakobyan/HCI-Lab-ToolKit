#!/bin/bash
# HCI Lab Analysis Dashboard Launcher
# Run this script to start the Streamlit dashboard

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Run the Streamlit dashboard
echo "Starting HCI Session Analysis Dashboard..."
streamlit run "$SCRIPT_DIR/src/analysis/dashboard.py" --server.port 8502 --browser.gatherUsageStats false
