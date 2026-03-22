#!/bin/bash
# HCI Lab Analysis Dashboard Launcher
# Configuration via environment variables (see .env.example)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

PORT="${HCI_DASHBOARD_PORT:-8502}"

echo "Starting HCI Session Analysis Dashboard on port $PORT..."
streamlit run "$SCRIPT_DIR/src/analysis/dashboard.py" \
    --server.port "$PORT" \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false
