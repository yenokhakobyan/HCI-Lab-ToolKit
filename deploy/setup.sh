#!/bin/bash
set -euo pipefail

# HCI ToolKit — Server Setup Script
# Run as root on Ubuntu/Debian

INSTALL_DIR="/opt/hci-toolkit"
HCI_USER="hci"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== HCI ToolKit Server Setup ==="
echo "Install dir: $INSTALL_DIR"
echo "Source:      $REPO_DIR"
echo ""

# --- 1. System dependencies ---
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip nginx certbot python3-certbot-nginx

# --- 2. Create service user ---
echo "[2/7] Creating service user '$HCI_USER'..."
if ! id "$HCI_USER" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$INSTALL_DIR" "$HCI_USER"
fi

# --- 3. Copy project files ---
echo "[3/7] Copying project to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
rsync -a --exclude='venv' --exclude='.venv' --exclude='data/raw' \
    --exclude='__pycache__' --exclude='.git' --exclude='.claude' \
    "$REPO_DIR/" "$INSTALL_DIR/"

# --- 4. Create venv and install dependencies ---
echo "[4/7] Setting up Python virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"

# --- 5. Setup configuration ---
echo "[5/7] Setting up configuration..."
mkdir -p "$INSTALL_DIR/data/raw/web_hci"

if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    # Set the data directory
    sed -i "s|HCI_DATA_DIR=.*|HCI_DATA_DIR=$INSTALL_DIR/data/raw/web_hci|" "$INSTALL_DIR/.env"
    echo "  Created .env from template — edit $INSTALL_DIR/.env to customize"
else
    echo "  .env already exists, skipping"
fi

chown -R "$HCI_USER:$HCI_USER" "$INSTALL_DIR"

# --- 6. Install systemd services ---
echo "[6/7] Installing systemd services..."
cp "$INSTALL_DIR/deploy/hci-collector.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/hci-dashboard.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable hci-collector hci-dashboard
systemctl start hci-collector hci-dashboard
echo "  Services started: hci-collector, hci-dashboard"

# --- 7. Install nginx config ---
echo "[7/7] Configuring nginx..."
cp "$INSTALL_DIR/deploy/nginx.conf" /etc/nginx/sites-available/hci-toolkit
ln -sf /etc/nginx/sites-available/hci-toolkit /etc/nginx/sites-enabled/hci-toolkit

if nginx -t 2>/dev/null; then
    systemctl reload nginx
    echo "  nginx configured and reloaded"
else
    echo "  WARNING: nginx config test failed — check /etc/nginx/sites-available/hci-toolkit"
    echo "  Remember to replace YOUR_DOMAIN with your actual domain name"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit /etc/nginx/sites-available/hci-toolkit — replace YOUR_DOMAIN"
echo "  2. sudo nginx -t && sudo systemctl reload nginx"
echo "  3. sudo certbot --nginx -d YOUR_DOMAIN"
echo ""
echo "Service management:"
echo "  sudo systemctl status hci-collector hci-dashboard"
echo "  sudo journalctl -u hci-collector -f"
echo "  sudo journalctl -u hci-dashboard -f"
