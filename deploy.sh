#!/bin/bash

# Fast deploy script - streamlined for quick updates
set -e

echo "================================================"
echo "Fast Deploy - ECE 461 Phase 2"
echo "================================================"

REPO_DIR="$HOME/117-Phase-2-CLI"

# Update repository
echo "[1/5] Updating repository..."
cd "$REPO_DIR"
git pull

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[2/5] Virtual environment exists"
fi

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "[4/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Restart service
echo "[5/5] Restarting service..."
sudo systemctl restart registry-api

# Wait and check status
sleep 3

if sudo systemctl is-active --quiet registry-api; then
    INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "localhost")
    echo ""
    echo "✓ Deployment complete!"
    echo ""
    echo "Server running at: http://$INSTANCE_IP:8000"
    echo "Docs at: http://$INSTANCE_IP:8000/docs"
    echo ""
    echo "Check logs: sudo journalctl -u registry-api -f"
else
    echo "✗ Service failed to start"
    echo "Checking logs..."
    sudo journalctl -u registry-api -n 20
    exit 1
fi
