#!/bin/bash

# Deploy script with automatic HTTPS setup
set -e

echo "================================================"
echo "Fast Deploy - ECE 461 Phase 2 (with HTTPS)"
echo "================================================"

REPO_DIR="$HOME/117-Phase-2-CLI"

# Update repository
echo "[1/6] Updating repository..."
cd "$REPO_DIR"
git pull

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[2/6] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[2/6] Virtual environment exists"
fi

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "[4/6] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Generate SSL certificates if they don't exist
echo "[5/6] Setting up HTTPS..."
if [ ! -d "ssl_certs" ] || [ ! -f "ssl_certs/cert.pem" ] || [ ! -f "ssl_certs/key.pem" ]; then
    echo "  Generating SSL certificates..."

    # Create ssl_certs directory
    mkdir -p ssl_certs

    # Generate self-signed certificate
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout ssl_certs/key.pem \
        -out ssl_certs/cert.pem \
        -days 365 \
        -subj "/C=US/ST=Indiana/L=West Lafayette/O=Purdue University/OU=ECE 461/CN=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo 'localhost')" \
        2>/dev/null

    if [ -f "ssl_certs/cert.pem" ]; then
        echo "  ✓ SSL certificates generated successfully"
    else
        echo "  ✗ Failed to generate SSL certificates - will use HTTP"
    fi
else
    echo "  ✓ SSL certificates already exist"
fi

# Restart service
echo "[6/6] Restarting service..."
sudo systemctl restart registry-api

# Wait and check status
sleep 3

if sudo systemctl is-active --quiet registry-api; then
    INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "localhost")
    echo ""
    echo "Deployment complete!"
    echo ""

    # Check if HTTPS is enabled
    if [ -f "ssl_certs/cert.pem" ]; then
        echo "Server running with HTTPS at: https://$INSTANCE_IP:8000"
        echo "   (You may see a browser warning - this is normal for self-signed certificates)"
        echo ""
        echo "API Docs at: https://$INSTANCE_IP:8000/docs"
    else
        echo "Server running with HTTP at: http://$INSTANCE_IP:8000"
        echo ""
        echo "API Docs at: http://$INSTANCE_IP:8000/docs"
    fi

    echo ""
    echo "Check logs: sudo journalctl -u registry-api -f"
    echo "Reload page: Ctrl+Shift+R (hard refresh to clear cache)"
else
    echo "❌ Service failed to start"
    echo "Checking logs..."
    sudo journalctl -u registry-api -n 20
    exit 1
fi
