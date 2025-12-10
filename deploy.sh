#!/bin/bash

# Deploy script with automatic HTTPS setup
set -e

echo "================================================"
echo "Fast Deploy - ECE 461 Phase 2 (with HTTPS)"
echo "================================================"

REPO_DIR="$HOME/117-Phase-2-CLI"

# Update repository
echo "[1/7] Updating repository..."
cd "$REPO_DIR"
git pull

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[2/7] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[2/7] Virtual environment exists"
fi

# Activate virtual environment
echo "[3/7] Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "[4/7] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Generate SSL certificates if they don't exist
echo "[5/7] Setting up HTTPS..."
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
        # Set proper permissions
        chmod 600 ssl_certs/key.pem
        chmod 644 ssl_certs/cert.pem
    else
        echo "  ✗ Failed to generate SSL certificates - will use HTTP"
    fi
else
    echo "  ✓ SSL certificates already exist"
fi

# Update systemd service configuration
echo "[6/7] Updating systemd service..."
cat > /tmp/registry-api.service << 'EOF'
[Unit]
Description=ECE 461 Registry API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
Environment="PATH=$REPO_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$REPO_DIR/venv/bin/python3 $REPO_DIR/start_server.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Replace placeholders with actual values
sed -i "s|\$USER|$USER|g" /tmp/registry-api.service
sed -i "s|\$REPO_DIR|$REPO_DIR|g" /tmp/registry-api.service

# Install the service file
sudo cp /tmp/registry-api.service /etc/systemd/system/registry-api.service
sudo systemctl daemon-reload

# Create start_server.py if it doesn't exist
if [ ! -f "$REPO_DIR/start_server.py" ]; then
    echo "  Creating start_server.py..."
    cat > "$REPO_DIR/start_server.py" << 'EOFPY'
#!/usr/bin/env python3
"""
Startup script for the registry API server with HTTPS support.
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    import uvicorn

    script_dir = Path(__file__).parent.absolute()
    ssl_keyfile = script_dir / "ssl_certs" / "key.pem"
    ssl_certfile = script_dir / "ssl_certs" / "cert.pem"

    use_ssl = ssl_keyfile.exists() and ssl_certfile.exists()

    if use_ssl:
        logger.info(" Starting server with HTTPS...")
        logger.info(f"   SSL Key: {ssl_keyfile}")
        logger.info(f"   SSL Cert: {ssl_certfile}")
        logger.info("   Access at: https://0.0.0.0:8000")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            workers=1,
            timeout_keep_alive=30,
            log_level="info",
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
        )
    else:
        logger.warning("  SSL certificates not found - starting with HTTP")
        logger.info(f"   Looking for: {ssl_certfile.parent}")
        logger.info("   Access at: http://0.0.0.0:8000")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            workers=1,
            timeout_keep_alive=30,
            log_level="info",
        )

if __name__ == "__main__":
    main()
EOFPY
    chmod +x "$REPO_DIR/start_server.py"
fi

# Restart service
echo "[7/7] Restarting service..."
sudo systemctl enable registry-api
sudo systemctl restart registry-api

# Wait and check status
sleep 3

if sudo systemctl is-active --quiet registry-api; then
    INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "localhost")
    echo ""
    echo " Deployment complete!"
    echo ""

    # Check if HTTPS is enabled by looking at the logs
    sleep 1
    if sudo journalctl -u registry-api -n 20 | grep -q "Starting server with HTTPS"; then
        echo " Server running with HTTPS at: https://$INSTANCE_IP:8000"
        echo "   (You may see a browser warning - this is normal for self-signed certificates)"
        echo ""
        echo " API Docs at: https://$INSTANCE_IP:8000/docs"
    else
        echo "  Server running with HTTP at: http://$INSTANCE_IP:8000"
        echo ""
        echo " API Docs at: http://$INSTANCE_IP:8000/docs"
        echo ""
        echo "Note: To enable HTTPS, ensure SSL certificates exist in $REPO_DIR/ssl_certs/"
    fi

    echo ""
    echo " Check logs: sudo journalctl -u registry-api -f"
    echo " Reload page: Ctrl+Shift+R (hard refresh to clear cache)"
else
    echo " Service failed to start"
    echo "Checking logs..."
    sudo journalctl -u registry-api -n 50
    exit 1
fi
