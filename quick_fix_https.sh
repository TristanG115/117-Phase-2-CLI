#!/bin/bash

# Quick fix script to force HTTPS setup
set -e

echo "================================================"
echo "HTTPS Quick Fix Script"
echo "================================================"

REPO_DIR="$HOME/117-Phase-2-CLI"

echo "[1/7] Stopping any running service..."
sudo systemctl stop registry-api 2>/dev/null || true
sleep 2

echo "[2/7] Ensuring SSL certificates exist..."
mkdir -p "$REPO_DIR/ssl_certs"

if [ ! -f "$REPO_DIR/ssl_certs/cert.pem" ] || [ ! -f "$REPO_DIR/ssl_certs/key.pem" ]; then
    echo "  Generating new SSL certificates..."
    INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'localhost')

    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout "$REPO_DIR/ssl_certs/key.pem" \
        -out "$REPO_DIR/ssl_certs/cert.pem" \
        -days 365 \
        -subj "/C=US/ST=Indiana/L=West Lafayette/O=Purdue/CN=$INSTANCE_IP" \
        2>/dev/null

    chmod 600 "$REPO_DIR/ssl_certs/key.pem"
    chmod 644 "$REPO_DIR/ssl_certs/cert.pem"
    echo "  ✓ Certificates generated"
else
    echo "  ✓ Certificates already exist"
fi

echo "[3/7] Verifying certificate files..."
if [ -f "$REPO_DIR/ssl_certs/cert.pem" ] && [ -f "$REPO_DIR/ssl_certs/key.pem" ]; then
    echo "  ✓ Certificate files verified:"
    ls -lh "$REPO_DIR/ssl_certs/"
else
    echo "  ✗ ERROR: Certificate files not found!"
    exit 1
fi

echo "[4/7] Creating start_server.py..."
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
        logger.info("="*50)
        logger.info("🔒 Starting server with HTTPS...")
        logger.info(f"   SSL Key: {ssl_keyfile}")
        logger.info(f"   SSL Cert: {ssl_certfile}")
        logger.info("   Access at: https://0.0.0.0:8000")
        logger.info("="*50)

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
        logger.warning("="*50)
        logger.warning("⚠️  SSL certificates not found - starting with HTTP")
        logger.info(f"   Looking for: {ssl_certfile.parent}")
        logger.info("   Access at: http://0.0.0.0:8000")
        logger.warning("="*50)

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
echo "  ✓ start_server.py created"

echo "[5/7] Updating systemd service..."
sudo tee /etc/systemd/system/registry-api.service > /dev/null << EOF
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

echo "  ✓ Service file updated"

echo "[6/7] Reloading systemd and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable registry-api
echo "  ✓ Systemd reloaded"

echo "[7/7] Starting service..."
sudo systemctl start registry-api
sleep 3

echo ""
echo "================================================"
if sudo systemctl is-active --quiet registry-api; then
    echo "✅ SUCCESS! Service is running"
    echo ""

    # Check logs for HTTPS confirmation
    sleep 1
    if sudo journalctl -u registry-api -n 20 --no-pager | grep -q "Starting server with HTTPS"; then
        INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
        echo "🔒 HTTPS is ENABLED!"
        echo ""
        echo "   Access your server at: https://$INSTANCE_IP:8000"
        echo "   API Docs: https://$INSTANCE_IP:8000/docs"
        echo ""
        echo "   Note: Your browser will show a warning about the self-signed"
        echo "   certificate. Click 'Advanced' and 'Proceed' to continue."
    else
        echo "⚠️  WARNING: Server may not be using HTTPS"
        echo "   Check logs: sudo journalctl -u registry-api -n 50"
    fi

    echo ""
    echo "📋 View logs: sudo journalctl -u registry-api -f"
    echo "🔄 Restart: sudo systemctl restart registry-api"
else
    echo "❌ ERROR: Service failed to start"
    echo ""
    echo "Showing last 50 log lines:"
    sudo journalctl -u registry-api -n 50 --no-pager
    exit 1
fi
echo "================================================"
