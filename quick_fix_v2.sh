#!/bin/bash

# Quick fix script with verbose error handling
set -e

echo "================================================"
echo "HTTPS Quick Fix Script (Verbose)"
echo "================================================"

REPO_DIR="$HOME/117-Phase-2-CLI"

echo "[1/7] Stopping any running service..."
sudo systemctl stop registry-api 2>/dev/null || true
sleep 2
echo "  ✓ Service stopped"

echo "[2/7] Ensuring SSL certificates exist..."
mkdir -p "$REPO_DIR/ssl_certs"

if [ ! -f "$REPO_DIR/ssl_certs/cert.pem" ] || [ ! -f "$REPO_DIR/ssl_certs/key.pem" ]; then
    echo "  Generating new SSL certificates..."

    # Get instance IP with timeout
    echo "  Getting instance IP..."
    INSTANCE_IP=$(timeout 5 curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '18.191.29.29')
    echo "  Using IP: $INSTANCE_IP"

    # Generate certificate with explicit output
    echo "  Running openssl..."
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$REPO_DIR/ssl_certs/key.pem" \
        -out "$REPO_DIR/ssl_certs/cert.pem" \
        -days 365 \
        -subj "/C=US/ST=Indiana/L=West Lafayette/O=Purdue/CN=$INSTANCE_IP"

    if [ $? -eq 0 ]; then
        chmod 600 "$REPO_DIR/ssl_certs/key.pem"
        chmod 644 "$REPO_DIR/ssl_certs/cert.pem"
        echo "  ✓ Certificates generated successfully"
    else
        echo "  ✗ ERROR: Failed to generate certificates"
        exit 1
    fi
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
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    import uvicorn

    script_dir = Path(__file__).parent.absolute()
    ssl_keyfile = script_dir / "ssl_certs" / "key.pem"
    ssl_certfile = script_dir / "ssl_certs" / "cert.pem"

    use_ssl = ssl_keyfile.exists() and ssl_certfile.exists()

    if use_ssl:
        logger.info("="*60)
        logger.info("🔒 STARTING SERVER WITH HTTPS")
        logger.info(f"SSL Key: {ssl_keyfile}")
        logger.info(f"SSL Cert: {ssl_certfile}")
        logger.info("Access at: https://0.0.0.0:8000")
        logger.info("="*60)

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
        logger.warning("="*60)
        logger.warning("⚠️ SSL CERTIFICATES NOT FOUND - USING HTTP")
        logger.info(f"Looking for: {ssl_certfile.parent}")
        logger.info("Access at: http://0.0.0.0:8000")
        logger.warning("="*60)

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
sleep 4

echo ""
echo "================================================"
if sudo systemctl is-active --quiet registry-api; then
    echo "✅ SUCCESS! Service is running"
    echo ""

    # Check logs for HTTPS confirmation
    echo "Checking logs for HTTPS..."
    sleep 2
    if sudo journalctl -u registry-api -n 30 --no-pager | grep -q "STARTING SERVER WITH HTTPS"; then
        echo ""
        echo "🔒 HTTPS IS ENABLED!"
        echo ""
        echo "   Your server: https://18.191.29.29:8000"
        echo "   API Docs: https://18.191.29.29:8000/docs"
        echo ""
        echo "   Browser warning is normal for self-signed certificates"
        echo "   Click 'Advanced' → 'Proceed' to continue"
    else
        echo ""
        echo "⚠️ WARNING: Server may not be using HTTPS"
        echo ""
        echo "Recent logs:"
        sudo journalctl -u registry-api -n 20 --no-pager
    fi

    echo ""
    echo "Test HTTPS: curl -k https://localhost:8000/health"
    echo "View logs: sudo journalctl -u registry-api -f"
else
    echo "❌ ERROR: Service failed to start"
    echo ""
    echo "Last 50 log lines:"
    sudo journalctl -u registry-api -n 50 --no-pager
    exit 1
fi
echo "================================================"
