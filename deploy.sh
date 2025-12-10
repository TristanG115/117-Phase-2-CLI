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

    # Generate self-signed certificate (with fallback IP)
    INSTANCE_IP=$(timeout 5 curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '18.191.29.29')

    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout ssl_certs/key.pem \
        -out ssl_certs/cert.pem \
        -days 365 \
        -subj "/C=US/ST=Indiana/L=WestLafayette/O=Purdue/CN=$INSTANCE_IP" \
        2>/dev/null

    if [ -f "ssl_certs/cert.pem" ]; then
        chmod 600 ssl_certs/key.pem
        chmod 644 ssl_certs/cert.pem
        echo "  ✓ SSL certificates generated successfully"
    else
        echo "  ✗ Failed to generate SSL certificates - will use HTTP"
    fi
else
    echo "  ✓ SSL certificates already exist"
fi

# Create/update start_server.py if it doesn't exist or is outdated
echo "[6/7] Ensuring start_server.py exists..."
if [ ! -f "$REPO_DIR/start_server.py" ]; then
    echo "  Creating start_server.py..."
    cat > "$REPO_DIR/start_server.py" << 'EOFPY'
#!/usr/bin/env python3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    import uvicorn
    script_dir = Path(__file__).parent.absolute()
    ssl_keyfile = script_dir / "ssl_certs" / "key.pem"
    ssl_certfile = script_dir / "ssl_certs" / "cert.pem"

    if ssl_keyfile.exists() and ssl_certfile.exists():
        logger.info("="*60)
        logger.info("STARTING SERVER WITH HTTPS")
        logger.info("="*60)
        uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1,
                    timeout_keep_alive=30, log_level="info",
                    ssl_keyfile=str(ssl_keyfile), ssl_certfile=str(ssl_certfile))
    else:
        logger.warning("SSL CERTIFICATES NOT FOUND - USING HTTP")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1,
                    timeout_keep_alive=30, log_level="info")

if __name__ == "__main__":
    main()
EOFPY
    chmod +x "$REPO_DIR/start_server.py"
    echo "  ✓ start_server.py created"
else
    echo "  ✓ start_server.py already exists"
fi

# Restart service
echo "[7/7] Restarting service..."
sudo systemctl restart registry-api

# Wait and check status
sleep 3

if sudo systemctl is-active --quiet registry-api; then
    INSTANCE_IP=$(timeout 5 curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    echo ""
    echo "Deployment complete!"
    echo ""

    # Check if HTTPS is enabled
    sleep 1
    if sudo journalctl -u registry-api -n 20 --no-pager | grep -q "STARTING SERVER WITH HTTPS"; then
        echo " Server running with HTTPS at: https://$INSTANCE_IP:8000"
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
    echo "Service failed to start"
    echo "Checking logs..."
    sudo journalctl -u registry-api -n 20
    exit 1
fi
