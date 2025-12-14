#!/bin/bash

# Deploy script for autograder compatibility (HTTP only)
set -e

echo "================================================"
echo "Fast Deploy - ECE 461 Phase 2 (HTTP for autograder)"
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

# Create/update start_server.py for HTTP
echo "[5/5] Ensuring start_server.py uses HTTP..."
cat > "$REPO_DIR/start_server.py" << 'EOFPY'
#!/usr/bin/env python3
"""
Startup script for the registry API server.
Uses HTTP on port 8000 for autograder compatibility.
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    import uvicorn

    logger.info("="*60)
    logger.info("STARTING SERVER WITH HTTP (for autograder)")
    logger.info("Access at: http://0.0.0.0:8000")
    logger.info("="*60)

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

# Restart service
echo ""
echo "Restarting service..."
sudo systemctl restart registry-api

# Wait and check status
sleep 3

if sudo systemctl is-active --quiet registry-api; then
    INSTANCE_IP=$(timeout 5 curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "18.191.29.29")
    echo ""
    echo " eployment complete!"
    echo ""
    echo "Server running with HTTP at: http://$INSTANCE_IP:8000"
    echo "API Docs at: http://$INSTANCE_IP:8000/docs"
    echo "Check logs: sudo journalctl -u registry-api -f"
else
    echo " Service failed to start"
    echo "Checking logs..."
    sudo journalctl -u registry-api -n 20
    exit 1
fi
