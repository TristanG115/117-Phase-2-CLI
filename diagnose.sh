#!/bin/bash

echo "================================================"
echo "HTTPS Diagnostic Script"
echo "================================================"
echo ""

REPO_DIR="$HOME/117-Phase-2-CLI"

# Check 1: Service Status
echo "1. Checking service status..."
if sudo systemctl is-active --quiet registry-api; then
    echo "   ✓ Service is running"
else
    echo "   ✗ Service is NOT running"
fi
echo ""

# Check 2: SSL Certificates
echo "2. Checking SSL certificates..."
if [ -f "$REPO_DIR/ssl_certs/cert.pem" ] && [ -f "$REPO_DIR/ssl_certs/key.pem" ]; then
    echo "   ✓ Certificates exist:"
    echo "     - $REPO_DIR/ssl_certs/cert.pem"
    echo "     - $REPO_DIR/ssl_certs/key.pem"
    ls -lh "$REPO_DIR/ssl_certs/"
else
    echo "   ✗ Certificates NOT found in $REPO_DIR/ssl_certs/"
fi
echo ""

# Check 3: What's listening on port 8000?
echo "3. Checking what's listening on port 8000..."
PORT_CHECK=$(sudo lsof -i :8000 2>/dev/null)
if [ -n "$PORT_CHECK" ]; then
    echo "$PORT_CHECK"
else
    echo "   ✗ Nothing is listening on port 8000"
fi
echo ""

# Check 4: Test HTTP vs HTTPS
echo "4. Testing connections..."
echo "   HTTP test:"
HTTP_RESULT=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>&1)
echo "     HTTP status: $HTTP_RESULT"

echo "   HTTPS test:"
HTTPS_RESULT=$(curl -k -s -o /dev/null -w "%{http_code}" https://localhost:8000/health 2>&1)
echo "     HTTPS status: $HTTPS_RESULT"
echo ""

# Check 5: Recent logs
echo "5. Recent service logs (last 30 lines)..."
echo "   Looking for SSL/HTTPS indicators..."
sudo journalctl -u registry-api -n 30 --no-pager
echo ""

# Check 6: Service configuration
echo "6. Current systemd service configuration..."
if [ -f "/etc/systemd/system/registry-api.service" ]; then
    echo "   Service file exists:"
    cat /etc/systemd/system/registry-api.service
else
    echo "   ✗ Service file NOT found"
fi
echo ""

# Check 7: Start script
echo "7. Checking start_server.py..."
if [ -f "$REPO_DIR/start_server.py" ]; then
    echo "   ✓ start_server.py exists"
else
    echo "   ✗ start_server.py NOT found"
fi
echo ""

echo "================================================"
echo "DIAGNOSIS COMPLETE"
echo "================================================"
