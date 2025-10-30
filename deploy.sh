#!/bin/bash

# deploy.sh - Automated deployment script for AWS EC2

set -e  # Exit on error

echo "================================================"
echo "ECE 461 Phase 2 - AWS EC2 Deployment"
echo "Instance: t2.micro (1 vCPU, 1GB RAM)"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run as root. Run as ec2-user or ubuntu."
    exit 1
fi

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    print_error "Cannot detect OS"
    exit 1
fi

echo "Step 1: System Updates"
echo "----------------------------------------"

if [ "$OS" = "amzn" ] || [ "$OS" = "amazon" ]; then
    print_info "Detected Amazon Linux"
    sudo yum update -y
    sudo yum install python3 python3-pip git gcc python3-devel -y
elif [ "$OS" = "ubuntu" ]; then
    print_info "Detected Ubuntu"
    sudo apt update
    sudo apt upgrade -y
    sudo apt install python3 python3-pip python3-venv git build-essential -y
else
    print_error "Unsupported OS: $OS"
    exit 1
fi

print_success "System packages updated"
echo ""

# Check Python version
echo "Step 2: Verify Python"
echo "----------------------------------------"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_info "Python version: $PYTHON_VERSION"

if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    print_error "Python 3.8+ required"
    exit 1
fi
print_success "Python version OK"
echo ""

# Setup swap space for t2.micro
echo "Step 3: Configure Swap (2GB)"
echo "----------------------------------------"
if [ ! -f /swapfile ]; then
    print_info "Creating swap file..."
    sudo dd if=/dev/zero of=/swapfile bs=1M count=2048 status=progress
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    # Make permanent
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    print_success "Swap space created"
else
    print_info "Swap already configured"
fi
free -h
echo ""

# Clone or update repository
echo "Step 4: Repository Setup"
echo "----------------------------------------"
REPO_DIR="$HOME/117-Phase-2-CLI"

if [ ! -d "$REPO_DIR" ]; then
    print_info "Cloning repository..."
    git clone https://github.com/TristanG115/117-Phase-2-CLI "$REPO_DIR"
    print_success "Repository cloned"
else
    print_info "Repository exists, pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    print_success "Repository updated"
fi

cd "$REPO_DIR"
echo ""

# Create virtual environment
echo "Step 5: Python Virtual Environment"
echo "----------------------------------------"
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Install dependencies
echo "Step 6: Install Dependencies"
echo "----------------------------------------"
print_info "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Dependencies installed"
echo ""

# Setup environment variables
echo "Step 7: Environment Configuration"
echo "----------------------------------------"

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    print_info "GITHUB_TOKEN not found in environment"
    read -p "Enter your GitHub token (or press Enter to skip): " github_token
    if [ -n "$github_token" ]; then
        export GITHUB_TOKEN="$github_token"
        echo "export GITHUB_TOKEN=\"$github_token\"" >> ~/.bashrc
        print_success "GitHub token configured"
    else
        print_info "Skipping GitHub token setup (some features may not work)"
    fi
else
    print_success "GitHub token found"
fi

# Setup log file
export LOG_FILE="$REPO_DIR/server.log"
export LOG_LEVEL="1"

# Add to bashrc if not already there
if ! grep -q 'LOG_FILE' ~/.bashrc; then
    echo "export LOG_FILE=\"$REPO_DIR/server.log\"" >> ~/.bashrc
    echo "export LOG_LEVEL=\"1\"" >> ~/.bashrc
fi

touch "$LOG_FILE"
chmod 666 "$LOG_FILE"
print_success "Log file configured: $LOG_FILE"
echo ""

# Initialize registry
echo "Step 8: Initialize Registry Database"
echo "----------------------------------------"
python3 -c "from handlers import registry_handler; registry_handler.init_registry()" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Registry initialized"
else
    print_error "Failed to initialize registry"
    exit 1
fi
echo ""

# Setup systemd service
echo "Step 9: Configure Systemd Service"
echo "----------------------------------------"

SERVICE_FILE="/etc/systemd/system/registry-api.service"

# Update service file with actual token
TEMP_SERVICE=$(mktemp)
cat > "$TEMP_SERVICE" << EOF
[Unit]
Description=ECE 461 Trustworthy Model Registry API Server
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$REPO_DIR
Environment="PATH=$REPO_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$REPO_DIR"
Environment="GITHUB_TOKEN=${GITHUB_TOKEN:-your_token_here}"
Environment="LOG_FILE=$REPO_DIR/server.log"
Environment="LOG_LEVEL=1"
ExecStart=$REPO_DIR/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 10
Restart=always
RestartSec=10
StandardOutput=append:$REPO_DIR/server.log
StandardError=append:$REPO_DIR/server.log

[Install]
WantedBy=multi-user.target
EOF

sudo cp "$TEMP_SERVICE" "$SERVICE_FILE"
rm "$TEMP_SERVICE"

sudo systemctl daemon-reload
sudo systemctl enable registry-api
print_success "Systemd service configured"
echo ""

# Start service
echo "Step 10: Start Service"
echo "----------------------------------------"
sudo systemctl start registry-api

# Wait a moment for service to start
sleep 3

if sudo systemctl is-active --quiet registry-api; then
    print_success "Service started successfully"
else
    print_error "Service failed to start"
    print_info "Checking logs..."
    sudo journalctl -u registry-api -n 20
    exit 1
fi
echo ""

# Test endpoint
echo "Step 11: Test API"
echo "----------------------------------------"
print_info "Testing /tracks endpoint..."
sleep 2
RESPONSE=$(curl -s http://localhost:8000/tracks || echo "failed")

if [[ "$RESPONSE" == *"Reliability"* ]]; then
    print_success "API is responding correctly"
else
    print_error "API test failed"
    print_info "Response: $RESPONSE"
fi
echo ""

# Get instance info
INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "unknown")

echo "================================================"
echo "Deployment Complete!"
echo "================================================"
echo ""
echo "Your API server is running at:"
echo "  • Dashboard: http://$INSTANCE_IP:8000"
echo "  • API Docs:  http://$INSTANCE_IP:8000/docs"
echo "  • API Base:  http://$INSTANCE_IP:8000"
echo ""
echo "Service Management:"
echo "  • Status:  sudo systemctl status registry-api"
echo "  • Stop:    sudo systemctl stop registry-api"
echo "  • Start:   sudo systemctl start registry-api"
echo "  • Restart: sudo systemctl restart registry-api"
echo "  • Logs:    sudo journalctl -u registry-api -f"
echo ""
echo "Next Steps:"
echo "  1. Update security group to allow port 8000"
echo "  2. Test API endpoints using the docs page"
echo "  3. Configure domain name (optional)"
echo "  4. Setup Nginx reverse proxy (optional)"
echo ""
print_success "Deployment successful!"
