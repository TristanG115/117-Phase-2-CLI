#!/bin/bash

# deploy.sh - Automated deployment script for AWS EC2 with DynamoDB Integration

set -e  # Exit on error

echo "================================================"
echo "ECE 461 Phase 2 - AWS EC2 Deployment"
echo "with DynamoDB + S3 Integration"
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

# Install dependencies (including boto3)
echo "Step 6: Install Dependencies"
echo "----------------------------------------"
print_info "Installing Python packages..."
pip install --upgrade pip

# Add boto3 if not in requirements.txt
if ! grep -q "boto3" requirements.txt; then
    echo "boto3>=1.34.0" >> requirements.txt
    print_info "Added boto3 to requirements.txt"
fi

pip install -r requirements.txt
print_success "Dependencies installed (including boto3)"
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

# AWS Configuration for DynamoDB + S3
echo "Step 8: AWS Configuration (DynamoDB + S3)"
echo "----------------------------------------"

# Create .aws directory if it doesn't exist
mkdir -p .aws

# Check if config.json exists
if [ ! -f .aws/config.json ]; then
    print_info "Creating AWS config file..."
    
    # Prompt for AWS region
    read -p "Enter AWS region [us-east-2]: " aws_region
    aws_region=${aws_region:-us-east-2}
    
    # Prompt for S3 bucket name
    read -p "Enter S3 bucket name: " s3_bucket
    
    if [ -z "$s3_bucket" ]; then
        print_error "S3 bucket name is required!"
        exit 1
    fi
    
    # Create config file
    cat > .aws/config.json << EOF
{
    "AWS_REGION": "$aws_region",
    "S3_BUCKET_NAME": "$s3_bucket"
}
EOF
    print_success "AWS config created"
else
    print_info "AWS config already exists"
fi

# Display config
print_info "Current AWS Configuration:"
cat .aws/config.json

# Add .aws to .gitignore
if ! grep -q ".aws/" .gitignore 2>/dev/null; then
    echo ".aws/" >> .gitignore
    print_success "Added .aws/ to .gitignore"
fi

echo ""

# Verify AWS Credentials (IAM Role)
echo "Step 9: Verify AWS Credentials"
echo "----------------------------------------"
print_info "Checking IAM role..."

python3 << 'PYEOF'
import boto3
try:
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    print(f"✓ AWS credentials working!")
    print(f"  Account: {identity['Account']}")
    print(f"  Role: {identity['Arn'].split('/')[-1]}")
except Exception as e:
    print(f"✗ AWS credentials failed: {e}")
    print("\nIMPORTANT: Make sure IAM role is attached to EC2 instance!")
    print("  1. Go to AWS Console → EC2 → Instances")
    print("  2. Select instance → Actions → Security → Modify IAM role")
    print("  3. Attach role with DynamoDB and S3 permissions")
    print("  4. Reboot instance")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    print_error "AWS credentials check failed"
    exit 1
fi
echo ""

# Verify DynamoDB Table
echo "Step 10: Verify DynamoDB Table"
echo "----------------------------------------"
print_info "Checking DynamoDB table..."

python3 << 'PYEOF'
import boto3
import json

# Load region from config
with open('.aws/config.json', 'r') as f:
    config = json.load(f)
    region = config['AWS_REGION']

try:
    dynamodb = boto3.client('dynamodb', region_name=region)
    response = dynamodb.describe_table(TableName='TrustworthyModelRegDB')
    print(f"✓ DynamoDB table found!")
    print(f"  Status: {response['Table']['TableStatus']}")
    print(f"  Items: {response['Table']['ItemCount']}")
except Exception as e:
    print(f"✗ DynamoDB table not found: {e}")
    print("\nIMPORTANT: Create DynamoDB table first!")
    print("  1. Go to AWS Console → DynamoDB → Create table")
    print("  2. Table name: TrustworthyModelRegDB")
    print("  3. Partition key: artifact_id (String)")
    print("  4. Add indexes: artifact_type-index, name-index")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    print_error "DynamoDB table check failed"
    exit 1
fi
echo ""

# Verify S3 Bucket
echo "Step 11: Verify S3 Bucket"
echo "----------------------------------------"
print_info "Checking S3 bucket..."

python3 << 'PYEOF'
import boto3
import json

# Load config
with open('.aws/config.json', 'r') as f:
    config = json.load(f)
    region = config['AWS_REGION']
    bucket = config['S3_BUCKET_NAME']

try:
    s3 = boto3.client('s3', region_name=region)
    s3.head_bucket(Bucket=bucket)
    print(f"✓ S3 bucket '{bucket}' found!")
except Exception as e:
    print(f"✗ S3 bucket error: {e}")
    print(f"\nWARNING: S3 bucket '{bucket}' not accessible")
    print("  Create bucket in AWS Console if needed")
    print("  Continuing anyway (S3 is optional)...")
PYEOF

echo ""

# Initialize registry with DynamoDB
echo "Step 12: Initialize Registry (DynamoDB)"
echo "----------------------------------------"
print_info "Initializing DynamoDB registry..."

python3 << 'PYEOF'
from handlers import registry_handler

try:
    registry_handler.init_registry()
    print("✓ Registry initialized")
    
    # Health check
    health = registry_handler.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Backend: {health['backend']}")
    print(f"  Total artifacts: {health['total_artifacts']}")
    
    if health['status'] != 'healthy':
        print(f"✗ Health check failed: {health.get('error')}")
        exit(1)
except Exception as e:
    print(f"✗ Failed to initialize registry: {e}")
    exit(1)
PYEOF

if [ $? -eq 0 ]; then
    print_success "Registry initialized with DynamoDB backend"
else
    print_error "Failed to initialize registry"
    exit 1
fi
echo ""

# Run integration tests
echo "Step 13: Integration Tests"
echo "----------------------------------------"
print_info "Running integration tests..."

python3 << 'PYEOF'
from handlers import registry_handler

try:
    # Test add/retrieve/delete
    artifact_id = registry_handler.add_artifact(
        name="deployment-test",
        artifact_type="model",
        score=0.9,
        url="https://test.com"
    )
    print(f"✓ Add artifact: {artifact_id}")
    
    artifact = registry_handler.get_artifact_by_id(artifact_id)
    if artifact:
        print(f"✓ Retrieve artifact: {artifact['name']}")
    else:
        print("✗ Failed to retrieve artifact")
        exit(1)
    
    registry_handler.delete_artifact(artifact_id)
    print("✓ Delete artifact")
    
    print("\n✓ All integration tests passed!")
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    print_error "Integration tests failed"
    exit 1
fi
echo ""

# Setup systemd service
echo "Step 14: Configure Systemd Service"
echo "----------------------------------------"

SERVICE_FILE="/etc/systemd/system/registry-api.service"

# Update service file with actual token
TEMP_SERVICE=$(mktemp)
cat > "$TEMP_SERVICE" << EOF
[Unit]
Description=ECE 461 Trustworthy Model Registry API Server (DynamoDB)
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
echo "Step 15: Start Service"
echo "----------------------------------------"
sudo systemctl restart registry-api

# Wait a moment for service to start
sleep 5

if sudo systemctl is-active --quiet registry-api; then
    print_success "Service started successfully"
else
    print_error "Service failed to start"
    print_info "Checking logs..."
    sudo journalctl -u registry-api -n 30
    exit 1
fi
echo ""

# Test endpoints
echo "Step 16: Test API Endpoints"
echo "----------------------------------------"
print_info "Testing /tracks endpoint..."
sleep 2
RESPONSE=$(curl -s http://localhost:8000/tracks || echo "failed")

if [[ "$RESPONSE" == *"Reliability"* ]]; then
    print_success "/tracks endpoint working"
else
    print_error "API test failed"
    print_info "Response: $RESPONSE"
fi

print_info "Testing /artifacts/model endpoint..."
RESPONSE=$(curl -s http://localhost:8000/artifacts/model || echo "failed")

if [[ "$RESPONSE" == *"["* ]]; then
    print_success "/artifacts/model endpoint working"
else
    print_error "/artifacts/model test failed"
fi
echo ""

# Get instance info
INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "unknown")

echo "================================================"
echo "Deployment Complete!"
echo "================================================"
echo ""
echo "✓ DynamoDB Backend: Active"
echo "✓ S3 Integration: Configured"
echo "✓ Server: Running"
echo ""
echo "Your API server is running at:"
echo "  • Dashboard: http://$INSTANCE_IP:8000"
echo "  • API Base:  http://$INSTANCE_IP:8000"
echo "  • Docs:      http://$INSTANCE_IP:8000/docs"
echo ""
echo "Storage Backend:"
echo "  • Database: AWS DynamoDB (TrustworthyModelRegDB)"
echo "  • Files: AWS S3"
echo "  • Region: $(cat .aws/config.json | grep AWS_REGION | cut -d'"' -f4)"
echo ""
echo "Service Management:"
echo "  • Status:  sudo systemctl status registry-api"
echo "  • Stop:    sudo systemctl stop registry-api"
echo "  • Start:   sudo systemctl start registry-api"
echo "  • Restart: sudo systemctl restart registry-api"
echo "  • Logs:    sudo journalctl -u registry-api -f"
echo ""
echo "Health Check:"
echo "  curl http://localhost:8000/tracks"
echo "  curl http://localhost:8000/artifacts/model"
echo ""
print_success "Deployment successful with DynamoDB + S3!"
echo ""
print_info "Ready for autograder testing!"
