#!/bin/bash

# Update system
echo "Updating system..."
sudo apt update && sudo apt full-upgrade -y

# Install Python venv
echo "Installing python3-venv..."
sudo apt install -y python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate venv and install requirements
echo "Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x install.sh run.sh

echo "Installation complete. Run ./run.sh to start."