#!/bin/bash

# Redirect stdout and stderr to a log file for debugging
LOG_FILE="/var/log/startup.log"
sudo touch $LOG_FILE
sudo chown $USER $LOG_FILE
exec > $LOG_FILE 2>&1

echo "======================================"
echo "Initializing the Deep Learning VM setup"
echo "======================================"

echo "======================================"
echo "Deep Learning VM setup completed"
echo "======================================"

echo "Cloning the project repository..."
git clone https://github.com/dzagardo/ncml_train.git
cd ncml_train

# Ensure deploy.sh is executable
echo "Making deploy.sh script executable..."
chmod +x deploy.sh

echo "Running the project's startup script..."
./deploy.sh || { echo "Execution of deploy.sh failed"; exit 1; }
