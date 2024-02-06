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