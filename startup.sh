#!/bin/bash

# Redirect stdout and stderr to a log file for debugging
LOG_FILE="/var/log/startup-script.log"
sudo touch $LOG_FILE
sudo chown $USER $LOG_FILE
exec > $LOG_FILE 2>&1

echo "======================================"
echo "Initializing the Deep Learning VM setup"
echo "======================================"

# Install Docker if not already installed
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
fi

# Configure Docker to start on boot
sudo systemctl enable docker

# Install NVIDIA Docker if not already installed
if ! command -v nvidia-docker &>/dev/null; then
    echo "Setting up NVIDIA Docker repository..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    echo "Installing NVIDIA Docker runtime..."
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
fi

# Restart Docker to apply NVIDIA runtime
sudo systemctl restart docker

# Configure Docker to use NVIDIA as the default runtime
echo "Configuring Docker to use NVIDIA as the default runtime..."
sudo mkdir -p /etc/docker
echo '{ "default-runtime": "nvidia", "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] }}}' | sudo tee /etc/docker/daemon.json
sudo systemctl daemon-reload
sudo systemctl restart docker

# Verify NVIDIA Docker installation
echo "Verifying NVIDIA Docker installation..."
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

echo "======================================"
echo "Deep Learning VM setup completed"
echo "======================================"

# Clone the repository and run the startup script
echo "Cloning the project repository..."
git clone https://github.com/dzagardo/ncml_train.git
cd ncml_train

echo "Running the project's startup script..."
./startup.sh
