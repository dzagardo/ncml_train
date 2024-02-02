#!/bin/bash

# Redirect stdout and stderr to a log file
LOG_FILE="/var/log/deploy-script.log"
exec > $LOG_FILE 2>&1

# Set environment variables for non-interactive mode
export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none

echo "Starting the deploy script..."

# Download and install Miniconda in silent mode
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
export PATH="/opt/conda/bin:$PATH"

# Create a new Conda environment with Python 3.9
echo "Creating a new Conda environment with Python 3.9..."
sudo conda create -n myenv2 python=3.9 -y

# Activate the environment
echo "Activating the Conda environment..."
source /opt/conda/bin/activate myenv

# Install Hugging Face CLI
echo "Installing Hugging Face CLI..."
pip install huggingface-hub

# Ensure dpkg is in a good state
echo "Ensuring dpkg is configured properly..."
sudo dpkg --configure -a

# Add NVIDIA repository and key
echo "Adding NVIDIA repository..."
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Update the package lists
sudo apt-get update

# Install the NVIDIA driver using the package manager
echo "Installing the latest NVIDIA driver..."
sudo apt-get install -y build-essential

# Install CUDA Toolkit
echo "Installing CUDA Toolkit..."
CUDA_VERSION="cuda-toolkit-11-4" # Adjust the package name as necessary
sudo apt-get install -y $CUDA_VERSION

# Additional step: Install cuDNN (if necessary for deep learning frameworks)
echo "Installing cuDNN..."
sudo apt-get install -y libcudnn8

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version

# Install necessary dependencies for secret retrieval and decryption
echo "Installing dependencies..."
pip install google-cloud-secret-manager cryptography

# Install requirements from requirements.txt
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Retrieve and decrypt the Hugging Face Access Token
echo "Retrieving and decrypting the Hugging Face Access Token..."
export HF_TOKEN=$(python3 -c 'from utils import access_secret_version, decrypt_token; print(decrypt_token(access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY"), "ENCRYPTED_TOKEN"))')

# Login to the HuggingFace CLI
echo "Logging in to the HuggingFace CLI..."
echo $HF_TOKEN | huggingface-cli login

# Run the training script with parameters
echo "Running the training script..."
python3 train.py

echo "Deploy script finished."
