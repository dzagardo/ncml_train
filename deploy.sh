#!/bin/bash

# Redirect stdout and stderr to a log file
LOG_FILE="/var/log/deploy-script.log"
exec > $LOG_FILE 2>&1

# Set environment variables for non-interactive mode
export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none

echo "Starting the deploy script..."

# Download and install Miniconda
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
export PATH="/opt/conda/bin:$PATH"

# Create a new Conda environment with Python 3.9
echo "Creating a new Conda environment with Python 3.9..."
conda create -n myenv python=3.9 -y

# Activate the environment
echo "Activating the Conda environment..."
source /opt/conda/bin/activate myenv

# Install Hugging Face CLI
echo "Installing Hugging Face CLI..."
pip install huggingface-hub

# Ensure system is up to date
echo "Updating system packages..."
sudo apt-get update && sudo apt-get -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade -y

# Install NVIDIA GPU driver
echo "Installing NVIDIA GPU driver..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository contrib
sudo add-apt-repository non-free
sudo apt-get update

# Automatically install the recommended driver
sudo ubuntu-drivers autoinstall || echo "Failed to install drivers automatically; check manually."

# Install CUDA Toolkit (if necessary)
echo "Installing CUDA Toolkit..."
# Specify the version you need
CUDA_VERSION="cuda-11-4"
sudo apt-get install -y $CUDA_VERSION

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
