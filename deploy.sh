#!/bin/bash

# Redirect stdout and stderr to a log file
LOG_FILE="/var/log/deploy-script.log"
exec > $LOG_FILE 2>&1

echo "Starting the deploy script..."

# Check if NVIDIA drivers are installed by attempting to run nvidia-smi
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi; then
    echo "NVIDIA drivers not found. Please ensure the startup script has run successfully before proceeding."
    exit 1
else
    echo "NVIDIA drivers found. Proceeding with the rest of the deploy script."
fi

# Set environment variables for non-interactive mode
export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none

echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
export PATH="/opt/conda/bin:$PATH"

echo "Creating a new Conda environment with Python 3.9..."
sudo conda create -n myenv2 python=3.9 -y

echo "Activating the Conda environment..."
source /opt/conda/bin/activate myenv

echo "Installing Hugging Face CLI..."
pip install huggingface-hub

echo "Ensuring dpkg is configured properly..."
sudo dpkg --configure -a

echo "Adding NVIDIA repository..."
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

sudo apt-get update

echo "Assuming NVIDIA drivers are handled by the startup script. Skipping driver installation here."

echo "Installing CUDA Toolkit..."
CUDA_VERSION="cuda-toolkit-11-4" # Adjust the package name as necessary
sudo apt-get install -y $CUDA_VERSION

echo "Installing cuDNN..."
sudo apt-get install -y libcudnn8

echo "Verifying CUDA installation..."
nvcc --version

echo "Installing dependencies..."
pip install google-cloud-secret-manager cryptography

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Retrieving and decrypting the Hugging Face Access Token..."
export HF_TOKEN=$(python3 -c 'from utils import access_secret_version, decrypt_token; print(decrypt_token(access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY"), "ENCRYPTED_TOKEN"))')

echo "Logging in to the HuggingFace CLI..."
echo $HF_TOKEN | huggingface-cli login

echo "Running the training script..."
python3 train.py

echo "Deploy script finished."
