#!/bin/bash

# Redirect stdout and stderr to a log file
LOG_FILE="/var/log/deploy-script.log"
exec > $LOG_FILE 2>&1

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

# Install necessary dependencies for secret retrieval and decryption
echo "Installing dependencies..."
pip install google-cloud-secret-manager cryptography

# Install requirements from requirements.txt
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Retrieve and decrypt the Hugging Face Access Token
# Using python3 explicitly as we are in the conda environment
echo "Retrieving and decrypting the Hugging Face Access Token..."
export HF_TOKEN=$(python3 -c 'from utils import access_secret_version, decrypt_token; print(decrypt_token(access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY"), "ENCRYPTED_TOKEN"))')

# Login to the HuggingFace CLI
echo "Logging in to the HuggingFace CLI..."
echo $HF_TOKEN | huggingface-cli login

# Run the training script with parameters
echo "Running the training script..."
python3 train.py

echo "Deploy script finished."
