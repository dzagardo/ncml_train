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
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
rm $HOME/miniconda.sh

# Update PATH immediately for the current session
export PATH="$HOME/miniconda/bin:$PATH"

echo "Initializing Conda for the current shell session..."
conda init bash

# Source the Conda configuration explicitly for the current session
source $HOME/.bashrc

echo "Creating a new Conda environment with Python 3.9..."
conda create -n myenv python=3.9 -y

echo "Activating the Conda environment directly..."
source $HOME/miniconda/bin/activate myenv

echo "Installing Hugging Face CLI..."
pip install huggingface-hub

echo "Ensuring dpkg is configured properly..."
sudo dpkg --configure -a

sudo apt-get update

echo "Installing dependencies..."
pip install google-cloud-secret-manager cryptography

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt
pip install peft==0.4.0
pip install peft

# Install necessary dependencies for secret retrieval and decryption
sudo pip3 install google-cloud-secret-manager cryptography

echo "Installing dependencies for secret retrieval and decryption..."

# Replace this with the appropriate command to fetch the encrypted token from your metadata or secret manager
ENCRYPTED_TOKEN=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/encryptedHFAccessToken" -H "Metadata-Flavor: Google")

echo "Retrieving and decrypting the Hugging Face Access Token..."
export HF_TOKEN=$(python3 -c "from utils import access_secret_version, decrypt_token; print(decrypt_token(access_secret_version('privacytoolbox', 'ENCRYPTION_SECRET_KEY'), '$ENCRYPTED_TOKEN'))")

echo "Logging in to the HuggingFace CLI..."
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
huggingface-cli login
huggingface-cli whoami

echo "Running the training script..."
python3 train.py

echo "Deploy script finished."
