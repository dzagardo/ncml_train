#!/bin/bash

# Redirect stdout and stderr to a log file
LOG_FILE="/var/log/deploy-script.log"
exec > $LOG_FILE 2>&1

echo "Starting the deploy script..."

# Ensure Python3 and pip3 are installed (Debian/Ubuntu specific)
# This step would ideally ensure Python 3.9 is installed, but requires custom handling as mentioned
apt-get update && apt-get install -y python3 python3-pip

# Install Hugging Face CLI
pip3 install huggingface-hub

# Install necessary dependencies for secret retrieval and decryption
pip3 install google-cloud-secret-manager cryptography

# Install requirements from requirements.txt
# Assumes requirements.txt is compatible with the installed Python version
pip3 install -r requirements.txt

# Retrieve and decrypt the Hugging Face Access Token
# Make sure to use python3 explicitly
export HF_TOKEN=$(python3 -c 'from utils import access_secret_version, decrypt_token; print(decrypt_token(access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY"), "ENCRYPTED_TOKEN"))')

# Login to the HuggingFace CLI
echo $HF_TOKEN | huggingface-cli login

# Run the training script with parameters
# Again, ensure python3 is used
python3 train.py

echo "Deploy script finished."
