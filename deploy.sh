#!/bin/bash

# Redirect stdout and stderr to a log file
LOG_FILE="/var/log/deploy-script.log"
exec > $LOG_FILE 2>&1

echo "Starting the deploy script..."

# Install necessary dependencies for secret retrieval and decryption
pip install google-cloud-secret-manager cryptography

# Install requirements
pip install -r requirements.txt

# Retrieve and decrypt the Hugging Face Access Token
export HF_TOKEN=$(python -c 'from utils import access_secret_version, decrypt_token; print(decrypt_token(access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY"), "ENCRYPTED_TOKEN"))')

# Login to the HuggingFace CLI
echo $HF_TOKEN | huggingface-cli login

# Run the training script with parameters
python train.py

# Shut down the instance after the script completes
sudo shutdown -h now

echo "Deploy script finished."
