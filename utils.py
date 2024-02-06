from google.cloud import secretmanager
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import binascii
from transformers import HfApi
import requests

def fetch_encrypted_token():
    """Fetch the encrypted Hugging Face Access Token from GCE metadata."""
    url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/encryptedHFAccessToken"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch encrypted token: {response.text}")

def access_secret_version(project_id, secret_id):
    print("here")
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    print(response)
    return response.payload.data.decode("UTF-8")

encryption_secret_key = access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY")

def decrypt_token(encryption_secret_key, token):
    parts = token.split(':')
    print("IV part (before decoding):", parts[0])
    print("Encrypted text part (before decoding):", parts[1])
    iv = bytes.fromhex(parts[0])  # Decode IV from hex
    encrypted_text = bytes.fromhex(parts[1])  # Decode encrypted text from hex

    cipher = Cipher(algorithms.AES(bytes.fromhex(encryption_secret_key)), modes.CTR(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return (decryptor.update(encrypted_text) + decryptor.finalize()).decode('utf-8')

def set_hf_authentication(project_id, secret_id):
    """Fetch, decrypt, and use the Hugging Face Access Token for authentication."""
    encrypted_token = fetch_encrypted_token()
    encryption_secret_key = access_secret_version(project_id, secret_id)
    decrypted_token = decrypt_token(encryption_secret_key, encrypted_token)
    HfApi().set_access_token(decrypted_token)
