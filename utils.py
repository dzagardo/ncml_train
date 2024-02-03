from google.cloud import secretmanager
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

def access_secret_version(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

encryption_secret_key = access_secret_version("privacytoolbox", "ENCRYPTION_SECRET_KEY")

def decrypt_token(encryption_secret_key, token):
    parts = token.split(':')
    iv = base64.b64decode(parts[0])
    encrypted_text = base64.b64decode(parts[1])

    cipher = Cipher(algorithms.AES(encryption_secret_key.encode()), modes.CTR(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(encrypted_text) + decryptor.finalize()