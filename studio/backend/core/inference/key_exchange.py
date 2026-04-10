# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
RSA + AES-GCM hybrid encryption for transmitting the Access Endpoint
API key to the Studio frontend over plain HTTP.

Flow (server -> client):
    1. Server generates an RSA-2048 key pair at startup (in memory).
    2. Frontend fetches the public key via GET /access-endpoint/public-key.
    3. Frontend generates a fresh AES-256 session key per reveal.
    4. Frontend RSA-OAEP-encrypts the session key with the server's public key
       and POSTs it to /access-endpoint/reveal.
    5. Server decrypts the session key with its private key, then AES-GCM
       encrypts the API key payload and returns {iv, ciphertext}.
    6. Frontend AES-GCM-decrypts the payload locally using the session key it
       generated.

The key pair is regenerated on every restart — there is no long-lived secret
material on disk. The only plaintext bearer token ever hitting the wire in
either direction is protected by RSA-OAEP-SHA256 + AES-256-GCM.
"""

import base64
import logging
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_private_key: rsa.RSAPrivateKey | None = None
_public_key_pem: str | None = None


def init_key_pair() -> None:
    """Generate an RSA-2048 key pair. Called once at server startup."""
    global _private_key, _public_key_pem
    _private_key = rsa.generate_private_key(
        public_exponent = 65537,
        key_size = 2048,
    )
    _public_key_pem = (
        _private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    logger.info("RSA key pair generated for access endpoint reveal")


def get_public_key_pem() -> str:
    """Return the PEM-encoded public key for the frontend."""
    if _public_key_pem is None:
        raise RuntimeError("Key pair not initialized. Call init_key_pair() first.")
    return _public_key_pem


def _decrypt_session_key(encrypted_session_key_b64: str) -> bytes:
    """Decrypt a client-supplied AES session key with the server private key."""
    if _private_key is None:
        raise RuntimeError("Key pair not initialized. Call init_key_pair() first.")
    ciphertext = base64.b64decode(encrypted_session_key_b64)
    session_key = _private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf = padding.MGF1(algorithm = hashes.SHA256()),
            algorithm = hashes.SHA256(),
            label = None,
        ),
    )
    if len(session_key) not in (16, 24, 32):
        raise ValueError("Session key must be 128/192/256 bits")
    return session_key


def encrypt_payload(encrypted_session_key_b64: str, plaintext: str) -> dict[str, str]:
    """
    Decrypt the client's session key, then AES-GCM encrypt ``plaintext`` with it.

    Returns a dict ``{"iv": <b64>, "ciphertext": <b64>}`` where ``ciphertext``
    includes the 16-byte GCM tag appended by the AESGCM primitive.
    """
    session_key = _decrypt_session_key(encrypted_session_key_b64)
    iv = os.urandom(12)  # 96-bit IV is the GCM standard
    aesgcm = AESGCM(session_key)
    ciphertext = aesgcm.encrypt(iv, plaintext.encode("utf-8"), associated_data = None)
    return {
        "iv": base64.b64encode(iv).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }
