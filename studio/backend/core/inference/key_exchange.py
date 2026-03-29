# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
RSA key pair for encrypting API keys in transit.

The frontend encrypts API keys with the server's public key before
including them in requests. The backend decrypts with its private key
before forwarding to external providers.

The key pair is generated at server startup and lives only in memory —
it is regenerated on each restart. The frontend fetches the public key
via GET /api/providers/public-key on load.
"""

import base64
import logging

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

logger = logging.getLogger(__name__)

_private_key: rsa.RSAPrivateKey | None = None
_public_key_pem: str | None = None


def init_key_pair() -> None:
    """Generate an RSA-2048 key pair. Called once at server startup."""
    global _private_key, _public_key_pem
    _private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    _public_key_pem = _private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    logger.info("RSA key pair generated for API key encryption")


def get_public_key_pem() -> str:
    """Return the PEM-encoded public key for the frontend."""
    if _public_key_pem is None:
        raise RuntimeError("Key pair not initialized. Call init_key_pair() first.")
    return _public_key_pem


def decrypt_api_key(encrypted_b64: str) -> str:
    """
    Decrypt an API key that was encrypted with the public key.

    Args:
        encrypted_b64: Base64-encoded RSA-OAEP ciphertext.

    Returns:
        The plaintext API key string.
    """
    if _private_key is None:
        raise RuntimeError("Key pair not initialized. Call init_key_pair() first.")

    ciphertext = base64.b64decode(encrypted_b64)
    plaintext = _private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return plaintext.decode("utf-8")
