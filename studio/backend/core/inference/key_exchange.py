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
import hashlib
import logging

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

logger = logging.getLogger(__name__)

_private_key: rsa.RSAPrivateKey | None = None
_public_key_pem: str | None = None
_public_key_fingerprint: str | None = None


def _compute_fingerprint(pem: str) -> str:
    """SHA256 of the PEM bytes, truncated for log compactness."""
    return hashlib.sha256(pem.encode("utf-8")).hexdigest()[:16]


def init_key_pair() -> None:
    """Generate an RSA-2048 key pair. Called once at server startup."""
    global _private_key, _public_key_pem, _public_key_fingerprint
    if _private_key is not None:
        # Re-entry is suspicious — every fresh keypair invalidates all
        # in-flight ciphertext encrypted against the previous public key.
        # Log loudly so a regression that calls init twice is visible.
        logger.warning(
            "init_key_pair called again — replacing existing RSA keypair "
            "(previous fingerprint=%s). Any frontend that cached the old "
            "public key will start hitting decryption failures.",
            _public_key_fingerprint,
        )
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
    _public_key_fingerprint = _compute_fingerprint(_public_key_pem)
    logger.info(
        "RSA key pair generated for API key encryption (fingerprint=%s)",
        _public_key_fingerprint,
    )


def get_public_key_fingerprint() -> str | None:
    """Short SHA256 of the current public key PEM; None before init."""
    return _public_key_fingerprint


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

    try:
        ciphertext = base64.b64decode(encrypted_b64)
    except Exception as exc:
        logger.warning(
            "decrypt_api_key: base64 decode failed (input_len=%d, fingerprint=%s): %s: %s",
            len(encrypted_b64),
            _public_key_fingerprint,
            type(exc).__name__,
            exc,
        )
        raise

    try:
        plaintext = _private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf = padding.MGF1(algorithm = hashes.SHA256()),
                algorithm = hashes.SHA256(),
                label = None,
            ),
        )
    except Exception as exc:
        # Surface enough state to distinguish key mismatch (wrong public key
        # used on encrypt) from a padding/algo mismatch or corrupted bytes.
        # Expected ciphertext length for RSA-2048 is exactly 256 bytes.
        logger.warning(
            "decrypt_api_key: RSA decrypt failed (ciphertext_len=%d, expected=256, "
            "fingerprint=%s, exc=%s): %s",
            len(ciphertext),
            _public_key_fingerprint,
            type(exc).__name__,
            exc,
        )
        raise

    return plaintext.decode("utf-8")
