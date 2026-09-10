# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
RSA key pair for encrypting API keys in transit.

The frontend encrypts API keys with the server's public key before sending
them; the backend decrypts with its private key before forwarding to external
providers.

The key pair is generated at server startup, lives only in memory, and is
regenerated on each restart. The frontend fetches the public key via
GET /api/providers/public-key on load.

A per-request AES-256-GCM key carries the secret and RSA-OAEP wraps only that
key: encrypting the secret under RSA directly would cap it at 190 bytes.
"""

import base64
import hashlib
import logging

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization, hashes

logger = logging.getLogger(__name__)

_ENVELOPE_VERSION = "v1"
_ENVELOPE_PARTS = 4
# Pins an envelope to this protocol and version, as the at-rest secrets do.
_ENVELOPE_AAD = b"unsloth-studio-provider-key-v1"

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


def _unwrap_oaep(ciphertext: bytes, *, what: str) -> bytes:
    try:
        return _private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf = padding.MGF1(algorithm = hashes.SHA256()),
                algorithm = hashes.SHA256(),
                label = None,
            ),
        )
    except Exception as exc:
        # RSA-2048 ciphertext is exactly 256 bytes; log state to separate key mismatch from padding
        logger.warning(
            "decrypt_api_key: RSA decrypt failed (%s, ciphertext_len=%d, expected=256, "
            "fingerprint=%s, exc=%s): %s",
            what,
            len(ciphertext),
            _public_key_fingerprint,
            type(exc).__name__,
            exc,
        )
        raise


def _b64decode_part(value: str, *, what: str, validate: bool) -> bytes:
    try:
        return base64.b64decode(value, validate = validate)
    except Exception as exc:
        logger.warning(
            "decrypt_api_key: base64 decode failed (%s, input_len=%d, fingerprint=%s): %s: %s",
            what,
            len(value),
            _public_key_fingerprint,
            type(exc).__name__,
            exc,
        )
        raise


def decrypt_api_key(encrypted_b64: str) -> str:
    """Decrypt an API key encrypted with the public key.

    Accepts ``v1.<wrapped AES key>.<nonce>.<ciphertext||tag>``, each part base64, and the
    bare RSA-OAEP ciphertext frontend builds predating the envelope send. Base64 has no
    ``.``, so the two can never be confused.
    """
    if _private_key is None:
        raise RuntimeError("Key pair not initialized. Call init_key_pair() first.")

    if "." in encrypted_b64:
        parts = encrypted_b64.split(".")
        if parts[0] != _ENVELOPE_VERSION or len(parts) != _ENVELOPE_PARTS:
            logger.warning(
                "decrypt_api_key: malformed envelope (version=%r, parts=%d, expected %r/%d, "
                "fingerprint=%s)",
                parts[0][:8],
                len(parts),
                _ENVELOPE_VERSION,
                _ENVELOPE_PARTS,
                _public_key_fingerprint,
            )
            raise ValueError("Unsupported encrypted API key envelope.")
        wrapped_key = _b64decode_part(parts[1], what = "wrapped_key", validate = True)
        nonce = _b64decode_part(parts[2], what = "nonce", validate = True)
        ciphertext = _b64decode_part(parts[3], what = "ciphertext", validate = True)
        aes_key = _unwrap_oaep(wrapped_key, what = "wrapped_key")
        try:
            plaintext = AESGCM(aes_key).decrypt(nonce, ciphertext, _ENVELOPE_AAD)
        except Exception as exc:
            logger.warning(
                "decrypt_api_key: AES-GCM decrypt failed (nonce_len=%d, ciphertext_len=%d, "
                "fingerprint=%s, exc=%s): %s",
                len(nonce),
                len(ciphertext),
                _public_key_fingerprint,
                type(exc).__name__,
                exc,
            )
            raise
    else:
        # Lenient, as this path was before the envelope: tightening it could reject a key
        # that works today.
        legacy_ciphertext = _b64decode_part(encrypted_b64, what = "legacy", validate = False)
        plaintext = _unwrap_oaep(legacy_ciphertext, what = "legacy")

    return plaintext.decode("utf-8")
