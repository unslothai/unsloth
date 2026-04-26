# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Resolves SSL/TLS configuration from CLI args, environment variables, and
the persistent ``app_secrets`` table, and generates a self-signed cert
on demand.

Precedence (highest first):
  1. ``--no-ssl`` CLI flag → force HTTP regardless of every other source.
  2. ``--ssl-certfile`` / ``--ssl-keyfile`` / ``--ssl-self-signed`` flags.
  3. ``UNSLOTH_SSL_CERTFILE`` / ``UNSLOTH_SSL_KEYFILE`` / ``UNSLOTH_SSL_SELF_SIGNED`` env vars.
  4. ``app_secrets`` rows written by the in-app Settings page.
  5. Default → SSL disabled (HTTP).
"""

from __future__ import annotations

import ipaddress
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from utils.paths import ensure_dir, studio_root

# ── app_secrets keys ───────────────────────────────────────────────────
SSL_ENABLED_KEY = "ssl_enabled"
SSL_CERTFILE_KEY = "ssl_certfile"
SSL_KEYFILE_KEY = "ssl_keyfile"
SSL_SELF_SIGNED_KEY = "ssl_self_signed"

# ── env var names ──────────────────────────────────────────────────────
ENV_CERTFILE = "UNSLOTH_SSL_CERTFILE"
ENV_KEYFILE = "UNSLOTH_SSL_KEYFILE"
ENV_SELF_SIGNED = "UNSLOTH_SSL_SELF_SIGNED"


def ssl_root() -> Path:
    """Directory holding the auto-generated self-signed cert + key."""
    return studio_root() / "ssl"


def _self_signed_cert_path() -> Path:
    return ssl_root() / "cert.pem"


def _self_signed_key_path() -> Path:
    return ssl_root() / "key.pem"


@dataclass(frozen = True)
class SslSettings:
    """Resolved SSL configuration handed to uvicorn.Config."""

    enabled: bool
    certfile: Optional[str]
    keyfile: Optional[str]
    # Which layer of the precedence chain produced this result. Used by
    # the Settings UI to explain why "Active" can diverge from the saved
    # toggle (e.g. running with --ssl-self-signed while DB has SSL off).
    source: str = "default"

    @property
    def scheme(self) -> str:
        return "https" if self.enabled else "http"


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _read_db_settings() -> dict:
    """Load SSL settings from ``app_secrets``. Returns an empty dict on failure
    so missing/uninitialized auth.db never blocks server startup."""
    try:
        from auth.storage import get_app_secret
    except Exception:
        return {}
    try:
        return {
            "enabled": _truthy(get_app_secret(SSL_ENABLED_KEY)),
            "certfile": get_app_secret(SSL_CERTFILE_KEY) or None,
            "keyfile": get_app_secret(SSL_KEYFILE_KEY) or None,
            "self_signed": _truthy(get_app_secret(SSL_SELF_SIGNED_KEY)),
        }
    except Exception:
        return {}


@dataclass
class SslCliArgs:
    """Subset of argparse namespace consumed by the resolver."""

    no_ssl: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_self_signed: bool = False


def resolve_ssl_settings(
    cli: SslCliArgs,
    *,
    env: Optional[Mapping[str, str]] = None,
    bind_host: str = "0.0.0.0",
) -> SslSettings:
    """Walk the precedence chain and return the resolved settings.

    Caller should pass ``bind_host`` so the self-signed cert can include
    it in its SubjectAltName when it is a concrete address.
    """
    if env is None:
        env = os.environ

    # 1. Explicit escape hatch — overrides everything.
    if cli.no_ssl:
        return SslSettings(
            enabled = False,
            certfile = None,
            keyfile = None,
            source = "cli_no_ssl",
        )

    # 2. Resolve sources in priority order.
    cli_has_paths = bool(cli.ssl_certfile and cli.ssl_keyfile)
    cli_has_self_signed = bool(cli.ssl_self_signed)
    env_certfile = env.get(ENV_CERTFILE)
    env_keyfile = env.get(ENV_KEYFILE)
    env_self_signed = _truthy(env.get(ENV_SELF_SIGNED))
    env_has_paths = bool(env_certfile and env_keyfile)

    db = _read_db_settings()

    if cli_has_paths:
        certfile = cli.ssl_certfile
        keyfile = cli.ssl_keyfile
        source = "cli_paths"
    elif cli_has_self_signed:
        certfile, keyfile = ensure_self_signed_cert(bind_host = bind_host)
        source = "cli_self_signed"
    elif env_has_paths:
        certfile = env_certfile
        keyfile = env_keyfile
        source = "env_paths"
    elif env_self_signed:
        certfile, keyfile = ensure_self_signed_cert(bind_host = bind_host)
        source = "env_self_signed"
    elif db.get("enabled"):
        if db.get("self_signed"):
            certfile, keyfile = ensure_self_signed_cert(bind_host = bind_host)
            source = "db_self_signed"
        else:
            certfile = db.get("certfile")
            keyfile = db.get("keyfile")
            source = "db_paths"
            if not certfile or not keyfile:
                raise RuntimeError(
                    "SSL is enabled in saved settings but certfile/keyfile are "
                    "missing. Re-open Settings → Server, or run "
                    "`unsloth studio --no-ssl` to recover."
                )
    else:
        return SslSettings(
            enabled = False,
            certfile = None,
            keyfile = None,
            source = "default",
        )

    _validate_cert_files(certfile, keyfile)
    return SslSettings(
        enabled = True,
        certfile = certfile,
        keyfile = keyfile,
        source = source,
    )


def _validate_cert_files(certfile: str, keyfile: str) -> None:
    """Raise ``RuntimeError`` if the cert/key cannot be loaded by OpenSSL."""
    cert_path = Path(certfile).expanduser()
    key_path = Path(keyfile).expanduser()
    if not cert_path.is_file():
        raise RuntimeError(f"SSL certificate file not found: {cert_path}")
    if not key_path.is_file():
        raise RuntimeError(f"SSL key file not found: {key_path}")
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile = str(cert_path), keyfile = str(key_path))
    except ssl.SSLError as exc:
        raise RuntimeError(f"SSL certificate/key are not a valid pair: {exc}") from exc


def ensure_self_signed_cert(*, bind_host: str = "0.0.0.0") -> tuple[str, str]:
    """Return ``(certfile, keyfile)`` paths, generating them once if missing.

    The generated cert is valid for 10 years. CN is ``localhost``;
    SubjectAltName covers ``localhost``, ``127.0.0.1``, ``::1``, and
    *bind_host* when it is a concrete (non-wildcard) IP.
    """
    cert_path = _self_signed_cert_path()
    key_path = _self_signed_key_path()
    if cert_path.is_file() and key_path.is_file():
        return (str(cert_path), str(key_path))

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError(
            "Self-signed cert generation requires the `cryptography` package. "
            "Install it (`pip install cryptography`) or pass --ssl-certfile / "
            "--ssl-keyfile with your own cert."
        ) from exc

    import datetime as _dt

    ensure_dir(ssl_root())

    private_key = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ]
    if bind_host and bind_host not in (
        "0.0.0.0",
        "::",
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(bind_host)))
        except ValueError:
            san_entries.append(x509.DNSName(bind_host))

    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes = 5))
        .not_valid_after(now + _dt.timedelta(days = 3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical = False)
        .add_extension(x509.BasicConstraints(ca = False, path_length = None), critical = True)
        .sign(private_key, hashes.SHA256())
    )

    key_path.write_bytes(
        private_key.private_bytes(
            encoding = serialization.Encoding.PEM,
            format = serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm = serialization.NoEncryption(),
        )
    )
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return (str(cert_path), str(key_path))


def regenerate_self_signed_cert(*, bind_host: str = "0.0.0.0") -> tuple[str, str]:
    """Force-rotate the self-signed cert. Used by the Settings router."""
    cert_path = _self_signed_cert_path()
    key_path = _self_signed_key_path()
    cert_path.unlink(missing_ok = True)
    key_path.unlink(missing_ok = True)
    return ensure_self_signed_cert(bind_host = bind_host)
