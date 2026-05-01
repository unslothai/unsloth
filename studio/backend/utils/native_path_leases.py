# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verification for Tauri native path signed grants.

Rust signs compact ``base64url(payload_json).base64url(hmac)`` grants. The
frontend can see and forward the grant, but cannot change it without breaking
the HMAC. The backend verifies the original payload segment bytes, then
re-stats the path before any native read.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from contextlib import contextmanager
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

LEASE_SECRET_ENV = "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET"
_MAX_NATIVE_PATH_REDACTIONS = 100

_REPLAY_LOCK = threading.Lock()
_USED_NONCES: dict[str, int] = {}
_REDACTION_LOCK = threading.Lock()
_NATIVE_PATH_REDACTIONS: list[str] = []
_NATIVE_PATH_LABELS: dict[str, str] = {}


class NativePathLeaseError(ValueError):
    """Raised when a native path grant is missing, invalid, or unsafe."""


@dataclass(frozen = True)
class NativePathGrant:
    operation: str
    canonical_path: Path
    path_kind: str
    path_type: str
    source_kind: str
    token_id_hash: str
    display_label: str
    expires_at_ms: int
    size_bytes: int | None
    modified_ms: int | None


def native_path_leases_supported() -> bool:
    return bool(os.environ.get(LEASE_SECRET_ENV))


def child_env_without_native_path_secret(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child-process env with the native path lease secret removed."""

    cleaned = dict(os.environ if env is None else env)
    cleaned.pop(LEASE_SECRET_ENV, None)
    return cleaned


@contextmanager
def native_path_secret_removed_from_environ():
    """Temporarily prevent multiprocessing children from inheriting the lease secret."""

    prior = os.environ.pop(LEASE_SECRET_ENV, None)
    try:
        yield
    finally:
        if prior is not None:
            os.environ[LEASE_SECRET_ENV] = prior


def verify_native_path_lease(
    lease: str | None,
    *,
    operation: str,
    expected_kind: str | None = None,
    expected_path_type: str | None = None,
    allowed_suffixes: Iterable[str] | None = None,
) -> NativePathGrant:
    if not lease:
        raise NativePathLeaseError("Native path grant is required.")

    secret = _decode_secret()
    payload_b64, signature_b64 = _split_lease(lease)
    expected_signature = hmac.new(
        secret,
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    supplied_signature = _b64decode(signature_b64)
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise NativePathLeaseError("Native path grant signature is invalid.")

    payload = _decode_payload(payload_b64)
    _validate_payload(payload, operation = operation, expected_kind = expected_kind)

    path = Path(str(payload["canonical_path"]))
    _reject_network_or_device_path(path)
    resolved = path.resolve(strict = True)
    if os.path.normcase(str(resolved)) != os.path.normcase(str(payload["canonical_path"])):
        raise NativePathLeaseError("Native path grant no longer resolves to the selected path.")

    grant = NativePathGrant(
        operation = str(payload["operation"]),
        canonical_path = resolved,
        path_kind = str(payload["path_kind"]),
        path_type = str(payload["path_type"]),
        source_kind = str(payload["source_kind"]),
        token_id_hash = str(payload["token_id_hash"]),
        display_label = str(payload.get("display_label") or resolved.name),
        expires_at_ms = int(payload["expires_at_ms"]),
        size_bytes = _optional_int(payload.get("size_bytes")),
        modified_ms = _optional_int(payload.get("modified_ms")),
    )

    if expected_path_type and grant.path_type != expected_path_type:
        raise NativePathLeaseError("Native path grant has the wrong path type.")
    suffixes = tuple(s.lower() for s in (allowed_suffixes or ()))
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise NativePathLeaseError("Native path grant has an unsupported file type.")

    _validate_current_stat(grant)
    _consume_nonce(str(payload["nonce"]), grant.expires_at_ms)
    _remember_native_path_for_redaction(str(resolved), grant.display_label)
    return grant


def display_label_for_native_path(value: str | None) -> str | None:
    if not value:
        return value
    with _REDACTION_LOCK:
        return _NATIVE_PATH_LABELS.get(value, value)


def is_registered_native_path_label(path_value: str | None, label: str | None) -> bool:
    if not path_value or not label:
        return False
    with _REDACTION_LOCK:
        return _NATIVE_PATH_LABELS.get(path_value) == label


def redact_native_paths(value: str) -> str:
    with _REDACTION_LOCK:
        paths = list(_NATIVE_PATH_REDACTIONS)
    redacted = value
    for path in paths:
        for variant in {path, path.replace("/", "\\"), path.replace("\\", "/")}:
            if variant:
                redacted = redacted.replace(variant, "<native_path>")
    return redacted


def _decode_secret() -> bytes:
    encoded = os.environ.get(LEASE_SECRET_ENV)
    if not encoded:
        raise NativePathLeaseError("Native path grants require the managed desktop backend.")
    try:
        return _b64decode(encoded)
    except Exception as exc:
        raise NativePathLeaseError("Native path grant secret is invalid.") from exc


def _split_lease(lease: str) -> tuple[str, str]:
    parts = lease.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise NativePathLeaseError("Native path grant has an invalid format.")
    return parts[0], parts[1]


def _decode_payload(payload_b64: str) -> dict[str, Any]:
    try:
        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise NativePathLeaseError("Native path grant payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise NativePathLeaseError("Native path grant payload is invalid.")
    return payload


def _validate_payload(
    payload: dict[str, Any], *, operation: str, expected_kind: str | None
) -> None:
    required = (
        "version",
        "operation",
        "canonical_path",
        "path_kind",
        "path_type",
        "source_kind",
        "token_id_hash",
        "issued_at_ms",
        "expires_at_ms",
        "nonce",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise NativePathLeaseError("Native path grant payload is missing required fields.")
    if int(payload["version"]) != 1:
        raise NativePathLeaseError("Native path grant version is unsupported.")
    if payload["operation"] != operation:
        raise NativePathLeaseError("Native path grant operation is invalid.")
    if expected_kind and payload["path_kind"] != expected_kind:
        raise NativePathLeaseError("Native path grant kind is invalid.")
    now_ms = int(time.time() * 1000)
    if int(payload["expires_at_ms"]) <= now_ms:
        raise NativePathLeaseError("Native path grant has expired.")
    if int(payload["issued_at_ms"]) > now_ms + 30_000:
        raise NativePathLeaseError("Native path grant issue time is invalid.")
    for key in ("canonical_path", "nonce", "token_id_hash"):
        value = str(payload[key])
        if "\x00" in value:
            raise NativePathLeaseError("Native path grant contains invalid characters.")


def _validate_current_stat(grant: NativePathGrant) -> None:
    if grant.path_type == "file":
        if not grant.canonical_path.is_file():
            raise NativePathLeaseError("Native path is no longer a regular file.")
    elif grant.path_type == "directory":
        if not grant.canonical_path.is_dir():
            raise NativePathLeaseError("Native path is no longer a directory.")
    else:
        raise NativePathLeaseError("Native path grant has an unsupported path type.")

    stat = grant.canonical_path.stat()
    if grant.size_bytes is not None and stat.st_size != grant.size_bytes:
        raise NativePathLeaseError("Native path changed after it was selected.")
    current_modified_ms = int(stat.st_mtime_ns // 1_000_000)
    if grant.modified_ms is not None and current_modified_ms != grant.modified_ms:
        raise NativePathLeaseError("Native path changed after it was selected.")


def _consume_nonce(nonce: str, expires_at_ms: int) -> None:
    now_ms = int(time.time() * 1000)
    with _REPLAY_LOCK:
        for key, expiry in list(_USED_NONCES.items()):
            if expiry <= now_ms:
                _USED_NONCES.pop(key, None)
        if nonce in _USED_NONCES:
            raise NativePathLeaseError("Native path grant was already used.")
        _USED_NONCES[nonce] = expires_at_ms


def _remember_native_path_for_redaction(path: str, display_label: str) -> None:
    with _REDACTION_LOCK:
        _NATIVE_PATH_LABELS[path] = display_label
        if path in _NATIVE_PATH_REDACTIONS:
            return
        _NATIVE_PATH_REDACTIONS.append(path)
        for stale_path in _NATIVE_PATH_REDACTIONS[:-_MAX_NATIVE_PATH_REDACTIONS]:
            _NATIVE_PATH_LABELS.pop(stale_path, None)
        del _NATIVE_PATH_REDACTIONS[:-_MAX_NATIVE_PATH_REDACTIONS]


def _reject_network_or_device_path(path: Path) -> None:
    text = str(path)
    if os.name == "nt" and (text.startswith("\\\\") or text.startswith("//")):
        raise NativePathLeaseError("Network paths are not supported for native grants.")
    if os.name != "nt":
        for root in ("/dev", "/proc", "/sys"):
            try:
                path.relative_to(root)
                raise NativePathLeaseError("Device and virtual filesystem paths are not supported.")
            except ValueError:
                pass
    if "\x00" in text:
        raise NativePathLeaseError("Native path contains invalid characters.")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
