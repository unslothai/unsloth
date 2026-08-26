# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verification for Tauri native path grants.

Production grants use an authenticated encrypted envelope, so the renderer can
forward a grant without learning or changing its path. The backend authenticates
the envelope before decryption, then re-stats the path before any native read.
Version 1 verification remains only for grants from an older desktop process
during an in-place backend transition.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import importlib
import json
import os
import stat as _stat_module
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping

LEASE_SECRET_ENV = "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET"
_MAX_NATIVE_PATH_LABELS = 10_000
_MAX_NATIVE_PATH_REDACTIONS = _MAX_NATIVE_PATH_LABELS
_MIN_LEASE_SECRET_BYTES = 32
_WINDOWS_STAT_USES_FILE_ID_INFO = os.name == "nt" and sys.version_info >= (3, 12)
_LEASE_V2_PREFIX = "2"
_LEASE_V2_NONCE_BYTES = 16
_LEASE_V2_ENCRYPTION_DOMAIN = b"unsloth-native-path-lease-v2-encryption\0"
_LEASE_V2_AUTH_DOMAIN = b"unsloth-native-path-lease-v2-auth\0"

_REPLAY_LOCK = threading.Lock()
_USED_NONCES: dict[str, int] = {}
_REDACTION_LOCK = threading.Lock()
_NATIVE_PATH_REDACTIONS: list[str] = []
_NATIVE_PATH_LABELS: dict[str, str] = {}
_NATIVE_PATH_ENV_LOCK = threading.RLock()
_SECRET_INIT_LOCK = threading.Lock()
_CACHED_LEASE_SECRET: bytes | None = None
_SCRUB_REFCOUNT = 0
_SCRUB_SAVED_SECRET: str | None = None


class NativePathLeaseError(ValueError):
    """Raised when a native path grant is missing, invalid, or unsafe."""


def native_gguf_companion_parent_allowed(
    companion_path: str | Path,
    gguf_path: str | Path,
    *,
    allowed_subdirs: Collection[str] = (),
    mtp_search_root: str | Path | None = None,
) -> bool:
    """Check whether a GGUF companion is in an allowed directory.

    ``allowed_subdirs`` names the companion directories (``mtp``, ``dspark``)
    this caller may reach into, beside the weight's own. A collection rather
    than one flag per kind: each caller admits exactly the kind it is
    resolving, so an MTP load never accepts a sidecar out of ``dspark/``.
    """
    companion_parent = Path(companion_path).resolve(strict = True).parent
    gguf_parent = Path(gguf_path).resolve(strict = True).parent
    if companion_parent == gguf_parent:
        return True
    permitted = {name.casefold() for name in allowed_subdirs}
    if companion_parent.name.casefold() not in permitted:
        return False
    allowed_roots = {gguf_parent}
    if mtp_search_root is not None:
        search_root = Path(mtp_search_root).resolve(strict = True)
        if search_root in {gguf_parent, gguf_parent.parent}:
            allowed_roots.add(search_root)
    return companion_parent.parent in allowed_roots


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
    device_id: int | None
    file_id: int | None


def native_path_leases_supported() -> bool:
    try:
        _decode_secret()
    except NativePathLeaseError:
        return False
    return True


def child_env_without_native_path_secret(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child-process env with the native path lease secret removed."""

    if env is None:
        with _NATIVE_PATH_ENV_LOCK:
            cleaned = dict(os.environ)
    else:
        cleaned = dict(env)
    cleaned.pop(LEASE_SECRET_ENV, None)
    return cleaned


def run_without_native_path_secret(
    target: Callable[..., Any] | str, *args: Any, **kwargs: Any
) -> Any:
    """Run a multiprocessing child target without the native path lease secret."""

    # Runs in the spawned child: bind it to the parent's death (Linux), since
    # multiprocessing children cannot be given a preexec_fn by the parent. Shared
    # entrypoint for the inference/export/training/data-recipe workers.
    # Two try blocks, not one: allow_child_processes is the newer name, so on an
    # older process_lifetime.py a combined import would lose the binding as well.
    try:
        from utils.process_lifetime import bind_current_process_to_parent_lifetime
        bind_current_process_to_parent_lifetime()
    except Exception:
        pass
    try:
        # Clear the worker's daemon policy so HF prefetch can spawn (#9094).
        from utils.process_lifetime import allow_child_processes
        allow_child_processes()
    except Exception:
        pass

    global _CACHED_LEASE_SECRET, _SCRUB_SAVED_SECRET
    os.environ.pop(LEASE_SECRET_ENV, None)
    _CACHED_LEASE_SECRET = None
    _SCRUB_SAVED_SECRET = None
    if isinstance(target, str):
        function_name, environment, *args = args
        for key, value in environment.items():
            os.environ[key] = value
        target = getattr(importlib.import_module(target), function_name)
    return target(*args, **kwargs)


@contextmanager
def native_path_secret_removed_for_child_start() -> Iterator[None]:
    global _SCRUB_REFCOUNT, _SCRUB_SAVED_SECRET, _CACHED_LEASE_SECRET
    with _NATIVE_PATH_ENV_LOCK:
        if _SCRUB_REFCOUNT == 0:
            _SCRUB_SAVED_SECRET = os.environ.pop(LEASE_SECRET_ENV, None)
            _CACHED_LEASE_SECRET = None
        _SCRUB_REFCOUNT += 1
        try:
            yield
        finally:
            _SCRUB_REFCOUNT -= 1
            if _SCRUB_REFCOUNT == 0 and _SCRUB_SAVED_SECRET is not None:
                os.environ[LEASE_SECRET_ENV] = _SCRUB_SAVED_SECRET
                _SCRUB_SAVED_SECRET = None


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
    payload, envelope_version = _decode_authenticated_payload(lease, secret)
    _validate_payload(
        payload,
        operation = operation,
        expected_kind = expected_kind,
        envelope_version = envelope_version,
    )

    path = Path(str(payload["canonical_path"]))
    _reject_network_or_device_path(path)
    try:
        signed_lstat = os.lstat(path)
    except OSError as exc:
        raise NativePathLeaseError("Native path is no longer accessible.") from exc
    if _stat_module.S_ISLNK(signed_lstat.st_mode):
        raise NativePathLeaseError("Native path is no longer a regular file.")
    try:
        resolved = path.resolve(strict = True)
    except OSError as exc:
        raise NativePathLeaseError("Native path is no longer accessible.") from exc
    _reject_network_or_device_path(resolved)
    if not _same_native_path(resolved, path):
        raise NativePathLeaseError("Native path grant no longer resolves to the selected path.")

    identity_options = _identity_options(payload)
    grant = NativePathGrant(
        operation = str(payload["operation"]),
        canonical_path = resolved,
        path_kind = str(payload["path_kind"]),
        path_type = str(payload["path_type"]),
        source_kind = str(payload["source_kind"]),
        token_id_hash = str(payload["token_id_hash"]),
        display_label = str(payload.get("display_label") or resolved.name),
        expires_at_ms = _required_int(payload, "expires_at_ms"),
        size_bytes = _optional_int(payload.get("size_bytes")),
        modified_ms = _optional_int(payload.get("modified_ms")),
        device_id = identity_options[0][0] if identity_options else None,
        file_id = identity_options[0][1] if identity_options else None,
    )

    if expected_path_type and grant.path_type != expected_path_type:
        raise NativePathLeaseError("Native path grant has the wrong path type.")
    suffixes = tuple(s.lower() for s in (allowed_suffixes or ()))
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise NativePathLeaseError("Native path grant has an unsupported file type.")

    current_identity = _validate_current_stat(grant, identity_options)
    if current_identity is not None:
        grant = replace(grant, device_id = current_identity[0], file_id = current_identity[1])
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
        paths = sorted(_NATIVE_PATH_REDACTIONS, key = len, reverse = True)
    redacted = value
    for path in paths:
        for variant in {path, path.replace("/", "\\"), path.replace("\\", "/")}:
            if variant:
                redacted = redacted.replace(variant, "<native_path>")
    return redacted


def _decode_secret() -> bytes:
    global _CACHED_LEASE_SECRET
    if _CACHED_LEASE_SECRET is not None:
        return _CACHED_LEASE_SECRET
    with _SECRET_INIT_LOCK:
        if _CACHED_LEASE_SECRET is not None:
            return _CACHED_LEASE_SECRET
        with _NATIVE_PATH_ENV_LOCK:
            encoded = os.environ.get(LEASE_SECRET_ENV)
            if encoded is None and _SCRUB_SAVED_SECRET is not None:
                encoded = _SCRUB_SAVED_SECRET
        if not encoded:
            raise NativePathLeaseError("Native path grants require the managed desktop backend.")
        try:
            secret = _b64decode(encoded)
        except Exception as exc:
            raise NativePathLeaseError("Native path grant secret is invalid.") from exc
        if len(secret) < _MIN_LEASE_SECRET_BYTES:
            raise NativePathLeaseError("Native path grant secret is invalid.")
        _CACHED_LEASE_SECRET = secret
        return secret


def _lease_parts(lease: str) -> list[str]:
    if not isinstance(lease, str):
        raise NativePathLeaseError("Native path grant has an invalid format.")
    try:
        lease.encode("ascii")
    except UnicodeEncodeError as exc:
        raise NativePathLeaseError("Native path grant has an invalid format.") from exc
    parts = lease.split(".")
    if any(not part for part in parts):
        raise NativePathLeaseError("Native path grant has an invalid format.")
    return parts


def _decode_payload_bytes(payload_bytes: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise NativePathLeaseError("Native path grant payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise NativePathLeaseError("Native path grant payload is invalid.")
    return payload


def _xor_lease_stream(secret: bytes, nonce: bytes, value: bytes) -> bytes:
    output = bytearray()
    for block_index in range(0, len(value), hashlib.sha256().digest_size):
        counter = block_index // hashlib.sha256().digest_size
        seed = _LEASE_V2_ENCRYPTION_DOMAIN + nonce + counter.to_bytes(8, "big")
        stream = hmac.new(secret, seed, hashlib.sha256).digest()
        chunk = value[block_index : block_index + len(stream)]
        output.extend(byte ^ mask for byte, mask in zip(chunk, stream))
    return bytes(output)


def _decode_authenticated_payload(lease: str, secret: bytes) -> tuple[dict[str, Any], int]:
    parts = _lease_parts(lease)
    if len(parts) == 3 and parts[0] == _LEASE_V2_PREFIX:
        envelope = _b64decode(parts[1])
        if len(envelope) <= _LEASE_V2_NONCE_BYTES:
            raise NativePathLeaseError("Native path grant has an invalid format.")
        supplied_signature = _b64decode(parts[2])
        if len(supplied_signature) != hashlib.sha256().digest_size:
            raise NativePathLeaseError("Native path grant has an invalid format.")
        expected_signature = hmac.new(
            secret,
            _LEASE_V2_AUTH_DOMAIN + envelope,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise NativePathLeaseError("Native path grant signature is invalid.")
        nonce = envelope[:_LEASE_V2_NONCE_BYTES]
        ciphertext = envelope[_LEASE_V2_NONCE_BYTES:]
        payload = _decode_payload_bytes(_xor_lease_stream(secret, nonce, ciphertext))
        return payload, 2
    if len(parts) == 2:
        payload_b64, signature_b64 = parts
        expected_signature = hmac.new(
            secret,
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).digest()
        supplied_signature = _b64decode(signature_b64)
        if len(supplied_signature) != hashlib.sha256().digest_size:
            raise NativePathLeaseError("Native path grant has an invalid format.")
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise NativePathLeaseError("Native path grant signature is invalid.")
        return _decode_payload_bytes(_b64decode(payload_b64)), 1
    raise NativePathLeaseError("Native path grant has an invalid format.")


def _validate_payload(
    payload: dict[str, Any], *, operation: str, expected_kind: str | None, envelope_version: int
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
    if _required_int(payload, "version") != envelope_version:
        raise NativePathLeaseError("Native path grant version is unsupported.")
    if payload["operation"] != operation:
        raise NativePathLeaseError("Native path grant operation is invalid.")
    if expected_kind and payload["path_kind"] != expected_kind:
        raise NativePathLeaseError("Native path grant kind is invalid.")
    now_ms = int(time.time() * 1000)
    issued_at_ms = _required_int(payload, "issued_at_ms")
    expires_at_ms = _required_int(payload, "expires_at_ms")
    if issued_at_ms >= expires_at_ms:
        raise NativePathLeaseError("Native path grant timestamps are inconsistent.")
    if expires_at_ms <= now_ms:
        raise NativePathLeaseError("Native path grant has expired.")
    if issued_at_ms > now_ms + 30_000:
        raise NativePathLeaseError("Native path grant issue time is invalid.")
    for key in ("canonical_path", "nonce", "token_id_hash", "display_label"):
        raw = payload.get(key)
        if raw is None:
            continue
        if "\x00" in str(raw):
            raise NativePathLeaseError("Native path grant contains invalid characters.")


def _validate_current_stat(
    grant: NativePathGrant, identity_options: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    try:
        st = os.lstat(grant.canonical_path)
    except OSError as exc:
        raise NativePathLeaseError("Native path is no longer accessible.") from exc
    if _stat_module.S_ISLNK(st.st_mode):
        raise NativePathLeaseError("Native path is no longer a regular file.")
    if grant.path_type == "file":
        if not _stat_module.S_ISREG(st.st_mode):
            raise NativePathLeaseError("Native path is no longer a regular file.")
    elif grant.path_type == "directory":
        if not _stat_module.S_ISDIR(st.st_mode):
            raise NativePathLeaseError("Native path is no longer a directory.")
    else:
        raise NativePathLeaseError("Native path grant has an unsupported path type.")

    if grant.size_bytes is not None and st.st_size != grant.size_bytes:
        raise NativePathLeaseError("Native path changed after it was selected.")
    current_modified_ms = int(st.st_mtime_ns // 1_000_000)
    if grant.modified_ms is not None and current_modified_ms != grant.modified_ms:
        raise NativePathLeaseError("Native path changed after it was selected.")
    if grant.path_kind == "document-folder" and not identity_options:
        raise NativePathLeaseError("Native path grant is missing its folder identity.")
    current_identity = (st.st_dev, st.st_ino)
    expected_identity = _runtime_identity(identity_options)
    if expected_identity is not None and current_identity != expected_identity:
        raise NativePathLeaseError("Native path changed after it was selected.")
    return current_identity if expected_identity is not None else None


def _consume_nonce(nonce: str, expires_at_ms: int) -> None:
    now_ms = int(time.time() * 1000)
    if expires_at_ms <= now_ms:
        raise NativePathLeaseError("Native path grant has expired.")
    try:
        nonce_digest = hashlib.sha256(nonce.encode("utf-8")).digest()
        from storage.studio_db import consume_native_path_lease_nonce
        consumed = consume_native_path_lease_nonce(
            nonce_digest,
            expires_at_ms,
            now_ms = now_ms,
        )
    except Exception as exc:
        raise NativePathLeaseError("Native path grant replay protection is unavailable.") from exc
    if not consumed:
        raise NativePathLeaseError("Native path grant was already used.")

    # Retain a bounded process-local diagnostic cache for compatibility with
    # older callers. It is not consulted for replay decisions; SQLite is the
    # authoritative cross-process consume boundary.
    nonce_key = nonce_digest.hex()
    with _REPLAY_LOCK:
        for key, expiry in list(_USED_NONCES.items()):
            if expiry <= now_ms:
                _USED_NONCES.pop(key, None)
        _USED_NONCES[nonce_key] = expires_at_ms


def _remember_native_path_for_redaction(path: str, display_label: str) -> None:
    with _REDACTION_LOCK:
        # Keep the display-label and redaction caches in the same recency order.
        # Every path retained for a durable label must also remain redactable.
        _NATIVE_PATH_LABELS.pop(path, None)
        _NATIVE_PATH_LABELS[path] = display_label
        if path in _NATIVE_PATH_REDACTIONS:
            _NATIVE_PATH_REDACTIONS.remove(path)
        _NATIVE_PATH_REDACTIONS.append(path)
        if len(_NATIVE_PATH_LABELS) > _MAX_NATIVE_PATH_LABELS:
            excess = len(_NATIVE_PATH_LABELS) - _MAX_NATIVE_PATH_LABELS
            for stale_path in list(_NATIVE_PATH_LABELS.keys())[:excess]:
                _NATIVE_PATH_LABELS.pop(stale_path, None)
                try:
                    _NATIVE_PATH_REDACTIONS.remove(stale_path)
                except ValueError:
                    pass
        del _NATIVE_PATH_REDACTIONS[:-_MAX_NATIVE_PATH_REDACTIONS]


def _reject_network_or_device_path(path: Path) -> None:
    text = str(path)
    if os.name == "nt":
        normalized = text.replace("/", "\\").lower()
        if normalized.startswith("\\\\?\\"):
            rest = normalized[4:]
            is_local_drive = len(rest) >= 3 and rest[0].isalpha() and rest[1:3] == ":\\"
            if not is_local_drive:
                raise NativePathLeaseError("Network paths are not supported for native grants.")
        elif normalized.startswith("\\\\"):
            raise NativePathLeaseError("Network paths are not supported for native grants.")
    if os.name != "nt":
        for root in ("/dev", "/proc", "/sys"):
            if path.is_relative_to(root):
                raise NativePathLeaseError("Device and virtual filesystem paths are not supported.")
    if "\x00" in text:
        raise NativePathLeaseError("Native path contains invalid characters.")


def _b64decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        encoded = value.encode("ascii")
        decoded = base64.b64decode(
            encoded + padding.encode("ascii"),
            altchars = b"-_",
            validate = True,
        )
        canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
        if canonical != value:
            raise NativePathLeaseError("Native path grant has an invalid format.")
        return decoded
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise NativePathLeaseError("Native path grant has an invalid format.") from exc


def _same_native_path(resolved: Path, signed: Path) -> bool:
    try:
        return resolved.samefile(signed)
    except OSError:
        return os.path.normcase(str(resolved)) == os.path.normcase(str(signed))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise NativePathLeaseError("Native path grant payload is invalid.") from exc


def _identity_options(payload: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    devices = _optional_identities(payload.get("device_id"))
    files = _optional_identities(payload.get("file_id"))
    if devices is None and files is None:
        return ()
    if devices is None or files is None or len(devices) != len(files):
        raise NativePathLeaseError("Native path grant payload is invalid.")
    return tuple(zip(devices, files))


def _runtime_identity(identity_options: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    if not identity_options:
        return None
    if len(identity_options) == 1:
        return identity_options[0]
    # Rust encodes the legacy Win32 pair first and FILE_ID_INFO second.
    return identity_options[1] if _WINDOWS_STAT_USES_FILE_ID_INFO else identity_options[0]


def _optional_identities(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, str) or value != value.lower():
        raise NativePathLeaseError("Native path grant payload is invalid.")
    parts = value.split(":")
    if not 1 <= len(parts) <= 2 or any(
        not part or any(char not in "0123456789abcdef" for char in part) for part in parts
    ):
        raise NativePathLeaseError("Native path grant payload is invalid.")
    try:
        return tuple(int(part, 16) for part in parts)
    except ValueError as exc:
        raise NativePathLeaseError("Native path grant payload is invalid.") from exc


def _required_int(payload: dict[str, Any], key: str) -> int:
    raw = payload.get(key)
    if raw is None:
        raise NativePathLeaseError("Native path grant payload is missing required fields.")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise NativePathLeaseError("Native path grant payload is invalid.") from exc
