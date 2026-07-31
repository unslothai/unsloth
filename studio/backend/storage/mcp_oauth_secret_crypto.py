# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated encryption for persisted MCP OAuth client secrets.

Windows uses DPAPI scoped to the current OS user and keeps no application key.
POSIX uses Fernet with a mode-0600 sidecar separate from ``studio.db``. Both
protect database copies and routine database inspection from exposing client
secrets. The same OS user (or an administrator) can still access credentials
through the running Studio process.
"""

from __future__ import annotations

import base64
import ctypes
import os
import tempfile
from ctypes import wintypes
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from utils.paths import ensure_dir
from utils.paths.storage_roots import studio_root

_KEY_FILENAME = ".mcp-oauth-client-secret.key"
_PRIVATE_FILE_MODE = 0o600
_IS_WINDOWS = os.name == "nt"
_FERNET_PREFIX = "fernet:"
_DPAPI_PREFIX = "dpapi:"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _key_path() -> Path:
    return studio_root() / _KEY_FILENAME


def _read_key(path: Path) -> bytes:
    with path.open("rb") as handle:
        key = handle.read()
    # Constructor validation rejects truncated/corrupt keys before any secret
    # can be stored with an unusable encryption setup.
    Fernet(key)
    return key


def _load_existing_key() -> bytes:
    path = _key_path()
    try:
        os.chmod(path, _PRIVATE_FILE_MODE)
    except FileNotFoundError:
        raise FileNotFoundError(
            "MCP OAuth client-secret key is missing; stored secrets cannot be decrypted"
        ) from None
    return _read_key(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_key(temporary_path: Path, path: Path) -> None:
    os.link(temporary_path, path)
    _fsync_directory(path.parent)


def _load_or_create_key() -> bytes:
    path = _key_path()
    ensure_dir(path.parent)
    if path.exists():
        return _load_existing_key()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{_KEY_FILENAME}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        key = Fernet.generate_key()
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, _PRIVATE_FILE_MODE)
        try:
            # Publication is atomic and never overwrites a key created
            # concurrently by another Studio process.
            _publish_key(temporary_path, path)
            return key
        except FileExistsError:
            return _load_existing_key()
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        removed = False
        try:
            temporary_path.unlink()
            removed = True
        except FileNotFoundError:
            pass
        if removed:
            _fsync_directory(path.parent)


def encrypt_client_secret(secret: str | None) -> str | None:
    if not secret:
        return None
    encoded = secret.encode("utf-8")
    if _IS_WINDOWS:
        protected = _dpapi_transform(encoded, decrypt=False)
        return f"{_DPAPI_PREFIX}{base64.b64encode(protected).decode('ascii')}"
    encrypted = Fernet(_load_or_create_key()).encrypt(encoded).decode("ascii")
    return f"{_FERNET_PREFIX}{encrypted}"


def decrypt_client_secret(encrypted_secret: str | None) -> str | None:
    if not encrypted_secret:
        return None
    if encrypted_secret.startswith(_DPAPI_PREFIX):
        if not _IS_WINDOWS:
            raise ValueError("Windows-protected MCP OAuth secret cannot be decrypted here")
        payload = base64.b64decode(encrypted_secret.removeprefix(_DPAPI_PREFIX))
        return _dpapi_transform(payload, decrypt=True).decode("utf-8")
    if not encrypted_secret.startswith(_FERNET_PREFIX):
        raise ValueError("Stored MCP OAuth client secret uses an unknown encryption format")
    if _IS_WINDOWS:
        raise ValueError("POSIX-protected MCP OAuth secret cannot be decrypted on Windows")
    payload = encrypted_secret.removeprefix(_FERNET_PREFIX)
    try:
        return Fernet(_load_existing_key()).decrypt(payload.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Stored MCP OAuth client secret cannot be decrypted") from exc


def _dpapi_transform(data: bytes, *, decrypt: bool) -> bytes:
    """Protect/unprotect bytes with Windows DPAPI for the current OS user."""
    input_buffer = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(
        len(data),
        ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    output_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    function.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    ctypes.set_last_error(0)
    succeeded = function(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    )
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output_blob.data, output_blob.size)
    finally:
        local_free(output_blob.data)
