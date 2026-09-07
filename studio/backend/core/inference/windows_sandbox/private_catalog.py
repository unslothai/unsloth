# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Invocation-owned Winsock APPKEY snapshots; never modifies the host registry.

Each target must open hive_path with RegLoadAppKeyW and verify the identity marker.
Standalone owners keep their roots until the target is reaped. Production creates
the hive in its bounded preparation worker, closes worker-local roots at handoff,
and retains file cleanup ownership through the invocation reservation. No raw
registry handle crosses processes.
A query-only root does not make subsequently opened private subkeys read-only.
Provider admission is a bounded catalog-format adapter, not LPAC qualification.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as W
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat

from .content_files import native_files
from .dependencies import checked_path
from .profiles import WindowsRuntimeError

_ROOTS = tuple(
    "SYSTEM\\CurrentControlSet\\Services\\" + name + r"\Parameters"
    for name in ("WinSock", "WinSock2")
)
_PROVIDERS = frozenset(
    {
        "mswsock.dll",
        "winrnr.dll",
        "napinsp.dll",
        "nlansp_c.dll",
        "wshbth.dll",
        "rasadhlp.dll",
        "fwpuclnt.dll",
    }
)
_TRANSPORTS = frozenset({"tcpip", "tcpip6", "vmbus", "psched", "afunix", "rfcomm"})
CATALOG_MARKER = "UnslothCatalogIdentity"
_ENTRY = re.compile(
    r"(?:Protocol_Catalog9|NameSpace_Catalog5)\\Catalog_Entries(?:64)?\\[0-9]{12}", re.IGNORECASE
)


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_CATALOG_UNSUPPORTED", message)


@dataclass(frozen = True)
class CatalogBounds:
    keys: int = 512
    values: int = 4096
    bytes: int = 4 * 1024 * 1024
    value_bytes: int = 1024 * 1024
    depth: int = 12


@dataclass(frozen = True)
class _Value:
    path: str
    name: str
    kind: int
    data: object


def _encoded(value):
    data = value.data
    if isinstance(data, bytes):
        data = {"hex": data.hex()}
    return [value.path, value.name, value.kind, data]


def _read_tree(
    source,
    path,
    records,
    counts,
    bounds,
    depth = 0,
):
    import winreg

    subkeys, values, _ = winreg.QueryInfoKey(source)
    counts["keys"] += 1
    if (
        depth > bounds.depth
        or counts["keys"] > bounds.keys
        or subkeys > bounds.keys
        or values > bounds.values
    ):
        raise _invalid("Private catalog key budget exceeded.")
    records.append(_Value(path, "", -1, None))
    for index in range(values):
        name, data, kind = winreg.EnumValue(source, index)
        if kind not in (
            winreg.REG_NONE,
            winreg.REG_BINARY,
            winreg.REG_SZ,
            winreg.REG_EXPAND_SZ,
            winreg.REG_MULTI_SZ,
            winreg.REG_DWORD,
            winreg.REG_QWORD,
        ):
            raise _invalid("Unsupported private catalog value type.")
        if data is None and kind in (winreg.REG_NONE, winreg.REG_BINARY):
            data = b""
        if isinstance(data, bytes):
            size = len(data)
        elif isinstance(data, str):
            size = len(data.encode("utf-16-le")) + 2
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            size = sum(len(item.encode("utf-16-le")) + 2 for item in data) + 2
            data = tuple(data)
        elif isinstance(data, int):
            size = 8
        else:
            raise _invalid("Unsupported private catalog value data.")
        counts["values"] += 1
        counts["bytes"] += size
        if (
            len(name) > 16384
            or size > bounds.value_bytes
            or counts["values"] > bounds.values
            or counts["bytes"] > bounds.bytes
        ):
            raise _invalid("Private catalog value budget exceeded.")
        records.append(_Value(path, name, kind, data))
    for index in range(subkeys):
        name = winreg.EnumKey(source, index)
        if not name or len(name) > 255 or "\\" in name or "\0" in name:
            raise _invalid("Invalid private catalog key name.")
        with winreg.OpenKey(source, name, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as child:
            _read_tree(child, path + "\\" + name, records, counts, bounds, depth + 1)


def _snapshot(bounds):
    import winreg

    records, counts = [], {"keys": 0, "values": 0, "bytes": 0}
    for path in _ROOTS:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        ) as source:
            _read_tree(source, path, records, counts, bounds)
    return tuple(
        sorted(records, key = lambda v: (v.path.casefold(), v.name.casefold(), v.kind))
    ), counts


def _provider_names(records, system_directory):
    """Validate the exact bytes that will be copied, including inactive entries.

    PackedCatalogItem's 260-byte ANSI path + 628-byte WSAPROTOCOL_INFOW layout
    is an observed Windows catalog format, not a public serialization contract.
    Refuse other layouts; the format version is included in the metadata digest.
    """
    import winreg

    names, entries, seen = set(), set(), set()
    prefix = _ROOTS[1] + "\\"
    for value in records:
        relative = value.path[len(prefix) :] if value.path.startswith(prefix) else ""
        if value.kind == -1 and _ENTRY.fullmatch(relative):
            entries.add(value.path)
        if value.path == _ROOTS[0] and value.name.casefold() == "transports":
            if (
                value.kind != winreg.REG_MULTI_SZ
                or not set(item.casefold() for item in value.data) <= _TRANSPORTS
            ):
                raise _invalid("Unreviewed Winsock transport.")
        raw = None
        if value.path == _ROOTS[1] and value.kind != -1:
            expected = {
                "current_namespace_catalog": "NameSpace_Catalog5",
                "current_protocol_catalog": "Protocol_Catalog9",
            }
            if value.name.casefold() in expected:
                if value.kind != winreg.REG_SZ or value.data != expected[value.name.casefold()]:
                    raise _invalid("Unreviewed Winsock catalog selector.")
            elif value.name.casefold() in {"autodialdll", "namespace_callout"}:
                if value.kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
                    raise _invalid("Unsupported Winsock callout format.")
                raw = value.data
            elif value.name.casefold() != "winsock_registry_version":
                raise _invalid("Unreviewed Winsock root setting.")
        if value.name.casefold() == "packedcatalogitem":
            if (
                not _ENTRY.fullmatch(relative)
                or not relative.casefold().startswith("protocol_")
                or value.kind != winreg.REG_BINARY
                or len(value.data) != 888
            ):
                raise _invalid("Unsupported Winsock transport catalog format.")
            path, separator, padding = value.data[:260].partition(b"\0")
            if not separator or any(padding):
                raise _invalid("Invalid Winsock provider path encoding.")
            try:
                raw = path.decode("ascii")
            except UnicodeError as error:
                raise _invalid("Unsupported Winsock provider path encoding.") from error
        elif value.name.casefold() == "librarypath":
            if (
                not _ENTRY.fullmatch(relative)
                or not relative.casefold().startswith("namespace_")
                or value.kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ)
            ):
                raise _invalid("Unsupported Winsock namespace catalog format.")
            raw = value.data
        if raw is not None:
            # Expand only the approved spelling, never ambient environment values.
            match = re.fullmatch(r"%SystemRoot%\\system32\\([a-z0-9_]+\.dll)", raw, re.IGNORECASE)
            name = match[1].casefold() if match else None
            if name is None:
                for candidate in _PROVIDERS:
                    if raw.casefold() == (str(system_directory) + "\\" + candidate).casefold():
                        name = candidate
                        break
            if name not in _PROVIDERS:
                raise _invalid("Unreviewed external Winsock provider.")
            names.add(name)
            if _ENTRY.fullmatch(relative):
                seen.add(value.path)
    if not entries or entries != seen:
        raise _invalid("Incomplete Winsock provider catalog.")
    return tuple(sorted(names))


def _system_providers(names, policy):
    identities = []
    for name in names:
        path = checked_path(Path(policy.directory) / name)
        # Windows servicing hardlinks are legitimate for OS-owned providers.
        # Application snapshot readers intentionally reject those hardlinks.
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 32 * 1024 * 1024:
                raise _invalid("Invalid Windows provider image.")
            data = source.read(32 * 1024 * 1024 + 1)
            after = os.fstat(source.fileno())
        current = path.stat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
            or len(data) != before.st_size
        ):
            raise _invalid("Windows provider changed during preparation.")
        identities.append(
            {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        )
    return identities


def _registry_api():
    api = ctypes.WinDLL("advapi32", use_last_error = True, winmode = 0x800)
    api.RegLoadAppKeyW.argtypes = [W.LPCWSTR, ctypes.POINTER(W.HKEY), W.DWORD, W.DWORD, W.DWORD]
    api.RegLoadAppKeyW.restype = W.LONG
    api.RegCloseKey.argtypes = [W.HKEY]
    api.RegCloseKey.restype = W.LONG
    return api


class PrivateCatalog:
    """Owns the broker query root and directory; target hive handles must die first."""

    def __init__(self, directory, api):
        self.directory = directory
        self.hive_path = directory / "winsock.hiv"
        self.metadata = None
        self._binding_digest = None
        self._api = api
        self._root = None
        self._extra_roots = []
        self.identity = secrets.token_bytes(32)
        info = directory.stat()
        self._directory_identity = (info.st_dev, info.st_ino)

    @property
    def binding_digest(self):
        """Stable preparation identity; readable after cleanup for probe evidence."""
        if self._binding_digest is None:
            raise _invalid("Private catalog binding is not prepared.")
        return self._binding_digest

    @property
    def query_root_handle(self):
        if self._root is None or self.metadata is None:
            raise _invalid("Private catalog is not prepared or is closed.")
        return self._root

    def release_registry_handles(self):
        """Release worker-local roots; the invocation reservation owns the files."""
        while self._extra_roots:
            status = self._api.RegCloseKey(W.HKEY(self._extra_roots[-1]))
            if status:
                raise OSError(status, "RegCloseKey(private catalog pending root)")
            self._extra_roots.pop()
        if self._root is not None:
            status = self._api.RegCloseKey(W.HKEY(self._root))
            if status:
                raise OSError(status, "RegCloseKey(private catalog)")
            self._root = None

    def close(self):
        self.release_registry_handles()
        if not self.directory.exists():
            return
        checked_path(self.directory)
        info = self.directory.stat()
        if (info.st_dev, info.st_ino) != self._directory_identity:
            raise _invalid("Private catalog directory was replaced.")
        children = list(self.directory.iterdir())
        # No recursive cleanup and no deletion of unexpected/replaced objects.
        for child in children:
            info = child.lstat()
            if (
                child.name not in {"winsock.hiv", "winsock.hiv.LOG1", "winsock.hiv.LOG2"}
                or not stat.S_ISREG(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise _invalid("Unexpected private catalog cleanup entry.")
        for child in children:
            child.unlink()
        self.directory.rmdir()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@dataclass(frozen = True)
class CatalogReceipt:
    """Validated worker data; file cleanup belongs to the invocation reservation."""

    hive_path: Path
    identity: bytes
    binding_digest: str


def catalog_receipt(catalog):
    return {
        "directory": str(catalog.directory),
        "identity": catalog.identity.hex(),
        "binding_digest": catalog.binding_digest,
    }


def adopt_catalog(value, temporary):
    """Decode only invocation-scoped data, without reopening prepared files."""
    if type(value) is not dict or set(value) != {"directory", "identity", "binding_digest"}:
        raise _invalid("Invalid private catalog handoff fields.")
    if any(
        type(value[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value[key])
        for key in ("identity", "binding_digest")
    ):
        raise _invalid("Invalid private catalog handoff identity.")
    directory = value["directory"]
    if type(directory) is not str or len(directory) > 1024 or "\0" in directory:
        raise _invalid("Invalid private catalog handoff directory.")
    path = Path(directory)
    if path.parent != Path(temporary) or not re.fullmatch(
        r"private-winsock-[0-9a-f]{32}", path.name
    ):
        raise _invalid("Private catalog handoff escaped its invocation.")
    return CatalogReceipt(
        path / "winsock.hiv", bytes.fromhex(value["identity"]), value["binding_digest"]
    )


def prepare_private_catalog(
    parent_directory: Path,
    *,
    package_sid: str | None = None,
    bounds = CatalogBounds(),
    expected_binding_digest: str | None = None,
) -> PrivateCatalog:
    """Snapshot fixed host trees as data and return a broker-owned private root.

    If supplied, package_sid gets read/write/traverse access to this owned
    directory and its hive files, without delete or ACL-changing rights.
    RegLoadAppKeyW requires writable backing-file access on the supported lane
    even for a query-only root. Targets open hive_path using RegLoadAppKeyW,
    KEY_QUERY_VALUE and options=0; verify binary CATALOG_MARKER against identity.
    Never transfer raw APPKEY handles across processes with DuplicateHandle.
    expected_binding_digest binds a new snapshot to prior broker qualification
    evidence before creating a hive. It excludes invocation SID and random marker;
    the child must still validate its exact per-invocation marker. Provider hashes
    describe inspected files, not a pin on DLL bytes later mapped by the OS loader.
    Returned metadata is preparation evidence, never a Required-mode readiness gate.
    """
    if os.name != "nt":
        raise _invalid("Private catalogs require Windows.")
    import winreg
    from .native_plan import windows_loader_policy

    if package_sid is not None and not re.fullmatch(r"S-1-15-2(?:-[0-9]{1,10}){7}", package_sid):
        raise _invalid("Invalid private catalog AppContainer SID.")
    if expected_binding_digest is not None and (
        not isinstance(expected_binding_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_binding_digest)
    ):
        raise _invalid("Invalid private catalog qualification binding.")
    records, counts = _snapshot(bounds)
    policy = windows_loader_policy()
    names = _provider_names(records, policy.directory)
    providers = _system_providers(names, policy)
    encoded = json.dumps(
        [_encoded(item) for item in records], separators = (",", ":"), ensure_ascii = True
    ).encode()
    evidence = {
        "schema": 1,
        "catalog_format": "windows-packed-888-path260-v1",
        "catalog_sha256": hashlib.sha256(encoded).hexdigest(),
        **counts,
        "providers": providers,
        "architecture": policy.architecture,
        "windows_version": list(policy.windows_version),
        "root_access": 1,
        "qualified": False,
        "private_subkeys_read_only": False,
        "private_catalog_writable": package_sid is not None,
        "handoff": "RegLoadAppKeyW-path-and-identity",
        "package_sid": package_sid,
    }
    binding = {key: value for key, value in evidence.items() if key != "package_sid"}
    evidence["binding_digest"] = hashlib.sha256(
        json.dumps(binding, sort_keys = True, separators = (",", ":")).encode()
    ).hexdigest()
    if expected_binding_digest is not None and not secrets.compare_digest(
        expected_binding_digest, evidence["binding_digest"]
    ):
        raise _invalid("Private catalog qualification binding changed.")
    evidence["digest"] = hashlib.sha256(
        json.dumps(evidence, sort_keys = True, separators = (",", ":")).encode()
    ).hexdigest()
    parent = checked_path(parent_directory)
    directory = parent / ("private-winsock-" + secrets.token_hex(16))
    files, api = native_files(), _registry_api()
    files.mkdir(directory)
    owner = PrivateCatalog(directory, api)
    root = W.HKEY()
    try:
        # Apply only to the newly owned directory before any hive file exists.
        directory_handle = files.open(directory, directory = True, write_dac = True)
        try:
            sddl = files.private_sddl.replace("(A;;", "(A;OICI;")
            if package_sid:
                sddl += f"(A;OICI;FRFWFX;;;{package_sid})"
            files.set_owned_dacl(directory_handle, sddl)
        finally:
            files.kernel.CloseHandle(directory_handle)
        status = api.RegLoadAppKeyW(
            str(owner.hive_path), ctypes.byref(root), winreg.KEY_ALL_ACCESS, 0, 0
        )
        if status:
            raise OSError(status, "RegLoadAppKeyW(private catalog)")
        owner._root = root.value
        winreg.SetValueEx(root.value, CATALOG_MARKER, 0, winreg.REG_BINARY, owner.identity)
        for value in records:
            with winreg.CreateKeyEx(root.value, value.path, 0, winreg.KEY_ALL_ACCESS) as target:
                if value.kind != -1:
                    data = list(value.data) if isinstance(value.data, tuple) else value.data
                    winreg.SetValueEx(target, value.name, 0, value.kind, data)
        with winreg.OpenKey(root.value, "", 0, winreg.KEY_QUERY_VALUE) as query:
            query_value = query.Detach()
            owner._extra_roots.append(query_value)
        status = api.RegCloseKey(root)
        if status:
            raise OSError(status, "RegCloseKey(private catalog writer)")
        owner._root = query_value
        owner._extra_roots.remove(query_value)
        owner._binding_digest = evidence["binding_digest"]
        owner.metadata = evidence
        return owner
    except BaseException as original:
        try:
            owner.close()
        except BaseException as cleanup_error:
            # Preserve retryable ownership rather than claiming failed cleanup.
            cleanup_error.catalog_owner = owner
            raise cleanup_error from original
        raise
