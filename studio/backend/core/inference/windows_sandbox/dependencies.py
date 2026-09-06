# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static native-image inventory. No target DLL or interpreter is ever loaded.

This inventory is not an approval to execute a native initializer before the
drop. The protected content store and reviewed startup profile own that decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat

from .profiles import WindowsRuntimeError

PEFILE_VERSION = "2024.8.26"
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_IMPORTS = 512
_MACHINES = {0x8664: "x64", 0xAA64: "arm64", 0x14C: "x86"}
_DLL_NAME = re.compile(r"[a-zA-Z0-9_.+\-]{1,200}\.(?:dll|pyd)", re.IGNORECASE)


@dataclass(frozen = True)
class FileIdentity:
    path: str
    sha256: str
    size: int
    device: int
    inode: int


@dataclass(frozen = True)
class NativeImage:
    file: FileIdentity
    architecture: str
    imports: tuple[str, ...]
    delay_imports: tuple[str, ...]
    file_version: tuple[int, int, int, int] | None
    product_version: str | None
    warnings: tuple[str, ...]


def checked_path(path: str | Path) -> Path:
    """Reject aliases before resolving them; never turn a junction into authority."""
    path = Path(path)
    if not path.is_absolute() or any(char in str(path) for char in ("\0", "\n", "\r")):
        raise WindowsRuntimeError("WINDOWS_SANDBOX_RUNTIME_INVALID", "Expected an absolute path.")
    if os.name == "nt" and (str(path).startswith(("\\\\", "//")) or ":" in str(path)[2:]):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "Network/device/stream paths are not runtime roots."
        )
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", f"Runtime reparse point: {part}"
            )
    return path.resolve(strict = True)


def read_regular_file(path: str | Path, *, limit: int) -> tuple[FileIdentity, bytes]:
    path = checked_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID",
            f"Runtime file is special, hardlinked or too large: {path}",
        )
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_CHANGED", f"Runtime file was replaced: {path}"
            )
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    current = checked_path(path).stat()
    # Windows stat/fstat in some CPython releases expose different ctime meanings.
    # Compare those timestamps only within the same API, never across the pair.
    fields = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)
    if (
        len(data) != before.st_size
        or len(data) > limit
        or fields(before) != fields(after)
        or fields(after) != fields(current)
        or before.st_ctime_ns != current.st_ctime_ns
        or opened.st_ctime_ns != after.st_ctime_ns
    ):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_CHANGED", f"Runtime file changed while reading: {path}"
        )
    return FileIdentity(
        str(path), hashlib.sha256(data).hexdigest(), len(data), before.st_dev, before.st_ino
    ), data


def _import_name(value: bytes) -> str:
    try:
        name = value.decode("ascii")
    except (AttributeError, UnicodeError) as exc:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PE_INVALID", "Non-ASCII dependency name."
        ) from exc
    if not _DLL_NAME.fullmatch(name) or ".." in name:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PE_INVALID", "A dependency contains a path or invalid name."
        )
    return name.lower()


def inspect_native_image(path: str | Path) -> NativeImage:
    identity, data = read_regular_file(path, limit = MAX_IMAGE_BYTES)
    try:
        import pefile
    except ImportError as exc:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_MISSING",
            f"Static runtime analysis requires pefile=={PEFILE_VERSION}.",
        ) from exc
    if pefile.__version__ != PEFILE_VERSION:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROTOCOL_MISMATCH",
            "The PE parser version does not match this profile.",
        )
    try:
        with pefile.PE(data = data, fast_load = True) as image:
            if not 0 < image.FILE_HEADER.NumberOfSections <= 96:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PE_INVALID", "PE section limit exceeded."
                )
            architecture = _MACHINES.get(image.FILE_HEADER.Machine)
            if architecture is None:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_ABI_UNSUPPORTED", "Unknown PE architecture."
                )
            expected_magic = 0x10B if architecture == "x86" else 0x20B
            if image.OPTIONAL_HEADER.Magic != expected_magic:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PE_INVALID", "PE architecture/header mismatch."
                )
            image.parse_data_directories(directories = [1, 2, 13], import_dllnames_only = True)
            groups = []
            for index, attribute in (
                (1, "DIRECTORY_ENTRY_IMPORT"),
                (13, "DIRECTORY_ENTRY_DELAY_IMPORT"),
            ):
                entries = getattr(image, attribute, ())
                directory = image.OPTIONAL_HEADER.DATA_DIRECTORY[index]
                if directory.VirtualAddress and not entries:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_PE_INVALID", "An import directory could not be parsed."
                    )
                if len(entries) > MAX_IMPORTS:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_PE_INVALID", "PE dependency limit exceeded."
                    )
                groups.append(tuple(sorted({_import_name(entry.dll) for entry in entries})))
            versions = set()
            for version in getattr(image, "VS_FIXEDFILEINFO", ()):
                versions.add(
                    (
                        version.FileVersionMS >> 16,
                        version.FileVersionMS & 0xFFFF,
                        version.FileVersionLS >> 16,
                        version.FileVersionLS & 0xFFFF,
                    )
                )
            if len(versions) > 1:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PE_INVALID", "Conflicting PE version resources."
                )
            product_versions = set()
            for group in getattr(image, "FileInfo", ()):
                for entry in group:
                    for table in getattr(entry, "StringTable", ()):
                        value = table.entries.get(b"ProductVersion")
                        if value is not None:
                            if len(value) > 64:
                                raise WindowsRuntimeError(
                                    "WINDOWS_SANDBOX_PE_INVALID", "Oversized product version."
                                )
                            product_versions.add(value.decode("ascii"))
            if len(product_versions) > 1:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PE_INVALID", "Conflicting product versions."
                )
            warnings = tuple(image.get_warnings())
            if warnings:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PE_INVALID", f"PE parser reported: {warnings[0]}"
                )
            return NativeImage(
                identity,
                architecture,
                groups[0],
                groups[1],
                next(iter(versions), None),
                next(iter(product_versions), None),
                warnings,
            )
    except (pefile.PEFormatError, IndexError, AttributeError, ValueError, TypeError) as exc:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PE_INVALID", "Malformed native image.") from exc
