# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static dependency plans; neither DLL discovery nor ordering grants startup trust.

Windows system modules/API sets terminate the application graph at the OS trust
boundary. Application DLLs are resolved only in explicitly selected directories,
never PATH, the current directory, or a recursive package/home search. A native
probe must still exercise dynamic loads and the eventual restricted loader policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

from .dependencies import NativeImage, _import_name, checked_path, inspect_native_image
from .profiles import WindowsRuntimeError

_API_SET = re.compile(r"(?:api|ext)-[a-z0-9-]+-l[0-9]+-[0-9]+-[0-9]+\.dll")


@dataclass(frozen = True)
class ScanBounds:
    directories: int = 32
    entries: int = 16384
    images: int = 256
    edges: int = 8192
    depth: int = 64
    bytes: int = 1024 * 1024 * 1024
    seconds: int = 30


@dataclass(frozen = True)
class SystemLoaderPolicy:
    directory: str
    known_dlls: tuple[str, ...]
    windows_version: tuple[int, int, int]
    architecture: str


@dataclass(frozen = True)
class DependencyEdge:
    source: str
    name: str
    kind: str
    target: str
    delayed: bool


@dataclass(frozen = True)
class DependencyPlan:
    schema_version: int
    feature_roots: tuple[str, ...]
    search_directories: tuple[str, ...]
    architecture: str
    images: tuple[NativeImage, ...]
    edges: tuple[DependencyEdge, ...]
    ordered_loads: tuple[str, ...]
    system: SystemLoaderPolicy
    bounds: ScanBounds
    dynamic_dependencies: str = "requires_isolated_probe"
    trust_classification: str = "payload_only"

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys = True, separators = (",", ":")).encode()
        ).hexdigest()


def windows_loader_policy() -> SystemLoaderPolicy:
    """Read OS metadata, not environment-supplied SystemRoot or a DLL directory."""
    if sys.platform != "win32" or sys.maxsize <= 2**32:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_ABI_UNSUPPORTED", "A native x64 broker is required."
        )
    import ctypes
    from ctypes import wintypes
    import winreg

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True, winmode = 0x800)
    _require_native_x64(kernel32)
    get_directory = kernel32.GetSystemDirectoryW
    get_directory.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    get_directory.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_directory(buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "Cannot read system directory."
        )
    directory = checked_path(buffer.value)
    known = set()
    key_name = r"SYSTEM\CurrentControlSet\Control\Session Manager\KnownDLLs"
    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, key_name, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    ) as key:
        _, values, _ = winreg.QueryInfoKey(key)
        if values > 512:
            raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "KnownDLLs limit exceeded.")
        for index in range(values):
            _, value, kind = winreg.EnumValue(key, index)
            if kind == winreg.REG_SZ and value.lower().endswith(".dll"):
                known.add(_import_name(value.encode("ascii")))
    version = sys.getwindowsversion()
    return SystemLoaderPolicy(str(directory), tuple(sorted(known)), version[:3], "x64")


def _require_native_x64(kernel32):
    """Use OS process/native machine identities, never environment architecture."""
    import ctypes
    from ctypes import wintypes

    try:
        query = kernel32.IsWow64Process2
    except AttributeError as error:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_ABI_UNSUPPORTED", "Native architecture query unavailable."
        ) from error
    query.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT),
        ctypes.POINTER(wintypes.USHORT),
    ]
    query.restype = wintypes.BOOL
    process_machine, native_machine = wintypes.USHORT(), wintypes.USHORT()
    if (
        not query(wintypes.HANDLE(-1), ctypes.byref(process_machine), ctypes.byref(native_machine))
        or process_machine.value != 0
        or native_machine.value != 0x8664
    ):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_ABI_UNSUPPORTED", "A native x64 process and OS are required."
        )


def api_set_implemented(name: str) -> bool:
    """Query the OS contract map; no untrusted DLL is loaded by this check."""
    import ctypes
    from ctypes import wintypes

    if not _API_SET.fullmatch(name):
        return False
    try:
        api = ctypes.WinDLL("api-ms-win-core-apiquery-l2-1-0.dll", winmode = 0x800)
        query = api.IsApiSetImplemented
        query.argtypes = [ctypes.c_char_p]
        query.restype = wintypes.BOOL
        return bool(query(name.removesuffix(".dll").encode("ascii")))
    except (OSError, AttributeError) as exc:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "API-set availability cannot be established."
        ) from exc


def build_dependency_plan(
    feature_roots: tuple[str, ...],
    search_directories: tuple[str, ...],
    *,
    architecture: str,
    system: SystemLoaderPolicy,
    bounds: ScanBounds = ScanBounds(),
) -> DependencyPlan:
    """Reject ambiguity/cycles rather than guessing the loader's eventual choice.

    Time checks bound work between parser calls, not a stuck parser call. A hard
    parser deadline must be enforced by the owning broker before production use.
    """
    if any(type(value) is not int or value <= 0 for value in asdict(bounds).values()):
        raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "Invalid scan bounds.")
    if architecture != system.architecture or architecture != "x64":
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_ABI_UNSUPPORTED", "Loader architecture mismatch."
        )
    if not feature_roots or len(feature_roots) > bounds.images or not search_directories:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "Invalid feature root count.")
    if len(search_directories) > bounds.directories:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "Directory limit exceeded.")
    deadline = time.monotonic() + bounds.seconds

    def check_time():
        if time.monotonic() >= deadline:
            raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "Dependency scan timed out.")

    directories = tuple(sorted({checked_path(path) for path in search_directories}, key = str))
    system_directory = checked_path(system.directory)
    candidates: dict[str, list[Path]] = {}
    entry_count = 0
    for directory in directories:
        if directory == system_directory or directory.parent == directory or not directory.is_dir():
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "Invalid application DLL directory."
            )
        for path in directory.iterdir():
            check_time()
            entry_count += 1
            if entry_count > bounds.entries:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_SCAN_LIMIT", "Directory entry limit exceeded."
                )
            if path.suffix.lower() in (".dll", ".pyd"):
                name = _import_name(path.name.encode("ascii"))
                candidates.setdefault(name, []).append(path)

    roots = tuple(sorted({checked_path(path) for path in feature_roots}, key = str))
    for path in roots:
        if path.parent not in directories:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "Feature outside selected directories."
            )
    images: dict[str, NativeImage] = {}
    edges = []
    ordered = []
    visiting = set()
    names: dict[str, str] = {}
    byte_count = 0
    contracts = {}

    def system_file(name):
        candidate = system_directory / name
        if not candidate.exists():
            return None
        candidate = checked_path(candidate)
        # Windows component-store hardlinks are expected only in the OS boundary.
        if not stat.S_ISREG(candidate.stat().st_mode):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "Special system dependency."
            )
        return str(candidate)

    def resolve(name):
        matches = candidates.get(name, [])
        if _API_SET.fullmatch(name):
            if matches:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_DEPENDENCY_COLLISION", f"API-set shadow: {name}"
                )
            if name not in contracts:
                contracts[name] = api_set_implemented(name)
            if not contracts[name]:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_DEPENDENCY_MISSING", f"API set unavailable: {name}"
                )
            return "api_set", name
        os_path = system_file(name)
        if name in system.known_dlls:
            if matches:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_DEPENDENCY_COLLISION", f"KnownDLL shadow: {name}"
                )
            if os_path is None:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_DEPENDENCY_MISSING", f"KnownDLL missing: {name}"
                )
            return "known_dll", os_path
        # DLL_LOAD_DIR/USER_DIRS precede SYSTEM32 for non-KnownDLLs. A single
        # explicit application candidate is therefore not ambiguous with an OS
        # fallback (notably CPython's bundled VC runtime). Multiple application
        # directories have unspecified relative order and must not be guessed.
        if len(matches) > 1:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_DEPENDENCY_COLLISION", f"Ambiguous dependency: {name}"
            )
        if matches:
            return "application", str(checked_path(matches[0]))
        if os_path:
            return "system", os_path
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_DEPENDENCY_MISSING", f"Unresolved dependency: {name}"
        )

    def visit(path: Path, depth: int):
        nonlocal byte_count
        check_time()
        key = str(path)
        if key in visiting:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_DEPENDENCY_CYCLE", f"Cyclic dependency: {path.name}"
            )
        if key in images:
            return
        if depth > bounds.depth or len(images) + len(visiting) >= bounds.images:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_SCAN_LIMIT", "Dependency graph limit exceeded."
            )
        name = path.name.lower()
        if name in names and names[name] != key:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_DEPENDENCY_COLLISION", f"Duplicate loaded name: {name}"
            )
        names[name] = key
        for suffix in (".local", ".manifest", ".1.manifest", ".2.manifest", ".3.manifest"):
            if os.path.lexists(str(path) + suffix):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_MANIFEST_UNSUPPORTED",
                    "External loader redirection is unsupported.",
                )
        size = path.stat().st_size
        if size > bounds.bytes - byte_count:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_SCAN_LIMIT", "Dependency byte limit exceeded."
            )
        image = inspect_native_image(path)
        check_time()
        byte_count += image.file.size
        if byte_count > bounds.bytes:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_SCAN_LIMIT", "Dependency byte limit exceeded."
            )
        if image.architecture != architecture:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_ABI_UNSUPPORTED", f"Dependency architecture mismatch: {name}"
            )
        if any(manifest.loader_directives for manifest in image.manifests):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_MANIFEST_UNSUPPORTED", f"Unresolved manifest redirection: {name}"
            )
        for manifest in image.manifests:
            for assembly in manifest.system_assemblies:
                if len(edges) >= bounds.edges:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_SCAN_LIMIT", "Dependency edge limit exceeded."
                    )
                edges.append(
                    DependencyEdge(
                        key, "Microsoft.Windows.Common-Controls", "system_assembly", assembly, False
                    )
                )
        visiting.add(key)
        for delayed, imports in ((False, image.imports), (True, image.delay_imports)):
            for imported in imports:
                check_time()
                if len(edges) >= bounds.edges:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_SCAN_LIMIT", "Dependency edge limit exceeded."
                    )
                imported = _import_name(imported.encode("ascii"))
                kind, target = resolve(imported)
                edges.append(DependencyEdge(key, imported, kind, target, delayed))
                if kind == "application":
                    visit(Path(target), depth + 1)
        visiting.remove(key)
        images[key] = image
        ordered.append(key)

    for root in roots:
        visit(root, 1)
    return DependencyPlan(
        1,
        tuple(map(str, roots)),
        tuple(map(str, directories)),
        architecture,
        tuple(images[path] for path in sorted(images)),
        tuple(edges),
        tuple(ordered),
        system,
        bounds,
    )
