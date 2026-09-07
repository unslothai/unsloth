# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Admission of Studio's already trusted CPython core, not discovered packages.

The running Studio installation is the trust root. This is not publisher
authentication or a repair mechanism for an already compromised host. Tool and
provider data cannot choose a different startup interpreter or approve a DLL.
The preparation owner runs this static work under a separate process deadline.
"""

from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .content import SnapshotFile
from .dependencies import checked_path, read_regular_file
from .native_plan import DependencyPlan, ScanBounds, build_dependency_plan, windows_loader_policy
from .profiles import PYTHON_PROFILE, WindowsRuntimeError
from .runtime import RuntimeDescriptor, discover_runtime, require_profile_runtime


def _denied(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_RUNTIME_UNTRUSTED", message)


def _loaded_python_path(*, canonical = True):
    # sys.dllhandle is CPython's documented Windows runtime handle. Query it;
    # never LoadLibrary a path from a descriptor to establish this identity.
    handle = getattr(sys, "dllhandle", None)
    if (
        sys.platform != "win32"
        or sys.implementation.name != "cpython"
        or type(handle) is not int
        or handle <= 0
    ):
        raise _denied("The running Windows CPython identity cannot be established.")
    kernel = ctypes.WinDLL("kernel32", use_last_error = True, winmode = 0x800)
    query = kernel.GetModuleFileNameW
    query.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    query.restype = wintypes.DWORD
    value = ctypes.create_unicode_buffer(32768)
    length = query(handle, value, len(value))
    if not 0 < length < len(value):
        raise _denied("The loaded Python DLL path is unavailable or truncated.")
    return checked_path(value.value) if canonical else value.value


@dataclass(frozen = True)
class _BrokerRuntime:
    executable: str
    loaded_dll: str
    prefix: str
    base_prefix: str
    version: tuple[int, int, int]
    module_names: tuple[str, ...]
    pid: int


def _capture_broker_runtime():
    # No runtime traversal on the owning thread: the hard-deadline worker does
    # filesystem canonicalization and inventory. These are running-process facts.
    return _BrokerRuntime(
        sys.executable,
        _loaded_python_path(canonical = False),
        sys.prefix,
        sys.base_prefix,
        tuple(sys.version_info[:3]),
        tuple(sorted(sys.stdlib_module_names)),
        os.getpid(),
    )


def _scanner_executable(broker):
    # Windows venv python.exe is a redirector that creates another process.
    # The static worker uses the running base interpreter, not that redirector;
    # discovery still describes the selected venv and keeps its package paths.
    return str(Path(broker.base_prefix) / "python.exe")


@dataclass(frozen = True)
class AdmittedCore:
    runtime: RuntimeDescriptor
    dependencies: DependencyPlan
    files: tuple[SnapshotFile, ...]
    broker_pid: int
    origin: str = "running_studio_cpython_core_v1"

    @property
    def digest(self):
        # PID is evidence of this admission call, not a content-cache key.
        value = {
            "origin": self.origin,
            "runtime": self.runtime.digest,
            "dependencies": self.dependencies.digest,
            "profile": PYTHON_PROFILE.digest,
            "files": [asdict(item) for item in self.files],
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys = True, separators = (",", ":")).encode()
        ).hexdigest()


def _stdlib_inventory(
    base: Path, module_names: frozenset[str], bounds: ScanBounds, deadline: float
):
    """Inventory only declared top-level stdlib modules, without import or hooks."""
    if (
        type(module_names) is not frozenset
        or not module_names
        or any(
            type(name) is not str or not name.isascii() or not name.isidentifier()
            for name in module_names
        )
    ):
        raise _denied("The broker's compiled standard-library inventory is invalid.")
    root = checked_path(base / "Lib")
    if not root.is_dir():
        raise _denied("This CPython core requires an ordinary source standard library.")
    result, names = [], set()
    count = total = 0

    def visit(directory, depth):
        nonlocal count, total
        if depth > min(bounds.depth, 12):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_SCAN_LIMIT", "Standard-library depth exceeded."
            )
        # Do not materialize an unbounded directory listing before counting it.
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > bounds.entries or time.monotonic() >= deadline:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_SCAN_LIMIT", "Standard-library scan limit exceeded."
                    )
                name = entry.name
                if name in ("site-packages", "__pycache__", "test", "tests"):
                    continue
                if depth == 0 and name.removesuffix(".py") not in module_names:
                    continue
                if entry.is_dir(follow_symlinks = False):
                    path = checked_path(entry.path)
                    visit(path, depth + 1)
                elif name.endswith(".py"):
                    identity, _ = read_regular_file(entry.path, limit = 4 * 1024 * 1024)
                    relative = "runtime/Lib/" + Path(identity.path).relative_to(root).as_posix()
                    total += identity.size
                    if relative.casefold() in names or total > bounds.bytes:
                        raise WindowsRuntimeError(
                            "WINDOWS_SANDBOX_SCAN_LIMIT",
                            "Standard-library file collision or byte limit.",
                        )
                    names.add(relative.casefold())
                    result.append(SnapshotFile(identity, relative))
                elif entry.is_symlink():
                    # An admitted package cannot hide an aliased child directory.
                    raise _denied("The standard library contains a symlink.")

    visit(root, 0)
    if "runtime/lib/encodings/__init__.py" not in names:
        raise _denied("The declared standard library has no encodings package.")
    return tuple(sorted(result, key = lambda item: item.relative_path))


def admit_studio_runtime(
    selected_executable: str, *, bounds: ScanBounds = ScanBounds()
) -> AdmittedCore:
    """Own the trust decision; accept no serialized descriptor or approval flag.

    Normal Python tools select sys.executable. Different installations can be
    discovered/probed as payload-only, but require a separately reviewed admission
    policy before any of their initializers may run in the temporary-token phase.
    """
    return _admit_broker_runtime(selected_executable, _capture_broker_runtime(), bounds)


def _admit_broker_runtime(
    selected_executable,
    broker,
    bounds,
    *,
    include_packages = False,
):
    if any(type(value) is not int or value <= 0 for value in asdict(bounds).values()):
        raise WindowsRuntimeError("WINDOWS_SANDBOX_SCAN_LIMIT", "Invalid admission bounds.")
    deadline = time.monotonic() + bounds.seconds
    loaded = checked_path(broker.loaded_dll)
    executable = checked_path(broker.executable)
    # -S intentionally suppresses venv site processing on 3.11/3.12. The worker
    # retains parent prefixes, but must prove its executable/DLL/version match.
    if (
        checked_path(sys.executable) not in (executable, checked_path(_scanner_executable(broker)))
        or _loaded_python_path() != loaded
        or tuple(sys.version_info[:3]) != broker.version
        or tuple(sorted(sys.stdlib_module_names)) != broker.module_names
    ):
        raise _denied("The preparation worker does not match the running Studio runtime.")
    if checked_path(selected_executable) != executable:
        raise _denied("Only Studio's running interpreter is admitted for core startup.")
    runtime = discover_runtime(str(executable))
    require_profile_runtime(runtime)
    if (
        runtime.version != broker.version
        or Path(runtime.runtime_dll.file.path) != loaded
        or Path(runtime.prefix) != checked_path(broker.prefix)
        or Path(runtime.base_prefix) != checked_path(broker.base_prefix)
    ):
        raise _denied("Discovered runtime metadata does not match the running Studio process.")
    identity, _ = read_regular_file(loaded, limit = 128 * 1024 * 1024)
    if identity != runtime.runtime_dll.file:
        raise _denied("The running Python DLL changed during admission.")
    base = Path(runtime.base_prefix)
    native = checked_path(base / "DLLs")
    module_names = frozenset(broker.module_names)
    features = [runtime.runtime_dll.file.path]
    for name in PYTHON_PROFILE.native_features:
        candidate = native / (name + ".pyd")
        if name not in module_names:
            raise _denied("A declared native feature is not a CPython standard-library module.")
        if candidate.exists():
            features.append(str(checked_path(candidate)))
    dependencies = build_dependency_plan(
        tuple(features),
        (str(base), str(native)),
        architecture = runtime.architecture,
        system = windows_loader_policy(),
        bounds = bounds,
    )
    files = list(_stdlib_inventory(base, module_names, bounds, deadline))
    for image in dependencies.images:
        path = Path(image.file.path)
        if path.parent not in (base, native):
            raise _denied("A core startup dependency escaped the selected runtime.")
        files.append(SnapshotFile(image.file, "runtime/" + path.relative_to(base).as_posix()))
    if time.monotonic() >= deadline or sum(item.source.size for item in files) > bounds.bytes:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_SCAN_LIMIT", "Core admission exceeded its bound."
        )
    # Snapshot publication rechecks every identity under source pins before copy.
    # This returned object itself is neither an immutable snapshot nor qualification.
    if include_packages:
        from .package_snapshot import inventory_package_files

        # Selected packages are copied as payload data. They are never added to
        # the CPython initializer graph or executed by the admission worker.
        files.extend(inventory_package_files(runtime.package_paths))
    return AdmittedCore(runtime, dependencies, tuple(files), broker.pid)
