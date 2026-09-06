# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Describe the explicitly selected Windows Python without executing it or hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from .dependencies import (
    FileIdentity,
    NativeImage,
    checked_path,
    inspect_native_image,
    read_regular_file,
)
from .profiles import PYTHON_PROFILE, WindowsRuntimeError, select_abi_adapter

_PYTHON_DLL = re.compile(r"python(3)(\d+)(t?)(_d)?\.dll", re.IGNORECASE)
_MAX_CONFIG_BYTES = 64 * 1024


@dataclass(frozen = True)
class RuntimeDescriptor:
    schema_version: int
    kind: str
    implementation: str
    version: tuple[int, int, int]
    architecture: str
    debug: bool
    free_threaded: bool
    executable: NativeImage
    runtime_dll: NativeImage
    prefix: str
    base_prefix: str
    stdlib_paths: tuple[str, ...]
    package_paths: tuple[str, ...]
    configuration_files: tuple[FileIdentity, ...]
    trust_classification: str = "payload_only"

    @property
    def digest(self) -> str:
        # Content hashes, not size/mtime alone. The content store must separately
        # bind every stdlib/package dependency it admits to a runtime generation.
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys = True, separators = (",", ":")).encode()
        ).hexdigest()


def _read_venv_config(path: Path) -> tuple[FileIdentity, dict[str, str]]:
    identity, data = read_regular_file(path, limit = _MAX_CONFIG_BYTES)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "pyvenv.cfg must be UTF-8."
        ) from exc
    values = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip().lower(), value.strip()
        if not separator or not key or key in values or "\0" in value:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "Malformed or duplicate pyvenv.cfg entry."
            )
        values[key] = value
    return identity, values


def discover_runtime(executable: str) -> RuntimeDescriptor:
    """Inventory only. The returned object has no authority to enter startup."""
    exe_path = checked_path(executable)
    exe = inspect_native_image(exe_path)
    prefix = exe_path.parent
    config_path = prefix / "pyvenv.cfg"
    if prefix.name.lower() == "scripts" and (prefix.parent / "pyvenv.cfg").exists():
        prefix = prefix.parent
        config_path = prefix / "pyvenv.cfg"
    configs = []
    base = prefix
    kind = "cpython"
    values = {}
    if config_path.exists():
        config_file, values = _read_venv_config(config_path)
        configs.append(config_file)
        home = values.get("home")
        if not home:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "pyvenv.cfg has no absolute base home."
            )
        base = checked_path(home)
        if not base.is_dir():
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "The venv base is not a directory."
            )
        kind = "venv"
    if (base / "conda-meta").exists():
        kind = "conda"
    if base.parent == base or prefix.parent == prefix:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "Drive roots cannot be runtime roots."
        )
    # Read only one directory; never a recursive scan of PATH or the user's home.
    candidates = []
    for index, item in enumerate(base.iterdir()):
        if index >= 4096:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_RUNTIME_INVALID", "Runtime root entry limit exceeded."
            )
        if _PYTHON_DLL.fullmatch(item.name):
            candidates.append(item)
    imported = {
        _PYTHON_DLL.fullmatch(name).group(0).lower()
        for name in exe.imports
        if _PYTHON_DLL.fullmatch(name)
    }
    if imported:
        candidates = [path for path in candidates if path.name.lower() in imported]
    if len(candidates) != 1:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID",
            "The selected interpreter has no unambiguous runtime DLL.",
        )
    dll_path = candidates[0]
    dll = inspect_native_image(dll_path)
    match = _PYTHON_DLL.fullmatch(dll_path.name)
    major, minor = int(match[1]), int(match[2])
    if (
        dll.architecture != exe.architecture
        or dll.file_version is None
        or dll.file_version[:2] != (major, minor)
    ):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "Interpreter and DLL architecture/version disagree."
        )
    # CPython's numeric FileVersion encodes micro/release level together (e.g.
    # 3.12.10150.1013 is 3.12.10). ProductVersion carries the Python version;
    # the native helper must independently confirm it with Py_GetVersion.
    product = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", dll.product_version or "")
    if product is None:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_ABI_UNSUPPORTED",
            "The runtime has no unambiguous release Python version.",
        )
    version = tuple(map(int, product.groups()))
    if version[:2] != (major, minor):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "Python version resources disagree."
        )
    if "version" in values and values["version"] != ".".join(map(str, version)):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_CHANGED", "The venv version does not match its base DLL."
        )
    stdlib = []
    for candidate in (base / f"python{major}{minor}.zip", base / "Lib", base / "DLLs"):
        if candidate.exists():
            stdlib.append(str(checked_path(candidate)))
    for candidate in (exe_path.with_suffix("._pth"), dll_path.with_suffix("._pth")):
        if candidate.exists():
            identity, _ = read_regular_file(candidate, limit = _MAX_CONFIG_BYTES)
            if identity not in configs:
                configs.append(identity)
            kind = "embedded"
    if not stdlib:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_INVALID", "No standard-library location was discovered."
        )
    packages = []
    candidate = prefix / "Lib" / "site-packages"
    if candidate.exists():
        packages.append(str(checked_path(candidate)))
    if kind == "venv" and values.get("include-system-site-packages", "false").lower() == "true":
        candidate = base / "Lib" / "site-packages"
        if candidate.exists():
            packages.append(str(checked_path(candidate)))
    return RuntimeDescriptor(
        1,
        kind,
        "cpython",
        version,
        dll.architecture,
        bool(match[4]),
        bool(match[3]),
        exe,
        dll,
        str(prefix),
        str(base),
        tuple(stdlib),
        tuple(packages),
        tuple(configs),
    )


def require_profile_runtime(runtime: RuntimeDescriptor) -> None:
    """Check declared compatibility, separately from trust and live qualification."""
    if runtime.kind not in PYTHON_PROFILE.runtime_families:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_LAYOUT_UNSUPPORTED",
            f"The {runtime.kind} layout has not been qualified for the Python bootstrap.",
        )
    select_abi_adapter(
        implementation = runtime.implementation,
        version = runtime.version,
        architecture = runtime.architecture,
        debug = runtime.debug,
        free_threaded = runtime.free_threaded,
    )
