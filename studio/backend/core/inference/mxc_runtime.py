# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Select and verify Studio's pinned Microsoft WXC executable."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import threading

MXC_REVISION = "7dac1a952f0c9ad13f0a4cb089c4e0e8b3e0013a"
MXC_SCHEMA_VERSION = "0.8.0-alpha"
PROFILE_ID = "unsloth-mxc-windows-processcontainer-v2"
RELEASE_REPOSITORY = "microsoft/mxc"
RELEASE_TAG = "v0.8.0"
RELEASE_ASSET = "mxc-release-binaries.zip"
RELEASE_URL = "https://github.com/microsoft/mxc/releases/download/v0.8.0/mxc-release-binaries.zip"
RELEASE_ARCHIVE_SIZE = 358_007_638
RELEASE_ARCHIVE_SHA256 = "5c3a27073ba18eddf97efb4caad0f8b201c40a18d17b70f3a1e3847fb6232e3c"
RELEASE_MEMBER = "x64/wxc-exec.exe"
WXC_EXEC_SIZE = 9_478_968
WXC_EXEC_SHA256 = "6049c64723af1173c3739dc6cd6b2f33f6c021bb2832c4216233cba7f71aee9a"
# Tier 3 host preparation, elevated; never needed by the BaseContainer path.
RELEASE_HOST_PREP_MEMBER = "x64/wxc-host-prep.exe"
WXC_HOST_PREP_SIZE = 913_728
WXC_HOST_PREP_SHA256 = "a9b8b14a11a1c5888641297c26abca547c2afa4435085c03ccfebd1deface310"
HOST_PREP_STEPS = ("prepare-system-drive", "prepare-null-device")
HOST_PREP_PROBE_SECONDS = 10.0

_WXC_EXEC_NAME = "wxc-exec.exe"
_WXC_HOST_PREP_NAME = "wxc-host-prep.exe"
_lock = threading.RLock()


class MxcRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen = True)
class RuntimeInfo:
    path: Path
    sha256: str

    @property
    def identity(self) -> str:
        material = {
            "architecture": "x86_64",
            "mxcRevision": MXC_REVISION,
            "releaseTag": RELEASE_TAG,
            "sha256": self.sha256,
            "size": WXC_EXEC_SIZE,
        }
        encoded = json.dumps(material, sort_keys = True, separators = (",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class RuntimeLease:
    info: RuntimeInfo
    _guard: object
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        close = getattr(self._guard, "close", None)
        if close is not None:
            close()
        self._guard = None

    def __enter__(self) -> RuntimeLease:
        return self

    def __exit__(self, *_args) -> None:
        self.release()


def _studio_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    return Path(override).expanduser() if override else Path.home() / ".unsloth" / "studio"


def _installed_package_root() -> Path:
    return _studio_root() / "mxc-runtime" / "windows-x86_64"


def dacl_state_path() -> Path:
    """WXC's DACL restore journal: one fixed place, so every start (probe included) reaps the same orphans."""
    # Absolute: wxc-exec runs with the runtime dir as its cwd, so a relative home would split the journal.
    return Path(os.path.abspath(_studio_root() / "mxc-runtime" / "dacl-restore"))


def dacl_state_dir() -> Path:
    path = dacl_state_path()
    path.mkdir(parents = True, exist_ok = True)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MxcRuntimeUnavailable("the managed wxc-exec.exe could not be read") from exc
    return digest.hexdigest()


def _require_plain_directory(path: Path) -> Path:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise MxcRuntimeUnavailable("the managed MXC runtime is not installed") from exc
    if not path.is_dir() or path.is_symlink() or attributes & 0x400:
        raise MxcRuntimeUnavailable(
            "the managed MXC runtime must be a non-reparse directory",
        )
    return path.resolve()


def _expected_architecture() -> str:
    machine = platform.machine().casefold()
    if machine not in {"amd64", "x86_64"}:
        raise MxcRuntimeUnavailable(
            f"Microsoft MXC v0.8.0 is supported only on Windows x86-64, not {machine or 'unknown'}",
        )
    return "x86_64"


def _validate_artifact(package_root: Path, name: str, size: int, sha256: str) -> RuntimeInfo:
    root = _require_plain_directory(package_root)
    executable = root / name
    if not executable.is_file() or executable.is_symlink():
        raise MxcRuntimeUnavailable(f"the managed {name} is missing or is not a regular file")
    try:
        metadata = executable.stat()
    except OSError as exc:
        raise MxcRuntimeUnavailable(f"the managed {name} is unreadable") from exc
    if getattr(metadata, "st_nlink", 1) != 1:
        raise MxcRuntimeUnavailable(
            f"the managed {name} has an unapproved hard link",
        )
    if metadata.st_size != size:
        raise MxcRuntimeUnavailable(f"the managed {name} size is not approved")
    digest = _sha256_file(executable)
    if digest != sha256:
        raise MxcRuntimeUnavailable(f"the managed {name} digest is not approved")
    return RuntimeInfo(path = executable.resolve(), sha256 = digest)


def _validate_runtime(package_root: Path) -> RuntimeInfo:
    return _validate_artifact(package_root, _WXC_EXEC_NAME, WXC_EXEC_SIZE, WXC_EXEC_SHA256)


def _validate_host_prep(package_root: Path) -> RuntimeInfo:
    return _validate_artifact(
        package_root, _WXC_HOST_PREP_NAME, WXC_HOST_PREP_SIZE, WXC_HOST_PREP_SHA256
    )


def selected_runtime(*, package_root: Path | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only")
    _expected_architecture()
    return _validate_runtime(package_root or _installed_package_root())


def selected_host_prep(*, package_root: Path | None = None) -> RuntimeInfo:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("MXC host preparation is Windows-only")
    _expected_architecture()
    return _validate_host_prep(package_root or _installed_package_root())


def wxc_path() -> Path:
    return selected_runtime().path


def installation_identity() -> str:
    return selected_runtime().identity


class _WindowsHandleGuard:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.windll.kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            close_handle(self.handle)
            self.handle = 0


def _open_artifact_guard(path: Path) -> object:
    if os.name != "nt":
        return path.open("rb")
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(str(path), 0x80000000, 0x1, None, 3, 0x80, None)
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        raise MxcRuntimeUnavailable(f"the managed {path.name} could not be locked for launch")
    return _WindowsHandleGuard(int(handle))


def _acquire(select, package_root: Path | None) -> RuntimeLease:
    with _lock:
        info = select(package_root = package_root)
        guard = _open_artifact_guard(info.path)
        try:
            if select(package_root = package_root) != info:
                raise MxcRuntimeUnavailable(
                    f"the managed {info.path.name} changed during acquisition"
                )
        except Exception:
            guard.close()
            raise
        return RuntimeLease(info = info, _guard = guard)


def acquire_runtime(*, package_root: Path | None = None) -> RuntimeLease:
    return _acquire(selected_runtime, package_root)


def acquire_host_prep(*, package_root: Path | None = None) -> RuntimeLease:
    """Pinned wxc-host-prep held deny-write, so the elevated launch runs the verified bytes."""
    return _acquire(selected_host_prep, package_root)


def _run_wxc_probe(
    package_root: Path | None,
    env: dict[str, str] | None,
    *,
    replay_journal: bool = True,
):
    # --probe reaps orphaned ACEs first: point it at Studio's journal, not %LOCALAPPDATA%'s.
    env = dict(os.environ if env is None else env)
    with tempfile.TemporaryDirectory(prefix = "unsloth-mxc-empty-journal-") as empty:
        # An elevated probe must not replay a user-writable journal: it names the ACLs to rewrite.
        env["MXC_DACL_STATE_DIR"] = str(dacl_state_dir()) if replay_journal else empty
        with acquire_runtime(package_root = package_root) as lease:
            return subprocess.run(
                [str(lease.info.path), "--probe"],
                stdin = subprocess.DEVNULL,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                cwd = str(lease.info.path.parent),
                env = env,
                timeout = HOST_PREP_PROBE_SECONDS,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check = False,
            )


def recover_dacl_state(env: dict[str, str] | None = None) -> bool:
    """Replay the DACL journal now; True only when wxc-exec reports no recovery error."""
    try:
        completed = _run_wxc_probe(None, env)
    except Exception:  # noqa: BLE001 - an unknown outcome is not a clean one
        return False
    stderr = completed.stderr or ""
    # main.rs prints "DACL recovery: ... N error(s)" only when there was work, or "DACL recovery failed".
    report = re.search(r"DACL recovery: .*?(\d+) error\(s\)", stderr)
    return (
        completed.returncode == 0
        and "DACL recovery failed" not in stderr
        and (report is None or report.group(1) == "0")
    )


def probe_host_prep_steps(
    *,
    package_root: Path | None = None,
    env: dict[str, str] | None = None,
    replay_journal: bool = True,
) -> tuple[str, ...] | None:
    """Host preparation `wxc-exec --probe` reports missing; None when it cannot tell."""
    try:
        completed = _run_wxc_probe(package_root, env, replay_journal = replay_journal)
        warnings = json.loads(completed.stdout).get("warnings")
    except Exception:  # noqa: BLE001 - advice only, never a capability verdict
        return None
    if completed.returncode != 0 or not isinstance(warnings, list):
        return None
    # MXC names the verb in each Tier 3 warning (fallback_detector.rs push_host_prep_warnings).
    return tuple(
        step
        for step in HOST_PREP_STEPS
        if any(isinstance(item, str) and f"wxc-host-prep {step}" in item for item in warnings)
    )
