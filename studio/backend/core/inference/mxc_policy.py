# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Trusted policy construction for the Windows MXC Studio profile."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import unicodedata
import uuid

from . import os_sandbox
from . import mxc_runtime
from .mxc_runtime import MXC_SCHEMA_VERSION

logger = logging.getLogger(__name__)

MAX_ENVIRONMENT_ENTRIES = 512
DACL_FALLBACK_ENV = "UNSLOTH_MXC_ALLOW_DACL_FALLBACK"
_WSL_TERMINAL_MARKERS = ("\\system32\\bash.exe", "\\windowsapps\\bash.exe")


def _studio_ui_policy() -> dict:
    return {
        "ui": {
            "disable": False,
            "clipboard": "none",
            "injection": False,
        },
        "processContainer": {
            "ui": {
                "isolation": "container",
                "desktopSystemControl": False,
                "systemSettings": "none",
                "ime": False,
            }
        },
    }


class MxcPolicyError(RuntimeError):
    pass


def dacl_fallback_enabled() -> bool:
    """Host opt-in to MXC's AppContainer tier, which temporarily adds ACEs to the granted host paths."""
    return os.environ.get(DACL_FALLBACK_ENV, "").strip() == "1"


def _reject_grants_over_dacl_journal(grants: list[str]) -> None:
    # A workload that can write the journal chooses which ACLs wxc-exec rewrites on recovery.
    journal = os.path.normcase(os.path.realpath(mxc_runtime.dacl_state_path()))
    for grant in grants:
        root = os.path.normcase(grant).rstrip("\\/")
        if journal == root or journal.startswith(root + os.sep):
            raise MxcPolicyError(f"an MXC grant would expose the DACL restore journal: {grant}")


def _object_identity(path: str, *, directory: bool) -> dict[str, int]:
    """Read stable volume/file identity without following the final reparse point."""
    if os.name != "nt":
        metadata = os.stat(path, follow_symlinks = False)
        return {"volumeSerialNumber": metadata.st_dev, "fileId": metadata.st_ino}
    import ctypes
    from ctypes import wintypes

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

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
    flags = 0x00200000 | (0x02000000 if directory else 0)
    handle = create_file(path, 0x80, 0x1 | 0x2, None, 3, flags, None)
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        raise MxcPolicyError(f"could not open the MXC policy object identity: {path}")
    try:
        information = FileInformation()
        if not ctypes.windll.kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            raise MxcPolicyError(f"could not read the MXC policy object identity: {path}")
        if information.dwFileAttributes & 0x400:
            raise MxcPolicyError(f"MXC policy objects may not be reparse points: {path}")
        return {
            "volumeSerialNumber": int(information.dwVolumeSerialNumber),
            "fileId": (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow),
        }
    finally:
        close_handle = ctypes.windll.kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(handle)


def _long_path(path: str) -> str:
    """Expand 8.3 components (RUNNER~1); realpath already returns the long form."""
    if os.name != "nt":
        return path
    import ctypes
    from ctypes import wintypes

    get_long_path = ctypes.windll.kernel32.GetLongPathNameW
    get_long_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_long_path.restype = wintypes.DWORD
    size = get_long_path(path, None, 0)
    if not size:
        return path
    buffer = ctypes.create_unicode_buffer(size)
    if not get_long_path(path, buffer, size):
        return path
    return buffer.value


def _is_reparse(path: str) -> bool:
    try:
        return bool(os.lstat(path).st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return os.path.islink(path)


def _safe_canonical_path(path: str, *, directory: bool) -> str:
    absolute = os.path.abspath(path)
    if absolute.startswith(("\\\\", "\\?\\", "\\.\\")):
        raise MxcPolicyError(
            f"UNC and device namespace paths are not supported by the MXC profile: {absolute}"
        )
    _, tail = os.path.splitdrive(absolute)
    if ":" in tail:
        raise MxcPolicyError(f"alternate data stream paths are not supported: {absolute}")
    if unicodedata.normalize("NFC", absolute) != absolute:
        raise MxcPolicyError(f"non-canonical Unicode paths are not supported: {absolute}")
    if directory and not os.path.isdir(absolute):
        raise MxcPolicyError(f"required MXC policy directory is unavailable: {absolute}")
    if not directory and not os.path.isfile(absolute):
        raise MxcPolicyError(f"required MXC runtime is unavailable: {absolute}")
    canonical = os.path.realpath(absolute)
    if os.path.normcase(canonical) != os.path.normcase(_long_path(absolute)):
        raise MxcPolicyError(f"MXC policy paths may not traverse a reparse point: {absolute}")
    current = Path(absolute)
    for candidate in (current, *current.parents):
        if _is_reparse(str(candidate)):
            raise MxcPolicyError(f"MXC policy paths may not contain a reparse point: {candidate}")
    return canonical


def _scan_workdir(workdir: str, *, required: bool) -> tuple[str, ...]:
    """Reparse points and outside hard links refuse; an overrun budget degrades `auto` only."""
    try:
        limitations = os_sandbox.scan_workdir_for_host_channels(workdir)
    except os_sandbox.WorkdirUnsafeError as exc:
        raise MxcPolicyError(str(exc)) from exc
    if required and os_sandbox.WORKDIR_SCAN_INCOMPLETE in limitations:
        raise MxcPolicyError(
            "the session workdir is too large to check for host channels, and `required` "
            "cannot promise a boundary it did not verify. Start a new chat, or use `auto`."
        )
    return limitations


def _runtime_read_roots(executable: str) -> list[str]:
    # Preserve the lexical executable for launch. Grants may use canonical roots,
    # but never broaden to the user profile or a drive root.
    roots = [os.path.dirname(os.path.abspath(executable)), sys.prefix, sys.base_prefix]
    roots.extend(site.getsitepackages())
    roots.append(str(Path(__file__).with_name("sandbox_site")))
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root:
        roots.append(system_root)
    result: list[str] = []
    for root in roots:
        if not root:
            continue
        if not os.path.isdir(root):
            continue
        canonical = _safe_canonical_path(root, directory = True)
        drive, tail = os.path.splitdrive(canonical)
        if not tail.strip("\\/"):
            raise MxcPolicyError(f"volume-root MXC grant is forbidden: {canonical}")
        if os.path.normcase(canonical) not in {os.path.normcase(value) for value in result}:
            result.append(canonical)
    return result


def _model_read_roots(workdir: str, granted: list[str]) -> list[str]:
    """Registered model folders, read-only as on Linux and macOS; one that cannot be granted is skipped."""
    covered = [os.path.normcase(value) for value in (workdir, *granted)]
    result: list[str] = []
    for root in os_sandbox.model_library_roots():
        try:
            canonical = _safe_canonical_path(root, directory = True)
        except MxcPolicyError as exc:
            logger.info("Not granting a model folder to the MXC profile: %s", exc)
            continue
        folded = os.path.normcase(canonical)
        if any(folded == value or folded.startswith(value.rstrip("\\/") + os.sep) for value in covered):
            continue
        covered.append(folded)
        result.append(canonical)
    return result


def _selected_runtime(plan) -> str:
    executable = plan.argv[0]
    if os.path.isabs(executable):
        selected = executable
    elif plan.execution_kind == "terminal":
        selected = shutil.which(executable, path = plan.env.get("PATH"))
        if selected is None and executable.casefold() in {"cmd", "cmd.exe"}:
            selected = os.environ.get("COMSPEC")
        if selected is None:
            raise MxcPolicyError(
                f"the selected terminal executable could not be resolved: {executable}"
            )
    else:
        raise MxcPolicyError("the selected Python executable must be absolute")
    canonical = _safe_canonical_path(selected, directory = False)
    lowered = canonical.replace("/", "\\").casefold()
    if plan.execution_kind == "terminal" and any(
        marker in lowered for marker in _WSL_TERMINAL_MARKERS
    ):
        raise MxcPolicyError(
            "the selected bash.exe is WSL-backed and is not covered by the Windows MXC profile"
        )
    return selected


def canonical_config_bytes(config: dict) -> bytes:
    """Serialize exactly the deterministic stable-v0.8 request sent to WXC."""
    return json.dumps(
        config,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":"),
    ).encode("utf-8")


def compute_policy_hash(config: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def verify_launch_identities(request: dict) -> None:
    """Re-check the selected workload and workdir immediately before WXC dispatch."""
    if _object_identity(request["runtimePath"], directory = False) != request["runtimeIdentity"]:
        raise MxcPolicyError("the selected workload executable changed before WXC dispatch")
    if _object_identity(request["cwd"], directory = True) != request["workdirIdentity"]:
        raise MxcPolicyError("the MXC workdir changed before WXC dispatch")
    _safe_canonical_path(request["cwd"], directory = True)
    _scan_workdir(request["cwd"], required = request["requireCompleteScan"])


def build_launch_request(plan, *, run_id: str | None = None) -> dict:
    if sys.platform != "win32":
        raise MxcPolicyError("MXC policy construction is Windows-only")
    if plan.execution_kind not in {"python", "terminal"}:
        raise MxcPolicyError("the MXC profile requires a trusted Python or Terminal launch plan")
    if not plan.argv:
        raise MxcPolicyError("the MXC launch argv is empty")
    if len(plan.env) > MAX_ENVIRONMENT_ENTRIES:
        raise MxcPolicyError("the sanitized environment exceeds the MXC policy bound")
    if any(not isinstance(k, str) or not k or "=" in k for k in plan.env):
        raise MxcPolicyError("the sanitized environment contains an invalid variable name")

    required = getattr(plan, "requested_mode", None) == "required"
    workdir = _safe_canonical_path(plan.workdir, directory = True)
    limitations = _scan_workdir(workdir, required = required)
    selected_runtime = _selected_runtime(plan)
    readonly = _runtime_read_roots(selected_runtime)
    readonly += _model_read_roots(workdir, readonly)
    _reject_grants_over_dacl_journal([workdir, *readonly])
    execution_argv = list(plan.argv)
    execution_argv[0] = selected_runtime
    run_id = run_id or uuid.uuid4().hex
    workload_env = dict(plan.env)
    # MXC's Windows ProcessContainer validator requires LOCALAPPDATA to be
    # present. Point it inside the session instead of exposing the real profile.
    workload_env.setdefault("LOCALAPPDATA", workdir)
    ui_policy = _studio_ui_policy()
    config = {
        "version": MXC_SCHEMA_VERSION,
        "containerId": f"unsloth-{run_id}",
        "containment": "processcontainer",
        "lifecycle": {"destroyOnExit": True, "preservePolicy": False},
        "process": {
            "commandLine": subprocess.list2cmdline(execution_argv),
            "cwd": workdir,
            "env": [f"{key}={value}" for key, value in sorted(workload_env.items())],
            "timeout": 0
            if plan.timeout_seconds is None
            else max(1, int(plan.timeout_seconds * 1000)),
        },
        "filesystem": {
            "readwritePaths": [workdir],
            "readonlyPaths": readonly,
            "deniedPaths": [],
        },
        "network": {
            "defaultPolicy": "allow",
            "allowLocalNetwork": True,
            "allowedHosts": [],
            "blockedHosts": [],
            "enforcementMode": "capabilities",
        },
        "processContainer": {
            "leastPrivilege": False,
            "capabilities": ["internetClient", "privateNetworkClientServer"],
            "ui": ui_policy["processContainer"]["ui"],
        },
        "ui": ui_policy["ui"],
        "fallback": {"allowDaclMutation": dacl_fallback_enabled()},
    }
    request = {
        "runtimePath": os.path.realpath(selected_runtime),
        "runtimeIdentity": _object_identity(selected_runtime, directory = False),
        "cwd": workdir,
        "workdirIdentity": _object_identity(workdir, directory = True),
        "config": config,
        "configBytes": canonical_config_bytes(config),
        "policyHash": compute_policy_hash(config),
        "requireCompleteScan": required,
        "launchLimitations": limitations,
    }
    return request
