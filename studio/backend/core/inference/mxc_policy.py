# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Trusted policy construction for the Windows MXC Studio profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import site
import sys
import unicodedata
import uuid

from .mxc_runtime import MXC_REVISION, MXC_SCHEMA_VERSION, RUNNER_PROTOCOL_VERSION

PROFILE_ID = "unsloth-mxc-windows-basecontainer-v1"
PROFILE_VERSION = 1
POLICY_IDENTITY_VERSION = 1
MAX_ENVIRONMENT_ENTRIES = 512
MAX_PATH_SCAN_ENTRIES = 50_000
_WSL_TERMINAL_MARKERS = ("\\system32\\bash.exe", "\\windowsapps\\bash.exe")


class MxcPolicyError(RuntimeError):
    pass


def _object_identity(path: str, *, directory: bool) -> dict[str, int]:
    """Read stable volume/file identity without following the final reparse point."""
    if os.name != "nt":
        metadata = os.stat(path, follow_symlinks=False)
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
        ctypes.windll.kernel32.CloseHandle(handle)


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
    if os.path.normcase(canonical) != os.path.normcase(absolute):
        raise MxcPolicyError(f"MXC policy paths may not traverse a reparse point: {absolute}")
    current = Path(absolute)
    for candidate in (current, *current.parents):
        if _is_reparse(str(candidate)):
            raise MxcPolicyError(f"MXC policy paths may not contain a reparse point: {candidate}")
    return canonical


def _reject_workdir_reparse_entries(workdir: str) -> None:
    seen = 0
    for root, directories, files in os.walk(workdir, followlinks=False):
        for name in (*directories, *files):
            seen += 1
            if seen > MAX_PATH_SCAN_ENTRIES:
                raise MxcPolicyError("the MXC workdir is too large to validate reparse boundaries")
            candidate = os.path.join(root, name)
            if _is_reparse(candidate):
                raise MxcPolicyError(f"the MXC workdir contains a reparse point: {candidate}")
            try:
                links = os.stat(candidate, follow_symlinks=False).st_nlink
            except OSError as exc:
                raise MxcPolicyError(
                    f"the MXC workdir changed during validation: {candidate}"
                ) from exc
            if os.path.isfile(candidate) and links > 1:
                raise MxcPolicyError(f"the MXC workdir contains a hard-linked file: {candidate}")


def _absolute_existing_directory(path: str) -> str:
    value = _safe_canonical_path(path, directory=True)
    _reject_workdir_reparse_entries(value)
    return value


def _runtime_read_roots(executable: str) -> list[str]:
    # Preserve the lexical executable for launch. Grants may use canonical roots,
    # but never broaden to the user profile or a drive root.
    roots = [os.path.dirname(os.path.abspath(executable)), sys.prefix, sys.base_prefix]
    roots.extend(site.getsitepackages())
    roots.append(str(Path(__file__).with_name("sandbox_site")))
    if Path(executable).name.casefold() in {"bash", "bash.exe"}:
        executable_dir = Path(executable).resolve().parent
        git_root = (
            executable_dir.parent
            if executable_dir.name.casefold() in {"bin", "usr"}
            else executable_dir
        )
        if (
            executable_dir.name.casefold() == "bin"
            and executable_dir.parent.name.casefold() == "usr"
        ):
            git_root = executable_dir.parent.parent
        roots.extend(
            str(path)
            for path in (
                git_root,
                git_root / "bin",
                git_root / "usr" / "bin",
                git_root / "mingw64" / "bin",
            )
        )
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root:
        roots.append(system_root)
    result: list[str] = []
    for root in roots:
        if not root:
            continue
        if not os.path.isdir(root):
            continue
        canonical = _safe_canonical_path(root, directory=True)
        drive, tail = os.path.splitdrive(canonical)
        if not tail.strip("\\/"):
            raise MxcPolicyError(f"volume-root MXC grant is forbidden: {canonical}")
        if os.path.normcase(canonical) not in {os.path.normcase(value) for value in result}:
            result.append(canonical)
    return result


def _selected_runtime(plan) -> str:
    executable = plan.argv[0]
    if os.path.isabs(executable):
        selected = executable
    elif plan.execution_kind == "terminal":
        selected = shutil.which(executable, path=plan.env.get("PATH"))
        if selected is None and executable.casefold() in {"cmd", "cmd.exe"}:
            selected = os.environ.get("COMSPEC")
        if selected is None:
            raise MxcPolicyError(
                f"the selected terminal executable could not be resolved: {executable}"
            )
    else:
        raise MxcPolicyError("the selected Python executable must be absolute")
    canonical = _safe_canonical_path(selected, directory=False)
    lowered = canonical.replace("/", "\\").casefold()
    if plan.execution_kind == "terminal" and any(
        marker in lowered for marker in _WSL_TERMINAL_MARKERS
    ):
        raise MxcPolicyError(
            "the selected bash.exe is WSL-backed and is not covered by the Windows MXC profile"
        )
    return selected


def canonical_policy_material(request: dict) -> dict:
    """Return the versioned policy identity independently reconstructed by Rust."""
    return {
        "identityVersion": POLICY_IDENTITY_VERSION,
        "protocolVersion": request["protocol"],
        "profileId": request["profileId"],
        "profileVersion": request["profileVersion"],
        "schemaVersion": request["schemaVersion"],
        "runtimeRevision": request["runtimeRevision"],
        "runId": request["runId"],
        "containerId": request["containerId"],
        "argv": list(request["argv"]),
        "executionKind": request["executionKind"],
        "runtimePath": request["runtimePath"],
        "runtimeIdentity": dict(request["runtimeIdentity"]),
        "cwd": request["cwd"],
        "workdirIdentity": dict(request["workdirIdentity"]),
        "environment": dict(request["environment"]),
        "environmentPolicy": request["environmentPolicy"],
        "commandLinePolicy": request["commandLinePolicy"],
        "filesystem": {
            "readwritePaths": list(request["readwritePaths"]),
            "readonlyPaths": list(request["readonlyPaths"]),
            "deniedPaths": list(request["deniedPaths"]),
            "clearPolicyOnExit": request["clearPolicyOnExit"],
        },
        "network": {
            "profile": request["networkProfile"],
            "allowOutbound": request["allowOutbound"],
            "allowLocalNetwork": request["allowLocalNetwork"],
        },
        "uiPolicy": request["uiPolicy"],
        "timeoutMs": request["timeoutMs"],
        "fallback": {"allowDaclMutation": request["allowDaclMutation"]},
        "admission": request["admission"],
    }


def compute_policy_hash(request: dict) -> str:
    encoded = json.dumps(
        canonical_policy_material(request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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

    workdir = _absolute_existing_directory(plan.workdir)
    selected_runtime = _selected_runtime(plan)
    readonly = _runtime_read_roots(selected_runtime)
    run_id = run_id or uuid.uuid4().hex
    workload_env = dict(plan.env)
    # MXC's Windows ProcessContainer validator requires LOCALAPPDATA to be
    # present. Point it inside the session instead of exposing the real profile.
    workload_env.setdefault("LOCALAPPDATA", workdir)
    request = {
        "protocol": RUNNER_PROTOCOL_VERSION,
        "runId": run_id,
        "profileId": PROFILE_ID,
        "profileVersion": PROFILE_VERSION,
        "schemaVersion": MXC_SCHEMA_VERSION,
        "runtimeRevision": MXC_REVISION,
        "containerId": f"unsloth-{run_id}",
        "argv": list(plan.argv),
        "executionKind": plan.execution_kind,
        "runtimePath": os.path.realpath(selected_runtime),
        "runtimeIdentity": _object_identity(selected_runtime, directory=False),
        "cwd": workdir,
        "workdirIdentity": _object_identity(workdir, directory=True),
        "environment": workload_env,
        "environmentPolicy": "sanitized-explicit-v1",
        "commandLinePolicy": "windows-createprocess-argv-v1",
        "readwritePaths": [workdir],
        "readonlyPaths": readonly,
        "deniedPaths": [],
        "clearPolicyOnExit": True,
        "networkProfile": "compatibility",
        "allowOutbound": True,
        "allowLocalNetwork": True,
        "uiPolicy": None,
        "timeoutMs": None
        if plan.timeout_seconds is None
        else max(1, int(plan.timeout_seconds * 1000)),
        "allowDaclMutation": False,
        "admission": "atomic-no-dacl-fallback",
    }
    request["policyHash"] = compute_policy_hash(request)
    return request
