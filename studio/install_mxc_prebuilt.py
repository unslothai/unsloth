#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Install Microsoft's pinned WXC v0.8.0 executables for Studio on Windows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

_STUDIO_DIR = Path(__file__).resolve().parent
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))

from backend.core.inference import mxc_runtime  # noqa: E402
from prebuilt_core import BusyInstallConflict, install_lock, install_lock_path, swap_into_place  # noqa: E402


class MxcInstallError(RuntimeError):
    pass


def _download(destination: Path) -> None:
    request = urllib.request.Request(
        mxc_runtime.RELEASE_URL,
        headers = {"User-Agent": "unsloth-studio-mxc-prebuilt"},
    )
    try:
        with urllib.request.urlopen(request, timeout = 60) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out, length = 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok = True)
        raise MxcInstallError(
            f"could not download the pinned Microsoft MXC release: {exc}"
        ) from exc


HOST_PREP_TIMEOUT_SECONDS = 120


def _approved_members() -> dict[str, tuple[str, int]]:
    return {
        mxc_runtime.RELEASE_MEMBER: ("wxc-exec.exe", mxc_runtime.WXC_EXEC_SIZE),
        mxc_runtime.RELEASE_HOST_PREP_MEMBER: (
            "wxc-host-prep.exe",
            mxc_runtime.WXC_HOST_PREP_SIZE,
        ),
    }


def _already_installed(install_dir: Path) -> bool:
    try:
        mxc_runtime._validate_runtime(install_dir)
        mxc_runtime._validate_host_prep(install_dir)
    except mxc_runtime.MxcRuntimeUnavailable:
        return False
    return True


def _validate_archive_entries(bundle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = _approved_members()
    seen: set[str] = set()
    approved: dict[str, list[zipfile.ZipInfo]] = {name: [] for name in members}
    for entry in bundle.infolist():
        name = entry.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise MxcInstallError("the Microsoft MXC archive contains an unsafe member path")
        folded = name.casefold()
        if folded in seen:
            raise MxcInstallError("the Microsoft MXC archive contains a duplicate member")
        seen.add(folded)
        mode = (entry.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise MxcInstallError("the Microsoft MXC archive contains a symbolic link")
        if name in approved:
            approved[name].append(entry)
    result: dict[str, zipfile.ZipInfo] = {}
    for name, (target, size) in members.items():
        if len(approved[name]) != 1:
            raise MxcInstallError(
                f"the Microsoft MXC archive does not contain one approved {target}"
            )
        entry = approved[name][0]
        if entry.is_dir() or entry.file_size != size:
            raise MxcInstallError(
                f"the approved {target} archive member has an unexpected shape or size"
            )
        result[target] = entry
    return result


def install_mxc_release(install_dir: Path) -> bool:
    """Install the approved WXC binary; return ``True`` only when files changed."""
    if sys.platform != "win32":
        raise MxcInstallError("the MXC runtime installer is Windows-only")
    if mxc_runtime._expected_architecture() != "x86_64":
        raise MxcInstallError("the Microsoft MXC release supports Windows x86-64 only")

    install_dir = install_dir.expanduser().resolve()
    install_dir.parent.mkdir(parents = True, exist_ok = True)
    with install_lock(install_lock_path(install_dir)):
        if _already_installed(install_dir):
            return False

        stage = Path(tempfile.mkdtemp(prefix = f".{install_dir.name}-", dir = install_dir.parent))
        try:
            archive = stage / ".mxc-release.zip"
            _download(archive)
            actual_size = archive.stat().st_size
            if actual_size != mxc_runtime.RELEASE_ARCHIVE_SIZE:
                raise MxcInstallError(
                    "MXC release size mismatch: "
                    f"expected {mxc_runtime.RELEASE_ARCHIVE_SIZE}, got {actual_size}"
                )
            actual_digest = mxc_runtime._sha256_file(archive)
            if actual_digest != mxc_runtime.RELEASE_ARCHIVE_SHA256:
                raise MxcInstallError(
                    "MXC release checksum mismatch: "
                    f"expected {mxc_runtime.RELEASE_ARCHIVE_SHA256}, got {actual_digest}"
                )
            try:
                with zipfile.ZipFile(archive) as bundle:
                    for target, entry in _validate_archive_entries(bundle).items():
                        with bundle.open(entry) as source, (stage / target).open("wb") as output:
                            shutil.copyfileobj(source, output, length = 1024 * 1024)
                            output.flush()
                            os.fsync(output.fileno())
            except (OSError, zipfile.BadZipFile) as exc:
                raise MxcInstallError("the pinned Microsoft MXC archive is invalid") from exc
            archive.unlink()
            mxc_runtime._validate_runtime(stage)
            mxc_runtime._validate_host_prep(stage)
            swap_into_place(stage, install_dir)
        finally:
            shutil.rmtree(stage, ignore_errors = True)
    return True


def _is_elevated() -> bool:
    import ctypes
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def _run_elevated(executable: Path, arguments: list[str], directory: str) -> int:
    """UAC prompt via ShellExecuteExW("runas"): CreateProcess cannot start this exe unelevated."""
    import ctypes
    from ctypes import wintypes

    class ShellExecuteInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", wintypes.LPVOID),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIconOrMonitor", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    shell32 = ctypes.WinDLL("shell32", use_last_error = True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(ShellExecuteInfo)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(ShellExecuteInfo)
    info.fMask = 0x40 | 0x100 | 0x400  # NOCLOSEPROCESS | NOASYNC | FLAG_NO_UI
    info.lpVerb = "runas"
    info.lpFile = str(executable)
    info.lpParameters = subprocess.list2cmdline(arguments)
    info.lpDirectory = directory
    info.nShow = 0
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        error = ctypes.get_last_error()
        if error == 1223:
            raise MxcInstallError("the Windows administrator prompt was declined")
        raise MxcInstallError(f"could not start wxc-host-prep elevated (Windows error {error})")
    if not info.hProcess:
        raise MxcInstallError("wxc-host-prep started without a process handle")
    try:
        if kernel32.WaitForSingleObject(info.hProcess, HOST_PREP_TIMEOUT_SECONDS * 1000) != 0:
            kernel32.TerminateProcess(info.hProcess, 1)
            raise MxcInstallError("wxc-host-prep did not finish in time")
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
            raise MxcInstallError("could not read the wxc-host-prep exit code")
        return int(code.value)
    finally:
        kernel32.CloseHandle(info.hProcess)


def _run_host_prep(executable: Path, step: str) -> int:
    arguments = [step, "--quiet"] if step == "prepare-null-device" else [step]
    directory = os.environ.get("SystemRoot") or str(executable.parent)
    if _is_elevated():
        try:
            return subprocess.run(
                [str(executable), *arguments],
                cwd = directory,
                stdin = subprocess.DEVNULL,
                timeout = HOST_PREP_TIMEOUT_SECONDS,
                check = False,
            ).returncode
        except subprocess.TimeoutExpired as exc:
            raise MxcInstallError("wxc-host-prep did not finish in time") from exc
    return _run_elevated(executable, arguments, directory)


def prepare_host(install_dir: Path) -> tuple[str, ...]:
    """Run the host preparation MXC's Tier 3 reports missing; both steps when it cannot tell."""
    if sys.platform != "win32":
        raise MxcInstallError("MXC host preparation is Windows-only")
    install_dir = install_dir.expanduser().resolve()
    steps = mxc_runtime.probe_host_prep_steps(package_root = install_dir)
    if steps is None:
        steps = mxc_runtime.HOST_PREP_STEPS
    if not steps:
        return ()
    with mxc_runtime.acquire_host_prep(package_root = install_dir) as lease:
        for step in steps:
            code = _run_host_prep(lease.info.path, step)
            if code != 0:
                raise MxcInstallError(f"wxc-host-prep {step} failed with exit code {code}")
    return tuple(steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--install-dir", type = Path)
    parser.add_argument(
        "--prepare-host",
        action = "store_true",
        help = "Run the elevated wxc-host-prep steps the opt-in DACL mode needs "
        "(prepare-null-device again after every reboot).",
    )
    args = parser.parse_args(argv)
    if args.prepare_host:
        try:
            steps = prepare_host(args.install_dir or mxc_runtime._installed_package_root())
        except (MxcInstallError, mxc_runtime.MxcRuntimeUnavailable, OSError) as exc:
            print(f"[mxc-prebuilt] {exc}", file = sys.stderr)
            return 1
        print(
            f"[mxc-prebuilt] host prepared: {', '.join(steps)}"
            if steps
            else "[mxc-prebuilt] host already prepared"
        )
        return 0
    if args.install_dir is None:
        parser.error("--install-dir is required")
    try:
        changed = install_mxc_release(args.install_dir)
    except BusyInstallConflict as exc:
        print(f"[mxc-prebuilt] install blocked by an active MXC process: {exc}", file = sys.stderr)
        return 3
    except (MxcInstallError, mxc_runtime.MxcRuntimeUnavailable) as exc:
        print(f"[mxc-prebuilt] {exc}", file = sys.stderr)
        return 1
    print("[mxc-prebuilt] installed and validated" if changed else "[mxc-prebuilt] already matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
