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
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(ShellExecuteInfo)
    info.fMask = 0x40 | 0x100 | 0x400  # NOCLOSEPROCESS | NOASYNC | FLAG_NO_UI
    info.lpVerb = "runas"
    info.lpFile = str(executable)
    info.lpParameters = subprocess.list2cmdline(arguments)
    info.lpDirectory = directory
    info.nShow = 1  # its console shows the progress of a multi-minute run
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        error = ctypes.get_last_error()
        if error == 1223:
            raise MxcInstallError("the Windows administrator prompt was declined")
        raise MxcInstallError(
            f"could not start the elevated host preparation (Windows error {error})"
        )
    if not info.hProcess:
        raise MxcInstallError("wxc-host-prep started without a process handle")
    try:
        # INFINITE: killing prepare-system-drive mid-propagation leaves C:\ half re-ACLed.
        if kernel32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF) != 0:
            raise MxcInstallError("could not wait for wxc-host-prep")
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
            raise MxcInstallError("could not read the wxc-host-prep exit code")
        return int(code.value)
    finally:
        kernel32.CloseHandle(info.hProcess)


def _system_directory() -> str:
    """System32 from the API: SystemRoot is a user-settable variable, and this path runs elevated."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(260)
    if not ctypes.windll.kernel32.GetSystemDirectoryW(buffer, 260):
        raise MxcInstallError("could not locate the Windows system directory")
    return buffer.value


def _host_prep_script(executable: Path, arguments: list[str]) -> str:
    """Elevated: copy the pinned exe into a fresh admin-only dir, re-hash it, run it from there.

    The install dir is user-writable, so running host-prep in place would let a planted DLL
    (it LoadLibrary's dbghelp.dll) execute as administrator.
    """

    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    return "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            # Known folder, not $env:ProgramData: user variables can shadow it.
            "$root = [Environment]::GetFolderPath('CommonApplicationData')",
            "$dir = Join-Path $root ('unsloth-mxc-host-prep-' + [guid]::NewGuid().ToString('N'))",
            "New-Item -ItemType Directory -Path $dir | Out-Null",
            "try {",
            "  & icacls.exe $dir /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' "
            "'*S-1-5-18:(OI)(CI)F' | Out-Null",
            "  if ($LASTEXITCODE -ne 0) { exit 90 }",
            # Anything planted before the ACL change stays: the dir must still be empty.
            "  if (@(Get-ChildItem -LiteralPath $dir -Force).Count -ne 0) { exit 92 }",
            "  $exe = Join-Path $dir 'wxc-host-prep.exe'",
            f"  Copy-Item -LiteralPath {quote(str(executable))} -Destination $exe",
            "  $hash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash",
            f"  if ($hash -ne '{mxc_runtime.WXC_HOST_PREP_SHA256.upper()}') {{ exit 91 }}",
            f"  & $exe {' '.join(quote(value) for value in arguments)}",
            "  exit $LASTEXITCODE",
            "} finally {",
            "  Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue",
            "}",
        )
    )


_HOST_PREP_SCRIPT_ERRORS = {
    90: "could not restrict the host-prep staging directory to administrators",
    91: "the staged wxc-host-prep.exe does not match its pinned digest",
    92: "the host-prep staging directory was not empty after it was locked down",
}


def _run_host_prep(executable: Path, step: str) -> int:
    import base64

    arguments = [step, "--quiet"] if step == "prepare-null-device" else [step]
    system = _system_directory()
    powershell = os.path.join(system, "WindowsPowerShell", "v1.0", "powershell.exe")
    encoded = base64.b64encode(_host_prep_script(executable, arguments).encode("utf-16-le"))
    launcher = ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded.decode()]
    # Measured 3 to 7 minutes on a CI runner: it propagates an ACE across the whole system drive.
    print(f"[mxc-prebuilt] running wxc-host-prep {step}; this can take several minutes")
    if _is_elevated():
        code = subprocess.run(
            [powershell, *launcher],
            cwd = system,
            stdin = subprocess.DEVNULL,
            check = False,
        ).returncode
    else:
        code = _run_elevated(Path(powershell), launcher, system)
    if code in _HOST_PREP_SCRIPT_ERRORS:
        raise MxcInstallError(_HOST_PREP_SCRIPT_ERRORS[code])
    return code


def prepare_host(install_dir: Path) -> tuple[str, ...]:
    """Run the host preparation MXC's Tier 3 reports missing; both steps when it cannot tell."""
    if sys.platform != "win32":
        raise MxcInstallError("MXC host preparation is Windows-only")
    install_dir = install_dir.expanduser().resolve()
    steps = mxc_runtime.probe_host_prep_steps(
        package_root = install_dir, replay_journal = not _is_elevated()
    )
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
