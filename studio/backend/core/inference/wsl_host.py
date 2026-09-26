# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Managed engines on native Windows run inside a private WSL2 distro that Studio owns.

Studio stays a Windows process: it enables WSL (one UAC prompt), imports a pinned Ubuntu rootfs
as its own distro, and drives everything through ``wsl.exe`` argv lists. The engine's HTTP
server is reached through WSL2's localhost forwarding. The user's own distros are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

# Pinned: a dated release, not "current", so the bytes never move under the hash.
ROOTFS = {
    "url": "https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz",
    "sha256": "8251e27ffff381a4af5f41dcb94d867de3e0d9774a9241908ab34555d99315ea",
    "size": 356739129,
}
UV = {
    "url": "https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz",
    "sha256": "23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8",
    "size": 19831732,
}
MIN_BUILD = 19044  # Windows 10 21H2: first build with CUDA in WSL2.
GUEST_ROOT = "/opt/unsloth"
_CREATE_NO_WINDOW = 0x08000000

# systemd off: nothing else runs, so the distro starts in about a second and idles cleanly.
WSL_CONF = """[boot]
systemd=false
[interop]
appendWindowsPath=false
[automount]
root=/mnt/
"""

# Runs the engine in its own process group. The engine exits when Studio closes the stdin
# pipe, which happens on stop and when Studio itself dies, so vLLM never outlives its owner.
# A non-interactive sh gives background jobs /dev/null as stdin, hence the fd 3 duplicate.
RUNNER = """#!/bin/sh
exec 3<&0
setsid "$@" </dev/null &
child=$!
trap 'kill -TERM -$child 2>/dev/null' TERM INT HUP
(cat <&3 >/dev/null; kill -TERM -$child 2>/dev/null; sleep 10; kill -KILL -$child 2>/dev/null) &
watcher=$!
exec 3<&-
wait $child
status=$?
kill $watcher 2>/dev/null
exit $status
"""

# Windows messages and HRESULTs `wsl --import` / `--status` print for hosts that cannot run WSL2.
_BLOCKERS = (
    (
        # learn.microsoft.com/windows/wsl/troubleshooting lists both codes under BIOS virtualization.
        ("0x80370102", "0x80070003", "HCS_E_HYPERV_NOT_INSTALLED"),
        "Turn on virtualization (Intel VT-x or AMD-V) in the BIOS/UEFI settings and make sure the "
        "Virtual Machine Platform Windows feature is on, then click Install again.",
    ),
    (
        ("0x800701bc",),
        "WSL needs a kernel update. Run 'wsl --update' in a terminal, then click Install again.",
    ),
    (
        ("Group Policy", "blocked by policy", "0x80070005"),
        "WSL is blocked on this computer by an administrator policy.",
    ),
)


class Waiting(Exception):
    """A step only the user can finish: a UAC prompt or a Windows restart."""


def active() -> bool:
    return sys.platform == "win32"


def host_dir() -> Path:
    from .engine_install import engine_root
    return engine_root() / "wsl"


def distro_name() -> str:
    from utils.paths.storage_roots import studio_root

    # One distro per Studio home, so two installs never drive the same guest.
    return "Unsloth-Engines-" + hashlib.sha256(str(studio_root()).encode()).hexdigest()[:8]


def wsl_exe() -> str | None:
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "wsl.exe"
    return str(system) if system.is_file() else shutil.which("wsl.exe")


def native_machine() -> str:
    """The OS architecture, not the interpreter's: an x64 Python under ARM64 emulation reports AMD64."""
    try:
        import ctypes

        process, native = ctypes.c_ushort(), ctypes.c_ushort()
        kernel32 = ctypes.windll.kernel32
        if kernel32.IsWow64Process2(
            kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)
        ):
            return {0x8664: "x86_64", 0xAA64: "arm64", 0x14C: "x86"}.get(native.value, "unknown")
    except (AttributeError, OSError):
        pass
    arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    return {"AMD64": "x86_64", "ARM64": "arm64"}.get(arch.upper(), arch.lower())


def windows_build() -> int:
    try:
        return sys.getwindowsversion().build
    except AttributeError:
        return 0


def support_reason() -> str | None:
    if native_machine() != "x86_64":
        return "Managed engines on Windows need an x64 PC; the engine packages are x86_64 only."
    if windows_build() < MIN_BUILD:
        return "Managed engines need Windows 10 21H2 (build 19044) or newer for GPU access in WSL."
    return None


def boot_id() -> int:
    """Minute the machine booted; unchanged across Studio restarts, new after a Windows restart."""
    try:
        import ctypes

        tick = ctypes.windll.kernel32.GetTickCount64
        tick.restype = ctypes.c_ulonglong
        return int(time.time() - tick() / 1000) // 60
    except (AttributeError, OSError):
        return 0


def decode(raw: bytes) -> str:
    """wsl.exe writes its own messages as UTF-16LE; guest programs write UTF-8."""
    if raw[:2] == b"\xff\xfe" or (len(raw) > 1 and raw[1:2] == b"\x00"):
        return raw.decode("utf-16-le", errors = "replace").lstrip("\ufeff")
    return raw.decode("utf-8", errors = "replace")


def run(
    args: list[str],
    *,
    timeout: float = 120,
    input: bytes | None = None,
    env = None,
):
    exe = wsl_exe()
    if exe is None:
        raise RuntimeError("WSL is not installed.")
    result = subprocess.run(
        [exe, *args],
        input = input,
        capture_output = True,
        timeout = timeout,
        env = env,
        creationflags = _CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return result.returncode, decode(result.stdout + result.stderr)


def blocker(output: str) -> str | None:
    for needles, message in _BLOCKERS:
        if any(needle.lower() in output.lower() for needle in needles):
            return message
    return None


def _state_file() -> Path:
    return host_dir() / "state.json"


def read_state() -> dict:
    try:
        return json.loads(_state_file().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(**values) -> None:
    from .engine_install import _atomic_json
    host_dir().mkdir(parents = True, exist_ok = True)
    _atomic_json(_state_file(), {**read_state(), **values})


def wsl_state() -> str:
    """ready | missing | restart_required | blocked: <message>. Never elevates or boots a distro."""
    if wsl_exe() is None:
        return "missing"
    code, output = run(["--status"], timeout = 60)
    if code == 0:
        write_state(state = "ready")
        return "ready"
    state = read_state()
    if state.get("state") == "restart_required":
        if state.get("boot") == boot_id():
            return "restart_required"
        # Restarted and WSL still does not start: something the user must fix.
        return "blocked: " + (blocker(output) or output.strip()[-300:] or "WSL did not start.")
    return "missing"


def enable_wsl() -> str:
    """Install WSL with one UAC prompt. Only this fixed argv is ever elevated; the distro import
    runs unelevated, since an elevating admin account would register it under its own profile."""
    import ctypes
    from ctypes import wintypes

    class ShellExecuteInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", ctypes.c_ulong),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = 0x00000040  # SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "wsl.exe")
    info.lpParameters = "--install --no-distribution"
    info.nShow = 0
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        if ctypes.GetLastError() == 1223:  # ERROR_CANCELLED
            raise Waiting(
                "Windows asked for permission to install WSL and it was declined. Click Install to ask again."
            )
        raise RuntimeError("Windows could not start the WSL installer.")
    kernel32 = ctypes.windll.kernel32
    kernel32.WaitForSingleObject(info.hProcess, 30 * 60 * 1000)
    code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
    kernel32.CloseHandle(info.hProcess)
    state = wsl_state()
    if state == "ready":
        return state
    write_state(state = "restart_required", boot = boot_id(), installer_exit = code.value)
    raise Waiting("Restart Windows to finish installing WSL, then click Install again.")


def to_guest_path(path: str | os.PathLike) -> str:
    """C:\\Users\\a -> /mnt/c/Users/a under the automount root this distro is configured with."""
    windows = PureWindowsPath(path)
    if not re.fullmatch(r"[A-Za-z]:", windows.drive):
        raise ValueError(f"Engines inside WSL can only read local drive paths, not {path}")
    rest = "/".join(windows.parts[1:])
    return f"/mnt/{windows.drive[0].lower()}/{rest}".rstrip("/")


def guest_command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """wsl.exe argv plus the Windows env for it. Secrets cross through WSLENV, never argv, so they
    stay out of the Windows process list."""
    exe = wsl_exe()
    if exe is None:
        raise RuntimeError("WSL is not installed.")
    windows_env = dict(os.environ)
    secrets = secrets or {}
    windows_env.update(secrets)
    shared = [item for item in windows_env.get("WSLENV", "").split(":") if item]
    shared = [item for item in shared if item.split("/")[0] not in secrets]
    windows_env["WSLENV"] = ":".join([*shared, *(f"{key}/u" for key in secrets)])
    command = [exe, "-d", distro_name(), "-u", "root", "--cd", "/root", "--", "/usr/bin/env"]
    command += [f"{key}={value}" for key, value in (env or {}).items()]
    return command + list(argv), windows_env


def guest(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 600,
    input: bytes | None = None,
) -> str:
    command, windows_env = guest_command(argv, env = env)
    code, output = run(command[1:], timeout = timeout, input = input, env = windows_env)
    if code:
        raise RuntimeError(output.strip()[-2000:] or f"{argv[0]} failed in WSL.")
    return output


def put(
    path: str,
    text: str,
    mode: str = "644",
) -> None:
    """Write a guest file through stdin: multi-line argv does not survive Windows quoting intact."""
    guest(["sh", "-c", f'cat > "$0" && chmod {mode} "$0"', path], input = text.encode())


def distro_ready() -> bool:
    try:
        command, windows_env = guest_command(["test", "-f", f"{GUEST_ROOT}/owner.json"])
        code, _ = run(command[1:], timeout = 120, env = windows_env)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return False
    return code == 0


def download(
    spec: dict,
    name: str,
    progress = None,
) -> Path:
    """Fetch a pinned artifact once; a partial or tampered file never survives the hash check."""
    import httpx

    folder = host_dir() / "downloads"
    folder.mkdir(parents = True, exist_ok = True)
    target = folder / f"{spec['sha256'][:16]}-{name}"
    if target.is_file() and _sha256(target) == spec["sha256"]:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    done = 0
    with httpx.stream("GET", spec["url"], follow_redirects = True, timeout = 60) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(f"Downloading {name}: {done >> 20} / {spec['size'] >> 20} MiB")
    if digest.hexdigest() != spec["sha256"]:
        partial.unlink(missing_ok = True)
        raise RuntimeError(f"{name} did not match its pinned checksum. Retry the installation.")
    os.replace(partial, target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_distro(progress = None) -> None:
    """Idempotent: import the private distro, configure it, and install uv inside it."""
    from utils.paths.storage_roots import studio_root

    if not distro_ready():
        rootfs = download(ROOTFS, "ubuntu-24.04-wsl.rootfs.tar.gz", progress)
        if progress:
            progress("Creating the Unsloth WSL environment")
        # A half-imported distro from an interrupted run would make --import fail.
        run(["--unregister", distro_name()], timeout = 300)
        target = host_dir() / "distro"
        target.mkdir(parents = True, exist_ok = True)
        code, output = run(
            ["--import", distro_name(), str(target), str(rootfs), "--version", "2"], timeout = 1800
        )
        if code:
            raise RuntimeError(
                blocker(output) or "Could not create the WSL environment. " + output.strip()[-500:]
            )
        put("/etc/wsl.conf", WSL_CONF)
        run(["--terminate", distro_name()], timeout = 120)
        # The GPU paravirtualization library comes from the Windows driver, not the distro.
        try:
            guest(["test", "-e", "/usr/lib/wsl/lib/libcuda.so"])
        except RuntimeError:
            raise RuntimeError(
                "WSL cannot see the NVIDIA GPU. Update the NVIDIA Windows driver, then click Install again."
            ) from None
        owner = json.dumps({"studio_home": str(studio_root()), "distro": distro_name()})
        guest(["mkdir", "-p", f"{GUEST_ROOT}/bin"])
        put(f"{GUEST_ROOT}/owner.json", owner)
    put(f"{GUEST_ROOT}/bin/run-engine", RUNNER, "755")
    try:
        guest(["test", "-x", f"{GUEST_ROOT}/bin/uv"])
    except RuntimeError:
        archive = download(UV, "uv-x86_64-unknown-linux-gnu.tar.gz", progress)
        guest(
            [
                "tar",
                "-xzf",
                to_guest_path(archive),
                "-C",
                f"{GUEST_ROOT}/bin",
                "--strip-components=1",
            ]
        )
    write_state(state = "ready", distro = distro_name())


def prepare(progress = None) -> None:
    """Everything before the engine's own packages. Raises ``Waiting`` for a user step."""
    state = wsl_state()
    if state.startswith("blocked"):
        raise RuntimeError(state.partition(": ")[2])
    if state == "restart_required":
        raise Waiting("Restart Windows to finish installing WSL, then click Install again.")
    if state == "missing":
        if progress:
            progress("Approve the Windows prompt to install WSL")
        enable_wsl()
    ensure_distro(progress)


def _uuid_table(output: str) -> dict[int, str]:
    uuids = {}
    for line in output.splitlines():
        index, _, uuid = (part.strip() for part in line.partition(","))
        if index.isdigit() and uuid.startswith("GPU-"):
            uuids[int(index)] = uuid
    return uuids


def gpu_uuids(gpu_ids: list[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 60,
        creationflags = _CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    uuids = _uuid_table(result.stdout)
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in uuids]
    if result.returncode or missing:
        raise RuntimeError("Could not identify the selected GPUs for WSL. Check the NVIDIA driver.")
    return [uuids[gpu_id] for gpu_id in gpu_ids]


def guest_gpu_indices(gpu_ids: list[int]) -> list[int]:
    """Windows and WSL may number the same GPUs differently, so match by UUID. vLLM parses
    CUDA_VISIBLE_DEVICES as integers, so the result is guest indices, used with
    CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA ordinals equal nvidia-smi's."""
    wanted = gpu_uuids(gpu_ids)
    table = _uuid_table(
        guest(
            [
                "sh",
                "-c",
                'PATH="$PATH:/usr/lib/wsl/lib" nvidia-smi --query-gpu=index,uuid --format=csv,noheader',
            ],
            timeout = 120,
        )
    )
    by_uuid = {uuid: index for index, uuid in table.items()}
    if any(uuid not in by_uuid for uuid in wanted):
        raise RuntimeError("WSL cannot see the selected GPU. Update the NVIDIA Windows driver.")
    return [by_uuid[uuid] for uuid in wanted]


def kill_environment(environment: str) -> None:
    """Last resort after the runner: end any process still running from this engine environment."""
    try:
        guest(["pkill", "-KILL", "-f", f"^{environment}/bin/python"], timeout = 60)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        pass


def unregister() -> None:
    """Deletes the private distro and everything in it; never touches the user's own distros."""
    if distro_ready():
        owner = json.loads(guest(["cat", f"{GUEST_ROOT}/owner.json"]))
        if owner.get("distro") != distro_name():
            raise RuntimeError("The WSL environment belongs to another Studio installation.")
    run(["--unregister", distro_name()], timeout = 300)
    shutil.rmtree(host_dir() / "distro", ignore_errors = True)
    write_state(state = "removed")


def summary() -> dict:
    """Status-poll view from the state file only: no wsl.exe call, so polling never boots WSL."""
    state = read_state()
    return {"state": state.get("state"), "distro": state.get("distro")}
