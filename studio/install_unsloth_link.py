# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installs the pinned ``unsloth-link`` tailcat helper on first need.

studio/unsloth_link_pins.json is the only trust anchor: an archive whose size or sha256 differs
from its pin is never extracted or run, and an installed executable whose sha256 differs is not used."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

_STUDIO_DIR = Path(__file__).resolve().parent
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))

from prebuilt_core import BusyInstallConflict, install_lock  # noqa: E402

PINS_PATH = _STUDIO_DIR / "unsloth_link_pins.json"
ENV_RELEASE_BASE = "UNSLOTH_LINK_RELEASE_BASE"
BINARY = "unsloth-link.exe" if os.name == "nt" else "unsloth-link"
LICENSE = "LICENSE.tailcat"
MARKER = "INSTALL.json"
LOCK_TIMEOUT_S = 300
DOWNLOAD_ATTEMPTS = 3
SMOKE_TIMEOUT_S = 30

_ARCH = {"x86_64": "amd64", "amd64": "amd64", "x64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
_OS = {"linux": "linux", "darwin": "darwin", "windows": "windows"}


class InstallError(RuntimeError):
    """The helper could not be installed; the message says what to do."""


def host_target(system: Optional[str] = None, machine: Optional[str] = None) -> Optional[str]:
    os_name = _OS.get((system or platform.system()).lower())
    arch = _ARCH.get((machine or platform.machine()).lower())
    return f"{os_name}-{arch}" if os_name and arch else None


def load_pins(path: Path = PINS_PATH) -> dict:
    pins = json.loads(Path(path).read_text(encoding = "utf-8"))
    if pins.get("schema_version") != 1 or not isinstance(pins.get("assets"), dict):
        raise InstallError(f"{path.name}: unsupported pins file")
    for target, asset in pins["assets"].items():
        name, digest, size = asset.get("name"), asset.get("sha256"), asset.get("size")
        exe = asset.get("binary_sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(exe, str)
            or len(exe) != 64
            or not isinstance(size, int)
            or size <= 0
        ):
            raise InstallError(f"{path.name}: bad entry for {target}")
    return pins


def install_root() -> Path:
    try:
        from studio.cluster_link.paths import _studio_root
    except ImportError:
        from cluster_link.paths import _studio_root
    return _studio_root() / "unsloth-link"


def _pinned(pins: Optional[dict]) -> tuple:
    pins = pins or load_pins()
    target = host_target()
    asset = pins["assets"].get(target) if target else None
    if asset is None:
        raise InstallError(
            f"unsloth-link has no build for {platform.system()} {platform.machine()}, so machines on "
            "other networks cannot be paired here (pairing on the local network still works)"
        )
    return pins, target, asset


_VERIFIED: dict = {}  # (path, size, mtime_ns) -> sha256, so a lookup hashes the executable once


def _binary_matches(binary: Path, digest: str) -> bool:
    try:
        st = binary.lstat()
        if not stat.S_ISREG(st.st_mode):
            return False
        key = (str(binary), st.st_size, st.st_mtime_ns)
        if key not in _VERIFIED:
            h = hashlib.sha256()
            with open(binary, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            _VERIFIED[key] = h.hexdigest()
        return _VERIFIED[key] == digest
    except OSError:
        return False


def installed_binary(pins: Optional[dict] = None) -> Optional[Path]:
    """The verified install for the current pins (its executable hashed against the pin), or None.
    Never downloads."""
    try:
        pins, target, asset = _pinned(pins)
    except (InstallError, OSError, ValueError):
        return None
    folder = install_root() / pins["tag"]
    try:
        marker = json.loads((folder / MARKER).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    binary = folder / BINARY
    if (
        marker.get("sha256") == asset["sha256"]
        and marker.get("target") == target
        and _binary_matches(binary, asset["binary_sha256"])
    ):
        return binary
    return None


def asset_url(pins: dict, asset: dict) -> str:
    base = (os.environ.get(ENV_RELEASE_BASE) or "").strip()
    base = base or f"https://github.com/{pins['repo']}/releases/download/{pins['tag']}"
    return f"{base.rstrip('/')}/{asset['name']}"


def _fetch(url: str, dest: Path, asset: dict) -> None:
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout = 60) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 20):
            size += len(chunk)
            if size > asset["size"]:
                raise InstallError(f"{asset['name']} is larger than its pin; refusing it")
            digest.update(chunk)
            out.write(chunk)
    if size != asset["size"]:
        raise InstallError(
            f"{asset['name']} is {size} bytes, its pin says {asset['size']}; refusing it"
        )
    if digest.hexdigest() != asset["sha256"]:
        raise InstallError(f"{asset['name']} does not match its pinned sha256; refusing to run it")


def _download(url: str, dest: Path, asset: dict) -> None:
    last: Optional[BaseException] = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            _fetch(url, dest, asset)
            return
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code not in (408, 429):
                break
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(0.5 * (2**attempt))
    if isinstance(last, urllib.error.HTTPError) and last.code == 404:
        raise InstallError(
            f"{asset['name']} is not published yet; point UNSLOTH_LINK_BIN at a copy of the helper"
        )
    raise InstallError(
        f"could not download unsloth-link ({last}); connect this machine to the internet once, "
        "or point UNSLOTH_LINK_BIN at a copy of the helper"
    )


def _extract(archive: Path, dest: Path) -> None:
    wanted = {BINARY, LICENSE}
    found = set()

    def take(name: str, data: Callable[[], bytes]) -> None:
        if name not in wanted or name in found:
            raise InstallError(f"unexpected entry {name!r} in {archive.name}")
        (dest / name).write_bytes(data())
        found.add(name)

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise InstallError(f"unexpected entry {info.filename!r} in {archive.name}")
                take(info.filename, lambda info = info: zf.read(info))
    else:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    raise InstallError(f"unexpected entry {member.name!r} in {archive.name}")
                take(member.name, lambda member = member: tar.extractfile(member).read())
    if BINARY not in found:
        raise InstallError(f"{archive.name} has no {BINARY}")


def _smoke(binary: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            out = subprocess.run(
                [
                    str(binary),
                    "genkey",
                    "--kind",
                    "client",
                    "--path",
                    str(Path(tmp) / "client.key"),
                ],
                capture_output = True,
                text = True,
                timeout = SMOKE_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise InstallError(f"the downloaded unsloth-link does not run here: {exc}") from exc
    if out.returncode != 0 or not out.stdout.strip().startswith("nodekey:"):
        raise InstallError(
            f"the downloaded unsloth-link does not run here: {(out.stderr or '').strip()}"
        )


def ensure(pins: Optional[dict] = None, log: Callable[[str], None] = lambda _msg: None) -> Path:
    """Path of the verified helper, downloading it the first time."""
    pins, target, asset = _pinned(pins)
    found = installed_binary(pins)
    if found is not None:
        return found
    root = install_root()
    root.mkdir(parents = True, exist_ok = True)
    try:
        with install_lock(root / ".install.lock", timeout = LOCK_TIMEOUT_S):
            found = installed_binary(pins)
            if found is not None:
                return found
            log(f"Downloading the unsloth-link helper ({asset['size'] / 2**20:.1f} MiB)...")
            staging = Path(tempfile.mkdtemp(prefix = ".staging-", dir = root))
            try:
                archive = staging / asset["name"]
                _download(asset_url(pins, asset), archive, asset)
                unpacked = staging / "pkg"
                unpacked.mkdir()
                _extract(archive, unpacked)
                binary = unpacked / BINARY
                if not _binary_matches(binary, asset["binary_sha256"]):
                    raise InstallError(f"{BINARY} in {asset['name']} does not match its pin")
                binary.chmod(0o755)
                _smoke(binary)
                (unpacked / MARKER).write_text(
                    json.dumps({"tag": pins["tag"], "target": target, "sha256": asset["sha256"]}),
                    encoding = "utf-8",
                )
                final = root / pins["tag"]
                if final.exists():
                    shutil.rmtree(final)
                os.replace(unpacked, final)
            finally:
                shutil.rmtree(staging, ignore_errors = True)
    except BusyInstallConflict as exc:
        raise InstallError(f"another unsloth-link install is still running: {exc}") from exc
    return final / BINARY


if __name__ == "__main__":
    try:
        print(ensure(log = print))
    except InstallError as exc:
        sys.exit(f"error: {exc}")
