#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install pinned Lemonade binaries when the user enables NPU support.

Verify the archive before extraction and pin FastFlowLM's version and checksums in
resources/backend_versions.json for lemond's /v1/install download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

import prebuilt_core as core  # noqa: E402
from backend.utils.auth_safe import auth_safe_open  # noqa: E402

PINS_PATH = Path(_STUDIO_DIR) / "lemonade_prebuilt_pins.json"
MARKER_NAME = "UNSLOTH_LEMONADE_INFO.json"
FLM_MARKER_NAME = "UNSLOTH_FLM_INFO.json"
# Stands in for lemond on Windows: lemond.exe is unsigned (Smart App Control), flm.exe is signed.
FLM_HOST = Path(_STUDIO_DIR) / "backend" / "core" / "inference" / "flm_host.py"
_CHUNK = 1 << 20
_DOWNLOAD_TIMEOUT_S = 60

ProgressCallback = Callable[[int, Optional[int]], None]


class LemonadeInstallCancelled(RuntimeError):
    pass


def load_pins(path: Path = PINS_PATH) -> dict:
    with open(path, encoding = "utf-8") as handle:
        pins = json.load(handle)
    if pins.get("schema_version") != 1:
        raise RuntimeError(f"{path} has an unsupported schema_version.")
    return pins


def host_asset_key() -> Optional[str]:
    """The pin key for this OS and CPU, or None where Lemonade publishes no FastFlowLM build."""
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return None
    if sys.platform.startswith("linux"):
        return "linux-x64"
    if sys.platform == "win32":
        return "windows-x64"
    return None


def lemond_name() -> str:
    return "lemond.exe" if sys.platform == "win32" else "lemond"


def install_dir(root: Path, pins: Optional[dict] = None) -> Path:
    pins = pins or load_pins()
    return Path(root) / pins["lemonade"]["version"]


def _read_marker(directory: Path) -> Optional[dict]:
    try:
        marker = json.loads((directory / MARKER_NAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return marker if isinstance(marker, dict) else None


def direct_flm() -> bool:
    """Run FastFlowLM without Lemonade (Windows); UNSLOTH_NPU_LEMONADE=1 keeps lemond."""
    return sys.platform == "win32" and os.environ.get("UNSLOTH_NPU_LEMONADE", "").strip() != "1"


def flm_dir(root: Path) -> Path:
    """Where lemond's /v1/install puts FastFlowLM, and where Studio looks for flm."""
    return Path(root) / "cache" / "bin" / "flm" / "npu"


def installed_flm_host(root: Path, pins: Optional[dict] = None) -> Optional[Path]:
    """flm_host.py when the pinned FastFlowLM is installed directly, else None."""
    pins = pins or load_pins()
    key = host_asset_key()
    target = flm_dir(root)
    try:
        marker = json.loads((target / FLM_MARKER_NAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        marker = None
    asset = pins["fastflowlm"]["assets"].get(key or "")
    if not isinstance(marker, dict) or not asset or marker.get("sha256") != asset["sha256"]:
        return None
    return (
        FLM_HOST if (target / ("flm.exe" if sys.platform == "win32" else "flm")).is_file() else None
    )


def installed_lemond(root: Path, pins: Optional[dict] = None) -> Optional[Path]:
    """The lemond binary of a complete install matching the pin, else None."""
    pins = pins or load_pins()
    if direct_flm():
        return installed_flm_host(root, pins)
    key = host_asset_key()
    if key is None:
        return None
    target = install_dir(root, pins)
    marker = _read_marker(target)
    if marker is None:
        return None
    asset = pins["lemonade"]["assets"][key]
    if marker.get("sha256") != asset["sha256"] or marker.get("asset") != asset["name"]:
        return None
    # A FastFlowLM-only pin change must rewrite backend_versions.json, whose version lemond installs.
    if marker.get("fastflowlm") != pins["fastflowlm"]:
        return None
    binary = target / lemond_name()
    return binary if binary.is_file() else None


def installed_version(lemond: Path) -> Optional[str]:
    """The Lemonade version recorded beside an installed lemond, or None."""
    marker = _read_marker(lemond.parent)
    version = marker.get("version") if marker else None
    return str(version) if version else None


def _download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    progress: Optional[ProgressCallback],
    cancel: Optional[threading.Event],
) -> None:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
    with (
        auth_safe_open(request, timeout = _DOWNLOAD_TIMEOUT_S) as response,
        open(destination, "wb") as handle,
    ):
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        done = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise LemonadeInstallCancelled("Lemonade download cancelled.")
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise core.ReleaseIntegrityError(
            f"{url} has sha256 {actual}, expected {expected_sha256}; refusing to install it."
        )


def _pin_fastflowlm_checksum(resources: Path, pins: dict) -> None:
    """Pin lemond's FastFlowLM download to the version and digests Unsloth tested."""
    path = resources / "backend_versions.json"
    versions = json.loads(path.read_text(encoding = "utf-8"))
    flm = pins["fastflowlm"]
    versions.setdefault("flm", {})["npu"] = flm["version"]
    github = versions.setdefault("checksums", {}).setdefault("github", {})
    by_tag = github.setdefault(flm["repo"], {}).setdefault(flm["version"], {})
    for asset in flm["assets"].values():
        by_tag[asset["name"]] = f"sha256:{asset['sha256']}"
    core.atomic_write_bytes(path, (json.dumps(versions, indent = 2) + "\n").encode("utf-8"))


def install(
    root: Path,
    *,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
    pins_path: Path = PINS_PATH,
) -> Path:
    """Install pinned lemond under root/<version>, reusing a matching install.

    Verify and extract in a temporary directory before replacing the installation.
    """
    pins = load_pins(pins_path)
    key = host_asset_key()
    if key is None:
        raise RuntimeError(
            "Lemonade publishes FastFlowLM builds for Windows and Linux on x86_64 only."
        )
    root = Path(root)
    if direct_flm():
        return _install_flm_direct(root, pins, key, progress, cancel)
    target = install_dir(root, pins)
    with core.install_lock(core.install_lock_path(target)):
        existing = installed_lemond(root, pins)
        if existing is not None:
            return existing
        asset = pins["lemonade"]["assets"][key]
        url = core.release_asset_download_url(
            pins["lemonade"]["repo"], f"v{pins['lemonade']['version']}", asset["name"]
        )
        root.mkdir(parents = True, exist_ok = True)
        with tempfile.TemporaryDirectory(prefix = ".lemonade-install-", dir = root) as scratch:
            scratch_dir = Path(scratch)
            archive = scratch_dir / asset["name"]
            _download(
                url, archive, expected_sha256 = asset["sha256"], progress = progress, cancel = cancel
            )
            extracted = scratch_dir / "extracted"
            extracted.mkdir()
            core.extract_archive(archive, extracted)
            core.restore_tar_exec_bits(archive, extracted)
            binaries = sorted(extracted.rglob(lemond_name()))
            if not binaries:
                raise RuntimeError(f"{asset['name']} does not contain {lemond_name()}.")
            payload = binaries[0].parent
            _pin_fastflowlm_checksum(payload / "resources", pins)
            (payload / MARKER_NAME).write_text(
                json.dumps(
                    {
                        "version": pins["lemonade"]["version"],
                        "asset": asset["name"],
                        "sha256": asset["sha256"],
                        "fastflowlm": pins["fastflowlm"],
                    },
                    indent = 2,
                )
                + "\n",
                encoding = "utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(payload, target)
    binary = installed_lemond(root, pins)
    if binary is None:
        raise RuntimeError(f"Lemonade install at {target} did not verify.")
    return binary


def _install_flm_direct(
    root: Path,
    pins: dict,
    key: str,
    progress: Optional[ProgressCallback],
    cancel: Optional[threading.Event],
) -> Path:
    """Install the pinned FastFlowLM where lemond would, verified, with no Lemonade at all."""
    target = flm_dir(root)
    with core.install_lock(core.install_lock_path(target)):
        existing = installed_flm_host(root, pins)
        if existing is not None:
            return existing
        flm = pins["fastflowlm"]
        asset = flm["assets"][key]
        url = core.release_asset_download_url(flm["repo"], flm["version"], asset["name"])
        target.parent.mkdir(parents = True, exist_ok = True)
        with tempfile.TemporaryDirectory(prefix = ".flm-install-", dir = target.parent) as scratch:
            archive = Path(scratch) / asset["name"]
            _download(
                url, archive, expected_sha256 = asset["sha256"], progress = progress, cancel = cancel
            )
            payload = Path(scratch) / "payload"
            payload.mkdir()
            core.extract_archive(archive, payload)
            core.restore_tar_exec_bits(archive, payload)
            (payload / FLM_MARKER_NAME).write_text(
                json.dumps(
                    {"version": flm["version"], "asset": asset["name"], "sha256": asset["sha256"]}
                )
                + "\n",
                encoding = "utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(payload, target)
    host = installed_flm_host(root, pins)
    if host is None:
        raise RuntimeError(f"FastFlowLM install at {target} did not verify.")
    return host


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type = Path,
        default = Path.home() / ".unsloth" / "studio" / "lemonade",
        help = "Directory that holds one subdirectory per Lemonade version.",
    )
    args = parser.parse_args(argv)
    print(install(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
