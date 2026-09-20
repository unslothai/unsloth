#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Install Microsoft's pinned WXC v0.8.0 executable for Studio on Windows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
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
        headers={"User-Agent": "unsloth-studio-mxc-prebuilt"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok=True)
        raise MxcInstallError(
            f"could not download the pinned Microsoft MXC release: {exc}"
        ) from exc


def _already_installed(install_dir: Path) -> bool:
    try:
        mxc_runtime._validate_runtime(install_dir)
    except mxc_runtime.MxcRuntimeUnavailable:
        return False
    return True


def _validate_archive_entries(bundle: zipfile.ZipFile) -> zipfile.ZipInfo:
    seen: set[str] = set()
    approved: list[zipfile.ZipInfo] = []
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
        if name == mxc_runtime.RELEASE_MEMBER:
            approved.append(entry)
    if len(approved) != 1:
        raise MxcInstallError("the Microsoft MXC archive does not contain one approved WXC binary")
    entry = approved[0]
    if entry.is_dir() or entry.file_size != mxc_runtime.WXC_EXEC_SIZE:
        raise MxcInstallError("the approved WXC archive member has an unexpected shape or size")
    return entry


def install_mxc_release(install_dir: Path) -> bool:
    """Install the approved WXC binary; return ``True`` only when files changed."""
    if sys.platform != "win32":
        raise MxcInstallError("the MXC runtime installer is Windows-only")
    if mxc_runtime._expected_architecture() != "x86_64":
        raise MxcInstallError("the Microsoft MXC release supports Windows x86-64 only")

    install_dir = install_dir.expanduser().resolve()
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    with install_lock(install_lock_path(install_dir)):
        if _already_installed(install_dir):
            return False

        stage = Path(tempfile.mkdtemp(prefix=f".{install_dir.name}-", dir=install_dir.parent))
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
                    entry = _validate_archive_entries(bundle)
                    executable = stage / "wxc-exec.exe"
                    with bundle.open(entry) as source, executable.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
            except (OSError, zipfile.BadZipFile) as exc:
                raise MxcInstallError("the pinned Microsoft MXC archive is invalid") from exc
            archive.unlink()
            mxc_runtime._validate_runtime(stage)
            swap_into_place(stage, install_dir)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        changed = install_mxc_release(args.install_dir)
    except BusyInstallConflict as exc:
        print(f"[mxc-prebuilt] install blocked by an active MXC process: {exc}", file=sys.stderr)
        return 3
    except (MxcInstallError, mxc_runtime.MxcRuntimeUnavailable) as exc:
        print(f"[mxc-prebuilt] {exc}", file=sys.stderr)
        return 1
    print("[mxc-prebuilt] installed and validated" if changed else "[mxc-prebuilt] already matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
