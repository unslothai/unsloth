# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install the pinned MXC sandbox executor for Windows tool isolation.

Run at Studio setup time, never during a tool call. A tool call that could
download its own sandbox executor would be choosing its own boundary.

MXC publishes signed native binaries two ways: a 358MB `mxc-release-binaries.zip`
on the GitHub release, and the same per-arch binaries inside the npm tarball at
`package/bin/<arch>/`. This uses the npm tarball, which is 26MB, because only
`wxc-exec.exe` and its helpers are wanted and the registry serves the tarball
over plain HTTPS with an integrity digest. Node is NOT required: the tarball is
a gzipped tar and is unpacked with the standard library.

Usage:
    python studio/install_mxc_runtime.py [--dest DIR] [--verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

# Pinned. An upgrade is a deliberate change that re-runs the Windows
# qualification job, not something that drifts in from the registry.
MXC_VERSION = "0.8.0"
REGISTRY_URL = f"https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{MXC_VERSION}.tgz"

# What the Windows backend actually launches, plus the helpers MXC's own
# ProcessContainer path expects to find beside it.
REQUIRED = ("wxc-exec.exe",)
OPTIONAL = (
    "wxc-host-prep.exe",
    "winhttp-proxy-shim.exe",
    "plm.exe",
    "mxc-diagnostic-console.exe",
)


def arch_dir() -> str:
    machine = (os.environ.get("PROCESSOR_ARCHITECTURE") or "").lower()
    if "arm64" in machine or "arm64" in (sys.platform or ""):
        return "arm64"
    return "x64"


def default_dest() -> str:
    override = os.environ.get("UNSLOTH_MXC_DIR")
    if override:
        return override
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
        from utils.studio_paths import studio_home

        return os.path.join(str(studio_home()), "mxc")
    except Exception:  # noqa: BLE001 - setup can run before the backend imports
        return os.path.join(os.path.expanduser("~"), ".unsloth", "mxc")


def download(url: str, into: str) -> str:
    target = os.path.join(into, "mxc-sdk.tgz")
    with urllib.request.urlopen(url, timeout = 300) as response:  # noqa: S310 - pinned https
        with open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def extract(tarball: str, arch: str, dest: str) -> list[str]:
    """Pull just the binaries for this architecture out of the tarball.

    Members are matched by exact name and written by basename, so a crafted
    tarball cannot traverse out of the destination.
    """
    wanted = {f"package/bin/{arch}/{name}": name for name in (*REQUIRED, *OPTIONAL)}
    written: list[str] = []
    os.makedirs(dest, exist_ok = True)
    with tarfile.open(tarball, "r:gz") as archive:
        for member in archive.getmembers():
            name = wanted.get(member.name)
            if name is None or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            out = os.path.join(dest, name)
            with source, open(out, "wb") as handle:
                shutil.copyfileobj(source, handle)
            written.append(name)
    return written


def install(dest: str) -> dict:
    arch = arch_dir()
    with tempfile.TemporaryDirectory() as scratch:
        tarball = download(REGISTRY_URL, scratch)
        tarball_sha = digest(tarball)
        written = extract(tarball, arch, dest)

    missing = [name for name in REQUIRED if name not in written]
    if missing:
        raise SystemExit(
            f"the MXC tarball did not contain {', '.join(missing)} for {arch}. "
            "Windows tool isolation cannot be installed."
        )

    executor = os.path.join(dest, "wxc-exec.exe")
    return {
        "version": MXC_VERSION,
        "arch": arch,
        "dest": dest,
        "executor": executor,
        "executor_sha256": digest(executor),
        "tarball_sha256": tarball_sha,
        "installed": sorted(written),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--dest", default = None, help = "where to install (default: Studio home)")
    parser.add_argument("--verify-only", action = "store_true",
                        help = "report an existing install without downloading")
    args = parser.parse_args()

    dest = args.dest or default_dest()
    if args.verify_only:
        executor = os.path.join(dest, "wxc-exec.exe")
        present = os.path.isfile(executor)
        print(json.dumps({
            "present": present,
            "executor": executor,
            "executor_sha256": digest(executor) if present else None,
        }, indent = 2))
        return 0 if present else 1

    print(json.dumps(install(dest), indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
