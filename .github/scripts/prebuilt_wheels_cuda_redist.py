#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install a CUDA toolkit subset from NVIDIA's redistributable archives, every byte hash-checked.

Windows ARM64 has no CUDA before 13.4, and its network installer's subpackage names are not the
x64 ones. The redist archives are what NVIDIA itself lists per platform, each with a sha256 in a
release manifest. The manifest's own sha256 is pinned by the caller, so the chain is: pinned
manifest -> per-archive sha256 -> extracted tree. Each archive unpacks to one top-level directory
holding bin/ include/ lib/ (and nvvm/); those are merged into DEST.

Usage: prebuilt_wheels_cuda_redist.py MANIFEST_URL MANIFEST_SHA256 PLATFORM DEST COMPONENT...
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REDIST_BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"


def _fetch(url: str, dest: Path) -> None:
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout = 300) as response, open(dest, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(10 * attempt)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(manifest: dict, platform: str, components: list[str]) -> list[tuple[str, str]]:
    """(relative_path, sha256) per component, refusing any the manifest lacks for PLATFORM."""
    missing = [name for name in components if platform not in manifest.get(name, {})]
    if missing:
        raise SystemExit(f"not in the manifest for {platform}: {', '.join(missing)}")
    return [
        (manifest[name][platform]["relative_path"], manifest[name][platform]["sha256"])
        for name in components
    ]


def merge(archive: Path, dest: Path) -> None:
    """Extract ARCHIVE's single top-level directory into DEST, refusing paths that escape it."""
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        tops = {name.split("/", 1)[0] for name in names}
        if len(tops) != 1:
            raise SystemExit(
                f"{archive.name}: expected one top-level directory, got {sorted(tops)}"
            )
        root = dest.resolve()
        for name in names:
            relative = name.split("/", 1)[1] if "/" in name else ""
            if not relative or relative == "LICENSE":
                continue
            target = (dest / relative).resolve()
            if root not in target.parents:
                raise SystemExit(f"{archive.name}: {name} escapes {dest}")
            target.parent.mkdir(parents = True, exist_ok = True)
            with zf.open(name) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(__doc__, file = sys.stderr)
        return 2
    manifest_url, manifest_sha, platform, dest, *components = argv
    dest_path = Path(dest)
    work = dest_path.parent / f"{dest_path.name}-redist"
    work.mkdir(parents = True, exist_ok = True)
    manifest_file = work / "manifest.json"
    _fetch(manifest_url, manifest_file)
    got = _sha256(manifest_file)
    if got != manifest_sha:
        raise SystemExit(f"manifest sha256 {got}, pinned {manifest_sha}")
    manifest = json.loads(manifest_file.read_text(encoding = "utf-8"))
    print(f"CUDA {manifest.get('release_label')} ({manifest.get('release_date')}) for {platform}")
    dest_path.mkdir(parents = True, exist_ok = True)
    for relative_path, sha in plan(manifest, platform, components):
        archive = work / Path(relative_path).name
        _fetch(REDIST_BASE + relative_path, archive)
        got = _sha256(archive)
        if got != sha:
            raise SystemExit(f"{archive.name}: sha256 {got}, manifest says {sha}")
        merge(archive, dest_path)
        archive.unlink()
        print(f"  {archive.name} ok")
    shutil.rmtree(work, ignore_errors = True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
