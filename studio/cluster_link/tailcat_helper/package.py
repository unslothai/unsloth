# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Packs build.sh's per-target dirs into byte-reproducible release archives plus SHA256SUMS.

Usage: package.py DIST_DIR OUT_DIR. Fixed mtimes, owners and member order, so the same binaries
always give the same archive digests that studio/unsloth_link_pins.json pins."""

from __future__ import annotations

import gzip
import hashlib
import io
import sys
import tarfile
import zipfile
from pathlib import Path

TARGETS = (
    "linux-amd64",
    "linux-arm64",
    "darwin-amd64",
    "darwin-arm64",
    "windows-amd64",
    "windows-arm64",
)
LICENSE = "LICENSE.tailcat"


def asset_name(target: str) -> str:
    return f"unsloth-link-{target}" + (".zip" if target.startswith("windows") else ".tar.gz")


def binary_name(target: str) -> str:
    return "unsloth-link.exe" if target.startswith("windows") else "unsloth-link"


def _members(src: Path, target: str):
    yield binary_name(target), (src / binary_name(target)).read_bytes(), 0o755
    yield LICENSE, (src / LICENSE).read_bytes(), 0o644


def _tar_gz(src: Path, target: str) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj = raw, mode = "w", format = tarfile.USTAR_FORMAT) as tar:
        for name, data, mode in _members(src, target):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), mode, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(filename = "", mode = "wb", fileobj = out, mtime = 0, compresslevel = 9) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def _zip(src: Path, target: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel = 9) as zf:
        for name, data, mode in _members(src, target):
            info = zipfile.ZipInfo(name, date_time = (1980, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | mode) << 16
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return out.getvalue()


def archived_binary(target: str, data: bytes) -> bytes:
    """The executable inside a release archive, whose sha256 the pins also carry."""
    if target.startswith("windows"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return zf.read(binary_name(target))
    with tarfile.open(fileobj = io.BytesIO(data), mode = "r:gz") as tar:
        return tar.extractfile(binary_name(target)).read()


def package(dist: Path, out: Path) -> dict:
    out.mkdir(parents = True, exist_ok = True)
    sums = {}
    for target in TARGETS:
        src = dist / target
        data = _zip(src, target) if target.startswith("windows") else _tar_gz(src, target)
        name = asset_name(target)
        (out / name).write_bytes(data)
        sums[name] = hashlib.sha256(data).hexdigest()
    (out / "SHA256SUMS").write_bytes(
        "".join(f"{d}  {n}\n" for n, d in sorted(sums.items())).encode()
    )
    (out / LICENSE).write_bytes((dist / TARGETS[0] / LICENSE).read_bytes())
    return sums


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: package.py DIST_DIR OUT_DIR")
    for name, digest in sorted(package(Path(sys.argv[1]), Path(sys.argv[2])).items()):
        print(f"{digest}  {name}")
