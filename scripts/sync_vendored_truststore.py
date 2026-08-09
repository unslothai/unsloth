#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Re-vendor studio/backend/vendor/truststore from a published release.

Maintainer tool, run when bumping the pin. It downloads the wheel straight from
PyPI, checks the hash PyPI publishes for it, then replaces the tree and rewrites
truststore_manifest.json, which tests/test_vendored_truststore.py verifies.

    python scripts/sync_vendored_truststore.py --version 0.10.5

Read upstream's changelog first: a 0.x minor is where truststore has changed
verification behaviour, and here that applies process-wide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "studio" / "backend" / "vendor"
_PACKAGE = "truststore"
# Ours, not upstream's: prose we may reword, and the manifest cannot hash itself.
_UNPINNED = {"README.md", "truststore_manifest.json"}


def _release(version: str) -> dict:
    url = f"https://pypi.org/pypi/{_PACKAGE}/{version}/json"
    with urllib.request.urlopen(url) as response:  # noqa: S310
        payload = json.load(response)
    wheels = [f for f in payload["urls"] if f["packagetype"] == "bdist_wheel"]
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel for {version}, got {len(wheels)}")
    return wheels[0]


def _fetch(wheel: dict, into: Path) -> Path:
    path = into / wheel["filename"]
    with urllib.request.urlopen(wheel["url"]) as response:  # noqa: S310
        path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != wheel["digests"]["sha256"]:
        raise SystemExit(
            f"wheel hash mismatch: got {digest}, PyPI says {wheel['digests']['sha256']}"
        )
    return path


def _write_manifest(version: str, wheel: dict) -> None:
    files = {
        path.relative_to(_VENDOR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_VENDOR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.name not in _UNPINNED
    }
    manifest = {
        "package": _PACKAGE,
        "version": version,
        "license": "MIT",
        "source": f"https://pypi.org/project/{_PACKAGE}/{version}/",
        "wheel": wheel["filename"],
        "wheel_sha256": wheel["digests"]["sha256"],
        "files": files,
    }
    (_VENDOR / "truststore_manifest.json").write_text(
        json.dumps(manifest, indent = 2, sort_keys = True) + "\n", encoding = "utf-8"
    )
    print(f"manifest rewritten: {len(files)} files at {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--version", required = True, help = "release to vendor, e.g. 0.10.5")
    args = parser.parse_args()

    wheel = _release(args.version)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(_fetch(wheel, work)) as archive:
            archive.extractall(work / "unpacked")
        unpacked = work / "unpacked"

        shutil.rmtree(_VENDOR / _PACKAGE, ignore_errors = True)
        shutil.copytree(unpacked / _PACKAGE, _VENDOR / _PACKAGE)
        # dist-info keeps the licence out of the package; put it beside the copy.
        licenses = list(unpacked.glob("*.dist-info/licenses/LICENSE")) or list(
            unpacked.glob("*.dist-info/LICENSE")
        )
        if not licenses:
            raise SystemExit("no LICENSE in the wheel; MIT requires shipping it")
        shutil.copyfile(licenses[0], _VENDOR / "LICENSE")

    _write_manifest(args.version, wheel)
    print(f"vendored {_PACKAGE} {args.version}; update the version in {_VENDOR / 'README.md'}")


if __name__ == "__main__":
    main()
