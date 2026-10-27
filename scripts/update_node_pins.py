#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Move studio/node_prebuilt_pins.json to the newest Node LTS release of the pinned major.

Moving to a new LTS major stays a reviewed change (studio/NODE_RUNTIME.md); this only takes
patch and minor releases within the pinned line, security releases included.

SHASUMS256.txt must carry a valid detached signature (SHASUMS256.txt.sig) from a Node release key (github.com/nodejs/release-keys),
and every archive is downloaded and hashed against it before the pins change.

Exit codes: 0 = current (or updated with --write); 1 = --check found a newer release; 2 = error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

PINS_PATH = Path(__file__).resolve().parents[1] / "studio" / "node_prebuilt_pins.json"
_DIST = "https://nodejs.org/dist"
_KEYRING = "https://raw.githubusercontent.com/nodejs/release-keys/HEAD/gpg/pubring.kbx"
_TIMEOUT_S = 60
_CHUNK = 1 << 20


def archive_names(version: str) -> list[str]:
    return [
        f"node-v{version}-linux-x64.tar.gz",
        f"node-v{version}-linux-arm64.tar.gz",
        f"node-v{version}-darwin-x64.tar.gz",
        f"node-v{version}-darwin-arm64.tar.gz",
        f"node-v{version}-win-x64.zip",
        f"node-v{version}-win-arm64.zip",
    ]


def version_key(version: str) -> tuple[int, ...]:
    parts = version.lstrip("v").split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{version!r} is not a Node release version.")
    return tuple(int(part) for part in parts)


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-pins-updater"})
    with urllib.request.urlopen(request, timeout = _TIMEOUT_S) as response:
        return response.read()


def hash_url(url: str) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-pins-updater"})
    with urllib.request.urlopen(request, timeout = _TIMEOUT_S) as response:
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_signature(shasums: bytes, signature: bytes, keyring: bytes) -> None:
    """Raise unless `signature` is a good signature over `shasums` by a key in `keyring`."""
    gpg = shutil.which("gpg")
    if not gpg:
        raise RuntimeError(
            "gpg is required to verify SHASUMS256.txt; refusing to pin unverified digests."
        )
    with tempfile.TemporaryDirectory() as home:
        paths = {
            name: os.path.join(home, name)
            for name in ("pubring.kbx", "SHASUMS256.txt", "SHASUMS256.txt.sig")
        }
        for name, data in (
            ("pubring.kbx", keyring),
            ("SHASUMS256.txt", shasums),
            ("SHASUMS256.txt.sig", signature),
        ):
            with open(paths[name], "wb") as handle:
                handle.write(data)
        result = subprocess.run(
            [
                gpg,
                "--homedir",
                home,
                "--no-default-keyring",
                "--keyring",
                paths["pubring.kbx"],
                "--status-fd",
                "1",
                "--verify",
                paths["SHASUMS256.txt.sig"],
                paths["SHASUMS256.txt"],
            ],
            capture_output = True,
            text = True,
            timeout = _TIMEOUT_S,
        )
    if result.returncode != 0 or "[GNUPG:] VALIDSIG" not in result.stdout:
        raise RuntimeError("SHASUMS256.txt is not signed by a Node release key.")


def newest_in_line(index: list[dict], major: int) -> Optional[str]:
    """The newest LTS release of `major` in nodejs.org's dist index."""
    best = None
    for release in index:
        version = str(release.get("version", "")).lstrip("v")
        try:
            key = version_key(version)
        except ValueError:
            continue
        if key[0] == major and release.get("lts") and (best is None or key > version_key(best)):
            best = version
    return best


def update_pins(
    pins: dict,
    fetch: Optional[Callable[[str], bytes]] = None,
    hasher: Optional[Callable[[str], str]] = None,
    verifier: Optional[Callable[[bytes, bytes, bytes], None]] = None,
) -> tuple[dict, list[str]]:
    fetch, hasher, verifier = fetch or _get, hasher or hash_url, verifier or verify_signature
    pinned = str(pins["default_version"])
    target = newest_in_line(json.loads(fetch(f"{_DIST}/index.json")), version_key(pinned)[0])
    if target is None or version_key(target) <= version_key(pinned):
        return pins, []
    shasums = fetch(f"{_DIST}/v{target}/SHASUMS256.txt")
    verifier(shasums, fetch(f"{_DIST}/v{target}/SHASUMS256.txt.sig"), fetch(_KEYRING))
    published = {}
    for line in shasums.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            published[fields[1]] = fields[0]
    digests = {}
    for name in archive_names(target):
        expected = published.get(name)
        if not expected:
            raise RuntimeError(f"SHASUMS256.txt for v{target} lists no {name}.")
        actual = hasher(f"{_DIST}/v{target}/{name}")
        if actual != expected:
            raise RuntimeError(
                f"{name} hashed to {actual}, but the signed SHASUMS256.txt lists {expected}."
            )
        digests[name] = expected
    result = json.loads(json.dumps(pins))
    result["default_version"] = target
    result["versions"] = {target: digests}
    return result, [f"node: {pinned} -> {target}"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("--pins", type = Path, default = PINS_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action = "store_true", help = "Exit 1 if a newer release exists.")
    mode.add_argument("--write", action = "store_true", help = "Rewrite the pins file.")
    args = parser.parse_args(argv)
    try:
        pins = json.loads(args.pins.read_text(encoding = "utf-8"))
        new_pins, changes = update_pins(pins)
    except Exception as exc:  # noqa: BLE001 -- one line for the CI log, exit 2
        print(f"error: {exc}", file = sys.stderr)
        return 2
    for line in changes or ["Node pin is current."]:
        print(line)
    if changes and args.write:
        # Replace atomically so an interrupted run never leaves a truncated pins file.
        tmp = args.pins.with_name(args.pins.name + ".tmp")
        tmp.write_text(json.dumps(new_pins, indent = 2) + "\n", encoding = "utf-8")
        os.replace(tmp, args.pins)
    return 1 if changes and args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
