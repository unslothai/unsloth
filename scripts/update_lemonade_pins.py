#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Move studio/lemonade_prebuilt_pins.json to the latest stable Lemonade and FastFlowLM releases.

A release whose Linux build needs a system library newer than LINUX_ABI_LIMITS is skipped for the
newest one that does not, since a lemond that cannot start breaks the NPU for every Linux user on that distro.

Exit codes: 0 = current (or updated with --write); 1 = --check found a newer release; 2 = error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

PINS_PATH = Path(__file__).resolve().parents[1] / "studio" / "lemonade_prebuilt_pins.json"
_API = "https://api.github.com"
_TIMEOUT_S = 60
_CHUNK = 1 << 20
# Newest symbol versions Ubuntu 24.04 ships; FastFlowLM's own Linux build already needs glibc 2.39.
LINUX_ABI_LIMITS = {
    "GLIBC": (2, 39),
    "GLIBCXX": (3, 4, 33),
    "CXXABI": (1, 3, 15),
    "OPENSSL": (3, 0, 9),
}
_SYMBOL_VERSION = re.compile(rb"(GLIBCXX|GLIBC|CXXABI|OPENSSL)_(\d+(?:\.\d+)+)")
Needs = dict[str, tuple[int, ...]]
_NEXT_PAGE = re.compile(r'<([^>]+)>;\s*rel="next"')


def _lemonade_assets(version: str) -> dict[str, str]:
    return {
        "linux-x64": f"lemonade-embeddable-{version}-ubuntu-x64.tar.gz",
        "windows-x64": f"lemonade-embeddable-{version}-windows-x64.zip",
    }


def _fastflowlm_assets(version: str) -> dict[str, str]:
    # The names lemond builds for its /v1/install download.
    return {
        "linux-x64": f"fastflowlm_{version}_linux.tar.gz",
        "windows-x64": f"fastflowlm_{version}_windows_amd64.zip",
    }


# section -> (asset names, pin keeps the tag's "v", pin records sizes)
_SECTIONS: dict[str, tuple[Callable[[str], dict[str, str]], bool, bool]] = {
    "lemonade": (_lemonade_assets, False, True),
    "fastflowlm": (_fastflowlm_assets, True, False),
}


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "unsloth-pins-updater", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith(_API):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers = headers)


def fetch_stable_releases(repo: str) -> list[dict]:
    # Every page: prerelease candidates can push a compatible stable release past the first one.
    releases, url = [], f"{_API}/repos/{repo}/releases?per_page=100"
    while url:
        with urllib.request.urlopen(_request(url), timeout = _TIMEOUT_S) as r:
            releases += json.load(r)
            next_page = _NEXT_PAGE.search(r.headers.get("Link") or "")
        url = next_page.group(1) if next_page else None
    return [r for r in releases if not r.get("draft") and not r.get("prerelease")]


def tar_abi_needs(fileobj) -> Needs:
    """Newest symbol version per LINUX_ABI_LIMITS library that any ELF file in a .tar.gz references."""
    newest: Needs = {}
    with tarfile.open(fileobj = fileobj, mode = "r:gz") as archive:
        for member in archive:
            handle = archive.extractfile(member) if member.isfile() else None
            if handle is None or handle.read(4) != b"\x7fELF":
                continue
            for match in _SYMBOL_VERSION.finditer(handle.read()):
                lib = match.group(1).decode()
                version = tuple(int(part) for part in match.group(2).split(b"."))
                newest[lib] = max(newest.get(lib, ()), version)
    return newest


def hash_asset(url: str) -> tuple[str, int, Optional[Needs]]:
    """sha256, size and, for a Linux tarball, the newest system library versions its binaries need."""
    digest = hashlib.sha256()
    size = 0
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(_request(url), timeout = _TIMEOUT_S) as response:
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                tmp.write(chunk)
                size += len(chunk)
        tmp.seek(0)
        needs = tar_abi_needs(tmp) if url.endswith(".tar.gz") else None
    return digest.hexdigest(), size, needs


class AbiTooNew(RuntimeError):
    """A release whose Linux build cannot start within LINUX_ABI_LIMITS."""


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))


def version_key(version: str) -> tuple[int, ...]:
    """11.9.0 < 2026.41.1 and v1.0.3 < v1.0.7; anything else is not a release version."""
    parts = version.lstrip("v").split(".")
    if not parts or not all(part.isdigit() for part in parts):
        raise ValueError(f"{version!r} is not a numeric release version.")
    return tuple(int(part) for part in parts)


def updated_section(
    name: str,
    section: dict,
    release: dict,
    hasher: Callable[[str], tuple[str, int, Optional[Needs]]],
) -> Optional[dict]:
    tag = str(release["tag_name"])
    bare = tag[1:] if tag.startswith("v") else tag
    asset_names, keep_v, with_size = _SECTIONS[name]
    pinned = section["version"]
    if version_key(tag) <= version_key(pinned):
        return None
    published = {asset["name"]: asset for asset in release.get("assets") or []}
    assets = {}
    for key, asset_name in asset_names(bare).items():
        asset = published.get(asset_name)
        if asset is None:
            raise RuntimeError(f"{section['repo']} {tag} has no {asset_name}.")
        sha256, size, needs = hasher(asset["browser_download_url"])
        github_digest = asset.get("digest")
        # Fail closed: without GitHub's digest the hash proves only what this runner downloaded.
        if not github_digest:
            raise RuntimeError(f"{asset_name}: GitHub publishes no digest; refusing to pin it.")
        if github_digest != f"sha256:{sha256}":
            raise RuntimeError(
                f"{asset_name} hashed to sha256:{sha256}, but GitHub publishes {github_digest}."
            )
        if asset.get("size") is not None and asset["size"] != size:
            raise RuntimeError(f"{asset_name} is {size} bytes, but GitHub lists {asset['size']}.")
        too_new = [
            f"{lib} {_dotted(version)} > {_dotted(LINUX_ABI_LIMITS[lib])}"
            for lib, version in sorted((needs or {}).items())
            if version > LINUX_ABI_LIMITS[lib]
        ]
        if too_new:
            raise AbiTooNew(f"{section['repo']} {tag}: {asset_name} needs {', '.join(too_new)}.")
        assets[key] = {"name": asset_name, "sha256": sha256}
        if with_size:
            assets[key]["size"] = size
    return {**section, "version": tag if keep_v else bare, "assets": assets}


def newer_releases(releases: list[dict], pinned: str) -> list[dict]:
    """Releases with a numeric tag above the pin, newest first."""
    newer = []
    for release in releases:
        try:
            key = version_key(str(release["tag_name"]))
        except ValueError:
            continue
        if key > version_key(pinned):
            newer.append((key, release))
    return [release for _, release in sorted(newer, key = lambda item: item[0], reverse = True)]


def update_pins(
    pins: dict,
    releases: Optional[Callable[[str], list[dict]]] = None,
    hasher: Optional[Callable[[str], tuple[str, int, Optional[Needs]]]] = None,
    only: Optional[str] = None,
) -> tuple[dict, list[str]]:
    releases = releases or fetch_stable_releases
    hasher = hasher or hash_asset
    result = json.loads(json.dumps(pins))
    changes = []
    for name in [only] if only else _SECTIONS:
        section = result[name]
        for release in newer_releases(releases(section["repo"]), section["version"]):
            try:
                moved = updated_section(name, section, release, hasher)
            except AbiTooNew as exc:
                # stderr: the workflow builds the PR title from stdout.
                print(f"skipped: {exc}", file = sys.stderr)
                continue
            changes.append(f"{name}: {section['version']} -> {moved['version']}")
            result[name] = moved
            break
    return result, changes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("--pins", type = Path, default = PINS_PATH)
    parser.add_argument("--only", choices = sorted(_SECTIONS), help = "Move one section only.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action = "store_true", help = "Exit 1 if a newer release exists.")
    mode.add_argument("--write", action = "store_true", help = "Rewrite the pins file.")
    args = parser.parse_args(argv)
    try:
        pins = json.loads(args.pins.read_text(encoding = "utf-8"))
        new_pins, changes = update_pins(pins, only = args.only)
    except Exception as exc:  # noqa: BLE001 -- one line for the CI log, exit 2
        print(f"error: {exc}", file = sys.stderr)
        return 2
    for line in changes or ["Lemonade and FastFlowLM pins are current."]:
        print(line)
    if changes and args.write:
        args.pins.write_text(json.dumps(new_pins, indent = 2) + "\n", encoding = "utf-8")
    return 1 if changes and args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
