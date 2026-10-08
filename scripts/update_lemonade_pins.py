#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Move studio/lemonade_prebuilt_pins.json to the latest stable Lemonade and FastFlowLM releases.

Every asset is downloaded and hashed here, and must match the digest GitHub publishes for it
when there is one, so a pin only ever names bytes this script has seen. Re-running against
the same releases changes nothing.

Exit codes: 0 = pins current (or updated with --write); 1 = --check found a newer release;
2 = error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

PINS_PATH = Path(__file__).resolve().parents[1] / "studio" / "lemonade_prebuilt_pins.json"
_API = "https://api.github.com"
_TIMEOUT_S = 60
_CHUNK = 1 << 20


def _lemonade_assets(version: str) -> dict[str, str]:
    return {
        "linux-x64": f"lemonade-embeddable-{version}-ubuntu-x64.tar.gz",
        "windows-x64": f"lemonade-embeddable-{version}-windows-x64.zip",
    }


def _fastflowlm_assets(version: str) -> dict[str, str]:
    # The names lemond builds for its /v1/install download, from the bare version.
    return {
        "linux-x64": f"fastflowlm_{version}_linux.tar.gz",
        "windows-x64": f"fastflowlm_{version}_windows_amd64.zip",
    }


# section -> (asset names for a bare version, whether the pin keeps the tag's "v", whether it records sizes)
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


def fetch_latest_release(repo: str) -> dict:
    """The latest stable release; GitHub's /latest skips drafts and prereleases."""
    with urllib.request.urlopen(
        _request(f"{_API}/repos/{repo}/releases/latest"), timeout = _TIMEOUT_S
    ) as r:
        return json.load(r)


def hash_asset(url: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(_request(url), timeout = _TIMEOUT_S) as response:
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def version_key(version: str) -> tuple[int, ...]:
    """11.9.0 < 2026.41.1 and v1.0.3 < v1.0.7; anything else is not a release version."""
    parts = version.lstrip("v").split(".")
    if not parts or not all(part.isdigit() for part in parts):
        raise ValueError(f"{version!r} is not a numeric release version.")
    return tuple(int(part) for part in parts)


def updated_section(
    name: str, section: dict, release: dict, hasher: Callable[[str], tuple[str, int]]
) -> Optional[dict]:
    """The section moved to ``release``, or None when the pin is already that new."""
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
        sha256, size = hasher(asset["browser_download_url"])
        github_digest = asset.get("digest")
        if github_digest and github_digest != f"sha256:{sha256}":
            raise RuntimeError(
                f"{asset_name} hashed to sha256:{sha256}, but GitHub publishes {github_digest}."
            )
        if asset.get("size") is not None and asset["size"] != size:
            raise RuntimeError(f"{asset_name} is {size} bytes, but GitHub lists {asset['size']}.")
        assets[key] = {"name": asset_name, "sha256": sha256}
        if with_size:
            assets[key]["size"] = size
    return {**section, "version": tag if keep_v else bare, "assets": assets}


def update_pins(
    pins: dict,
    releases: Optional[Callable[[str], dict]] = None,
    hasher: Optional[Callable[[str], tuple[str, int]]] = None,
    only: Optional[str] = None,
) -> tuple[dict, list[str]]:
    """New pins and one line per section that moved; the input is not modified."""
    releases = releases or fetch_latest_release
    hasher = hasher or hash_asset
    result = json.loads(json.dumps(pins))
    changes = []
    for name in [only] if only else _SECTIONS:
        section = result[name]
        moved = updated_section(name, section, releases(section["repo"]), hasher)
        if moved is not None:
            changes.append(f"{name}: {section['version']} -> {moved['version']}")
            result[name] = moved
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
