#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Select the newest SemVer-tagged desktop release carrying a required asset."""

from __future__ import annotations

import argparse
import json
import re

import subprocess
import sys
from typing import Any

SEMVER_TAG = re.compile(
    r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def resolve(releases: Any, asset_suffix: str) -> str | None:
    if not isinstance(releases, list):
        raise ValueError("release listing must be a JSON array")

    candidates: list[tuple[str, str]] = []
    for release in releases:
        if not isinstance(release, dict):
            continue
        tag = release.get("tagName")
        created_at = release.get("createdAt")
        assets = release.get("assets")
        if not isinstance(tag, str) or not SEMVER_TAG.fullmatch(tag):
            continue
        if not isinstance(created_at, str) or not isinstance(assets, list):
            continue
        if not any(
            isinstance(asset, dict)
            and isinstance(asset.get("name"), str)
            and asset["name"].endswith(asset_suffix)
            for asset in assets
        ):
            continue
        candidates.append((created_at, tag))

    return max(candidates, default = None)[1] if candidates else None


def fetch_assets(tag: str, repo: str) -> Any:
    """Asset names for one release, including authorized drafts."""
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
        check = False,
        capture_output = True,
        text = True,
    )
    if result.returncode != 0:
        raise ValueError(f"could not inspect assets for {tag}: {result.stderr.strip()}")
    return json.loads(result.stdout).get("assets")


def resolve_newest(releases: Any, asset_suffix: str, fetch) -> str | None:
    """Newest SemVer release carrying the asset, hydrating one release at a time.

    Newest first and stop on the first match: hydrating every entry cost one
    subprocess per release per matrix leg, and let a transient lookup on an
    irrelevant historical release fail a leg whose answer was the newest one.
    """
    if not isinstance(releases, list):
        raise ValueError("release listing must be a JSON array")

    candidates = [
        (release["createdAt"], release["tagName"])
        for release in releases
        if isinstance(release, dict)
        and isinstance(release.get("tagName"), str)
        and SEMVER_TAG.fullmatch(release["tagName"])
        and isinstance(release.get("createdAt"), str)
    ]
    for created_at, tag in sorted(candidates, reverse = True):
        if resolve([{"tagName": tag, "createdAt": created_at, "assets": fetch(tag)}], asset_suffix):
            return tag
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-suffix", required = True)
    parser.add_argument("--repo", required = True)
    args = parser.parse_args()

    try:
        tag = resolve_newest(
            json.load(sys.stdin),
            args.asset_suffix,
            lambda release_tag: fetch_assets(release_tag, args.repo),
        )
    except (json.JSONDecodeError, ValueError) as error:
        print(f"invalid GitHub release listing: {error}", file = sys.stderr)
        return 2

    if tag is None:
        print(
            f"no SemVer v... desktop release contains an asset ending in {args.asset_suffix}",
            file = sys.stderr,
        )
        return 1
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
