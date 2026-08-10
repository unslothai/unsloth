#!/usr/bin/env python3
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


def hydrate_assets(releases: Any, repo: str) -> Any:
    """Fetch asset names for SemVer candidates, including authorized drafts."""
    if not isinstance(releases, list):
        raise ValueError("release listing must be a JSON array")
    for release in releases:
        if not isinstance(release, dict):
            continue
        tag = release.get("tagName")
        if not isinstance(tag, str) or not SEMVER_TAG.fullmatch(tag):
            continue
        result = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
            check = False,
            capture_output = True,
            text = True,
        )
        if result.returncode != 0:
            raise ValueError(f"could not inspect assets for {tag}: {result.stderr.strip()}")
        payload = json.loads(result.stdout)
        release["assets"] = payload.get("assets")
    return releases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-suffix", required = True)
    parser.add_argument("--repo", required = True)
    args = parser.parse_args()

    try:
        releases = hydrate_assets(json.load(sys.stdin), args.repo)
        tag = resolve(releases, args.asset_suffix)
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
