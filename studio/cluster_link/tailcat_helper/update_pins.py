# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Writes studio/unsloth_link_pins.json from a release's archives and SHA256SUMS, or checks them.

gh release download unsloth-link-v1 -R unslothai/unsloth -D rel
python update_pins.py rel --tag unsloth-link-v1          # rewrite the pins
python update_pins.py rel --check                        # exit 1 unless rel matches the pins"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from package import TARGETS, archived_binary, asset_name

PINS = Path(__file__).resolve().parents[2] / "unsloth_link_pins.json"
REPO = "unslothai/unsloth"


def _sums(folder: Path) -> dict:
    sums = {}
    for line in (folder / "SHA256SUMS").read_text(encoding = "utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        if digest:
            sums[name.lstrip("./")] = digest.lower()
    return sums


def collect(folder: Path) -> dict:
    sums, assets = _sums(folder), {}
    for target in TARGETS:
        name = asset_name(target)
        data = (folder / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if sums.get(name) != digest:
            raise SystemExit(
                f"{name}: sha256 {digest} does not match SHA256SUMS ({sums.get(name)})"
            )
        assets[target] = {
            "name": name,
            "sha256": digest,
            "size": len(data),
            "binary_sha256": hashlib.sha256(archived_binary(target, data)).hexdigest(),
        }
    return assets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type = Path)
    ap.add_argument("--tag")
    ap.add_argument("--repo", default = REPO)
    ap.add_argument("--check", action = "store_true")
    args = ap.parse_args()
    assets = collect(args.folder)
    if args.check:
        pinned = json.loads(PINS.read_text(encoding = "utf-8"))["assets"]
        bad = [t for t in TARGETS if pinned.get(t) != assets[t]]
        for target in bad:
            print(f"{target}: built {assets[target]}, pinned {pinned.get(target)}")
        return 1 if bad else 0
    if not args.tag:
        ap.error("--tag is required to rewrite the pins")
    pins = {
        "schema_version": 1,
        "comment": "Trust anchor for install_unsloth_link.py: sha256 and size of each release archive "
        "built by .github/workflows/unsloth-link.yml, and the sha256 of the executable inside it. "
        "Regenerate with "
        "studio/cluster_link/tailcat_helper/update_pins.py from the release's own assets.",
        "repo": args.repo,
        "tag": args.tag,
        "assets": assets,
    }
    PINS.write_text(json.dumps(pins, indent = 2) + "\n", encoding = "utf-8")
    print(f"wrote {PINS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
