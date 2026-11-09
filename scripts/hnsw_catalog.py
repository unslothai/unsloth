#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Writes studio/hnsw_prebuilt_pins.json from PyPI's metadata for the pinned usearch + numkong pair;
--check exits 1 when the committed file differs."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PAIR = (("usearch", "2.26.6"), ("numkong", "7.8.6"))
PYTHONS = ("cp311", "cp312", "cp313", "cp314")
MIRROR = "https://github.com/unslothai/unsloth/releases/download/HNSW-Wheels/"
OUT = Path(__file__).resolve().parents[1] / "studio" / "hnsw_prebuilt_pins.json"


def _wheels(name: str, version: str) -> list[dict]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout = 30) as resp:
        files = json.load(resp)["urls"]
    out = []
    for f in files:
        fn = f["filename"]
        if not fn.endswith(".whl"):
            continue
        py, abi = fn[:-4].split("-")[-3:-1]
        # Free-threaded builds (cp314t) need their own probe coverage: left out until then.
        if py not in PYTHONS or abi.endswith("t"):
            continue
        out.append(
            {
                "filename": fn,
                "sha256": f["digests"]["sha256"],
                "size": f["size"],
                "url": f["url"],
            }
        )
    return sorted(out, key = lambda w: w["filename"])


def build() -> dict:
    return {
        "format": 1,
        "mirror": MIRROR,
        "packages": {
            name: {"version": version, "wheels": _wheels(name, version)} for name, version in PAIR
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--check", action = "store_true")
    args = ap.parse_args()
    text = json.dumps(build(), indent = 1, sort_keys = True) + "\n"
    if args.check:
        current = OUT.read_text(encoding = "utf-8") if OUT.is_file() else ""
        if current != text:
            print(f"{OUT} is stale: run scripts/hnsw_catalog.py", file = sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
