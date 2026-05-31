#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Relax strict metadata pins so pip check matches known working single-env stack.

Why:
- data-designer pins huggingface-hub>=1.0.1 and pyarrow<20.
- unsloth/transformers pins huggingface-hub<1.
- studio datasets pins pyarrow>=21.

Runtime works in this app with hub 0.36.x + pyarrow 23.x, but metadata conflicts.
"""

from __future__ import annotations

import importlib.metadata as im
import re
from pathlib import Path

TARGETS = (
    "data-designer",
    "data-designer-engine",
    "data-designer-config",
)

PATCHES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^Requires-Dist: huggingface-hub<2,>=1\.0\.1$", re.MULTILINE),
        "Requires-Dist: huggingface-hub<2,>=0.34.0",
    ),
    (
        re.compile(r"^Requires-Dist: pyarrow<20,>=19\.0\.1$", re.MULTILINE),
        "Requires-Dist: pyarrow>=21.0.0",
    ),
)


def metadata_path(dist_name: str) -> Path | None:
    try:
        dist = im.distribution(dist_name)
    except im.PackageNotFoundError:
        return None
    for f in dist.files or []:
        sf = str(f)
        if sf.endswith(".dist-info/METADATA"):
            return Path(dist.locate_file(f))
    return None


def patch_file(path: Path) -> bool:
    original = path.read_text(encoding = "utf-8")
    updated = original
    for pattern, repl in PATCHES:
        updated = pattern.sub(repl, updated)
    if updated == original:
        return False
    path.write_text(updated, encoding = "utf-8")
    return True


def main() -> int:
    changed = 0
    checked = 0
    for name in TARGETS:
        p = metadata_path(name)
        if p is None:
            continue
        checked += 1
        if patch_file(p):
            changed += 1
    print(f"single-env metadata patch: checked={checked}, changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
