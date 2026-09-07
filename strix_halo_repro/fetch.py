#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Download one file, record what arrived.

For inputs that are not release archives and not Hub files: a single library
supplied by hand, for instance. Records size and sha256 so the report can name
exactly which artifact was measured, and so a swapped file is visible without
trusting the URL.

--zip-member takes one file out of a downloaded archive. The hash that is
checked is then the MEMBER's, not the archive's: a vendor who re-zips the same
library changes the archive hash and nothing else, and it is the library that
gets loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--dest", required = True, help = "file path to write")
    ap.add_argument("--expect-sha256", default = "")
    ap.add_argument(
        "--zip-member",
        default = "",
        help = "extract this member from the download and write IT to --dest",
    )
    ap.add_argument("--out", default = "")
    a = ap.parse_args()

    dest = Path(a.dest)
    dest.parent.mkdir(parents = True, exist_ok = True)
    landing = dest.with_suffix(dest.suffix + ".download") if a.zip_member else dest
    with urllib.request.urlopen(a.url, timeout = 600) as r, open(landing, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    info = {"url": a.url, "path": str(dest)}
    if a.zip_member:
        info["archive_bytes"] = landing.stat().st_size
        info["archive_sha256"] = hashlib.sha256(landing.read_bytes()).hexdigest()
        try:
            with zipfile.ZipFile(landing) as zf:
                names = zf.namelist()
                info["archive_members"] = names
                # Match on the base name: an archive is free to carry its own
                # directory layout, and the caller asked for a library, not a path.
                hit = [n for n in names if n.rsplit("/", 1)[-1].lower() == a.zip_member.lower()]
                if len(hit) != 1:
                    print(f"FATAL: expected exactly one member named {a.zip_member}, found {hit}")
                    return 1
                dest.write_bytes(zf.read(hit[0]))
                info["archive_member"] = hit[0]
        except zipfile.BadZipFile:
            print("FATAL: --zip-member was given but the download is not a zip archive")
            return 1
        finally:
            landing.unlink(missing_ok = True)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    info.update({"bytes": dest.stat().st_size, "sha256": digest})
    print(json.dumps(info, indent = 2))
    if a.out:
        Path(a.out).write_text(json.dumps(info, indent = 2), encoding = "utf-8")
    if a.expect_sha256 and a.expect_sha256.lower() != digest:
        # A file that is not the one asked for is not a file that will do.
        print(f"FATAL: expected sha256 {a.expect_sha256}, got {digest}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
