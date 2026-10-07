# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Writes <lock>.compat.json: every constraint the locked packages place on each other, and the
download size of each locked wheel this platform installs.

A shared engine skips a locked package when Studio already has a version all of these accept,
so the engine's own dependency graph decides, not the lock's exact pin. The lock's .in pins are
Studio's choices and are left out. Run after `uv pip compile`:

    python studio/backend/requirements/engines/engine_compat.py vllm-linux-cu130-torch213
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

HERE = Path(__file__).resolve().parent
ENVIRONMENT = {
    **default_environment(),
    "implementation_name": "cpython",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "python_full_version": "3.13.0",
    "python_version": "3.13",
    "sys_platform": "linux",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked(lock: Path) -> dict[str, str]:
    pins = {}
    for line in lock.read_text(encoding = "utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s;\\]+)", line)
        if match:
            pins[normalize(match.group(1))] = match.group(2)
    return pins


def release(name: str, version: str) -> dict:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout = 60) as r:
        return json.load(r)


def hashes(lock: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    name = None
    for line in lock.read_text(encoding = "utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==", line)
        if match:
            name = normalize(match.group(1))
        elif name and line.strip().startswith("--hash=sha256:"):
            found.setdefault(name, set()).add(line.strip().split(":", 1)[1].rstrip(" \\"))
    return found


def wheel_size(files: list[dict], allowed: set[str]) -> int | None:
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import parse_wheel_filename

    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(34, 4, -1)]
    platforms += [
        "manylinux2014_x86_64",
        "manylinux2010_x86_64",
        "manylinux1_x86_64",
        "linux_x86_64",
    ]
    order = {
        tag: rank
        for rank, tag in enumerate(
            [
                *cpython_tags((3, 13), platforms = platforms),
                *compatible_tags((3, 13), "cp313", platforms),
            ]
        )
    }
    best = None
    for item in files:
        if item["digests"]["sha256"] not in allowed or not item["filename"].endswith(".whl"):
            continue
        ranks = [order[t] for t in parse_wheel_filename(item["filename"])[3] if t in order]
        if ranks and (best is None or min(ranks) < best[0]):
            best = (min(ranks), item["size"])
    return best[1] if best else None


def build(stem: str) -> dict:
    lock = HERE / f"{stem}.txt"
    pins = locked(lock)
    allowed = hashes(lock)
    releases = {name: release(name, v) for name, v in pins.items()}
    metadata = {
        name: [Requirement(r) for r in data["info"]["requires_dist"] or []]
        for name, data in releases.items()
    }
    sizes = {
        name: wheel_size(data["urls"], allowed.get(name, set())) for name, data in releases.items()
    }
    extras: dict[str, set[str]] = {name: {""} for name in pins}
    requires: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for name, reqs in metadata.items():
            for req in reqs:
                target = normalize(req.name)
                if target not in pins or not any(
                    req.marker is None or req.marker.evaluate({**ENVIRONMENT, "extra": extra})
                    for extra in extras[name]
                ):
                    continue
                if str(req.specifier):
                    requires.setdefault(target, set()).add(str(req.specifier))
                new = set(req.extras) - extras[target]
                if new:
                    extras[target] |= new
                    changed = True
    return {
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "requires": {name: sorted(specs) for name, specs in sorted(requires.items())},
        # Wheel bytes on CPython 3.13 manylinux x86_64; null when only an sdist fits.
        "sizes": dict(sorted(sizes.items())),
    }


if __name__ == "__main__":
    for stem in sys.argv[1:]:
        out = HERE / f"{stem}.compat.json"
        out.write_text(json.dumps(build(stem), indent = 1) + "\n", encoding = "utf-8")
        print(out)
