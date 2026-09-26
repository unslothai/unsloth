# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Writes <lock>.compat.json: every constraint the locked packages place on each other.

A shared engine skips a locked package when Studio already has a version all of these accept,
so the engine's own dependency graph decides, not the lock's exact pin. The lock's .in pins are
Studio's choices and are left out. Run after `uv pip compile`:

    python studio/backend/requirements/engines/engine_compat.py vllm
"""

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


def requires_dist(name: str, version: str) -> list[str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout = 60) as r:
        return json.load(r)["info"]["requires_dist"] or []


def build(engine: str) -> dict:
    lock = HERE / f"{engine}-linux-cu130.txt"
    pins = locked(lock)
    metadata = {name: [Requirement(r) for r in requires_dist(name, v)] for name, v in pins.items()}
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
    }


if __name__ == "__main__":
    for engine in sys.argv[1:]:
        out = HERE / f"{engine}-linux-cu130.compat.json"
        out.write_text(json.dumps(build(engine), indent = 1) + "\n", encoding = "utf-8")
        print(out)
