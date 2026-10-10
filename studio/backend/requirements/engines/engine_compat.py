# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Writes <lock>.compat.json: every constraint the locked packages place on each other, and the
download size of each locked wheel this platform installs.

A shared engine skips a locked package when Studio already has a version all of these accept,
so the engine's own dependency graph decides, not the lock's exact pin. The lock's .in pins are
Studio's choices and are left out. Run after `uv pip compile`:

    python studio/backend/requirements/engines/engine_compat.py vllm-linux-cu130-torch213

Packages a lock takes from its engine's own index (TARGETS) are priced from that index; their
requirements are not read, since such a lock is always an isolated environment.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

HERE = Path(__file__).resolve().parent
# Interpreter, newest manylinux glibc minor and extra index of the locks that differ from the default.
# Kept equal to the profile in core/inference/engine_install.py (this script imports nothing of Studio's).
TARGETS = {
    "vllm-linux-rocm723": {
        "python": (3, 12),
        "glibc": 39,
        "index": "https://wheels.vllm.ai/rocm/0.30.0/rocm723/",
    },
}
DEFAULT_TARGET = {"python": (3, 13), "glibc": 34, "index": None}


def environment(python: tuple[int, int]) -> dict:
    return {
        **default_environment(),
        "implementation_name": "cpython",
        "platform_machine": "x86_64",
        "platform_system": "Linux",
        "python_full_version": "{}.{}.0".format(*python),
        "python_version": "{}.{}".format(*python),
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


def index_files(index: str, name: str, version: str) -> list[dict]:
    """The wheels an extra index lists for one locked version, in PyPI's JSON shape. These indexes
    publish no digests, so every such wheel is offered and the tag ranking picks the one installed."""
    page = urllib.parse.urljoin(index, name + "/")
    with urllib.request.urlopen(page, timeout = 60) as r:
        hrefs = re.findall(r'href="([^"]+)"', r.read().decode("utf-8"))
    files = []
    for href in hrefs:
        url = urllib.parse.urljoin(page, href.split("#", 1)[0])
        if f"-{version}-" not in urllib.parse.unquote(url.rsplit("/", 1)[1]).replace("_", "-"):
            continue
        with urllib.request.urlopen(urllib.request.Request(url, method = "HEAD"), timeout = 60) as r:
            size = int(r.headers["Content-Length"])
        filename = urllib.parse.unquote(url.rsplit("/", 1)[1])
        files.append({"filename": filename, "size": size, "digests": {"sha256": None}})
    return files


def wheel_size(
    files: list[dict],
    allowed: set[str] | None,
    python: tuple[int, int] = (3, 13),
    glibc: int = 34,
) -> int | None:
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import parse_wheel_filename

    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(glibc, 4, -1)]
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
                *cpython_tags(python, platforms = platforms),
                *compatible_tags(python, "cp{}{}".format(*python), platforms),
            ]
        )
    }
    best = None
    for item in files:
        if (allowed is not None and item["digests"]["sha256"] not in allowed) or not item[
            "filename"
        ].endswith(".whl"):
            continue
        ranks = [order[t] for t in parse_wheel_filename(item["filename"])[3] if t in order]
        if ranks and (best is None or min(ranks) < best[0]):
            best = (min(ranks), item["size"])
    return best[1] if best else None


def build(stem: str) -> dict:
    target = TARGETS.get(stem, DEFAULT_TARGET)
    lock = HERE / f"{stem}.txt"
    pins = locked(lock)
    allowed = hashes(lock)
    extra = {}
    if target["index"]:
        with urllib.request.urlopen(target["index"], timeout = 60) as r:
            listed = {
                normalize(n) for n in re.findall(r'href="([^"/]+)/"', r.read().decode("utf-8"))
            }
        extra = {
            name: index_files(target["index"], name, version)
            for name, version in pins.items()
            if name in listed
        }
    releases = {name: release(name, v) for name, v in pins.items() if name not in extra}
    metadata = {
        name: [Requirement(r) for r in data["info"]["requires_dist"] or []]
        for name, data in releases.items()
    }
    sizes = {
        name: wheel_size(data["urls"], allowed.get(name, set()), target["python"], target["glibc"])
        for name, data in releases.items()
    }
    for name, files in extra.items():
        sizes[name] = wheel_size(files, None, target["python"], target["glibc"])
    markers = environment(target["python"])
    extras: dict[str, set[str]] = {name: {""} for name in pins}
    requires: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for name, reqs in metadata.items():
            for req in reqs:
                target = normalize(req.name)
                if target not in pins or not any(
                    req.marker is None or req.marker.evaluate({**markers, "extra": extra})
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
        # Bytes of the wheel installed on the lock's CPython, manylinux x86_64; null when only an sdist fits.
        "sizes": dict(sorted(sizes.items())),
    }


if __name__ == "__main__":
    for stem in sys.argv[1:]:
        out = HERE / f"{stem}.compat.json"
        out.write_text(json.dumps(build(stem), indent = 1) + "\n", encoding = "utf-8")
        print(out)
