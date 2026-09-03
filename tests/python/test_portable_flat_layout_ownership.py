#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The flat layout is decided by ownership, the way install.sh decides it.

`install.sh --portable --root DIR` builds NESTED at `DIR/studio` unless an
ownership sentinel proves `DIR` already holds a flat Unsloth install: the in-venv
`.unsloth-studio-owned` marker, `share/studio.conf`, or the `bin/unsloth` shim,
with an already-nested `DIR/studio/unsloth_studio` excluded FIRST because the
last two sit at `DIR` in BOTH layouts.

Deciding on the bare existence of `DIR/unsloth_studio` instead makes the
resolvers disagree with the installer by construction, and `UNSLOTH_HOME=DIR`
with no `UNSLOTH_STUDIO_HOME` is enough to hit it: an empty leftover directory of
that name, or somebody's unrelated dev venv, relocates the Studio root from
`DIR/studio` to `DIR`. A stray `DIR/unsloth_studio` beside a REAL nested install
moves that install's root out from under it.

Both resolvers are checked in one probe. The CLI exports UNSLOTH_HOME into the
backend's environment, so the two silently disagreeing is worse than either being
wrong alone.

[3] is the anti-regression half: the ownership requirement must not collapse into
"never flat", and the precedence order and legacy default must be untouched.

Subprocess per case: these resolvers read the environment at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
OWNED = ".unsloth-studio-owned"

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
out = {"backend": str(sr.studio_root())}
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
out["cli"] = str(cli.STUDIO_HOME)
print("__JSON__" + json.dumps(out))
"""


def _run(env_extra: dict, home: Path) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
        "_REPO": str(REPO),
    }
    env.update({k: v for k, v in env_extra.items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], env = env, capture_output = True, text = True, timeout = 300
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


FAILS: list[str] = []


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def both(label: str, expected: Path, env_extra: dict, home: Path) -> None:
    """Both resolvers must answer *expected*, and so must agree with each other."""
    r = _run(env_extra, home)
    check(f"{label}: backend", str(expected), r["backend"])
    check(f"{label}: CLI", str(expected), r["cli"])


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()

        def root(name: str) -> Path:
            return tmp / name

        print("\n[1] a directory that cannot prove it is ours stays NESTED")
        leftover = root("leftover")
        (leftover / "unsloth_studio").mkdir(parents = True)
        both("empty leftover", leftover / "studio", {"UNSLOTH_HOME": str(leftover)}, home)

        dev = root("dev-venv")
        (dev / "unsloth_studio" / "bin").mkdir(parents = True)
        (dev / "unsloth_studio" / "pyvenv.cfg").write_text("")
        both("unrelated dev venv", dev / "studio", {"UNSLOTH_HOME": str(dev)}, home)

        print("\n[2] a stray venv name cannot relocate a REAL nested install")
        # share/studio.conf and bin/unsloth sit at <root> in both layouts, so
        # without the nested exclusion first these two sentinels would say flat.
        stray = root("nested-with-stray")
        (stray / "studio" / "unsloth_studio").mkdir(parents = True)
        (stray / "unsloth_studio").mkdir()
        (stray / "share").mkdir()
        (stray / "share" / "studio.conf").write_text("")
        (stray / "bin").mkdir()
        (stray / "bin" / "unsloth").write_text("")
        both("nested + stray venv", stray / "studio", {"UNSLOTH_HOME": str(stray)}, home)

        print("\n[3] a genuine flat install still resolves FLAT")
        # One case per sentinel: the requirement must not collapse into never-flat.
        for label, build in (
            ("owned marker", lambda p: (p / "unsloth_studio" / OWNED).write_text("")),
            ("share/studio.conf", lambda p: (p / "share" / "studio.conf").write_text("")),
            ("bin/unsloth shim", lambda p: (p / "bin" / "unsloth").write_text("")),
        ):
            flat = root("flat-" + label.split("/")[0].replace(" ", "-"))
            (flat / "unsloth_studio").mkdir(parents = True)
            (flat / "share").mkdir()
            (flat / "bin").mkdir()
            build(flat)
            both(f"flat via {label}", flat, {"UNSLOTH_HOME": str(flat)}, home)

        print("\n[4] the nested layout install.sh builds by default")
        nested = root("nested")
        (nested / "studio" / "unsloth_studio").mkdir(parents = True)
        (nested / "share").mkdir()
        (nested / "share" / "studio.conf").write_text("")
        both("plain nested", nested / "studio", {"UNSLOTH_HOME": str(nested)}, home)

        no_venv = root("fresh")
        no_venv.mkdir()
        both("root with no venv at all", no_venv / "studio", {"UNSLOTH_HOME": str(no_venv)}, home)

        print("\n[5] precedence and the legacy default are unchanged")
        # UNSLOTH_STUDIO_HOME > STUDIO_HOME > <UNSLOTH_HOME>/studio.
        explicit = root("explicit")
        explicit.mkdir()
        both(
            "UNSLOTH_STUDIO_HOME outranks UNSLOTH_HOME",
            explicit,
            {"UNSLOTH_HOME": str(nested), "UNSLOTH_STUDIO_HOME": str(explicit)},
            home,
        )
        both(
            "UNSLOTH_STUDIO_HOME outranks STUDIO_HOME",
            explicit,
            {"UNSLOTH_STUDIO_HOME": str(explicit), "STUDIO_HOME": str(leftover)},
            home,
        )
        both(
            "STUDIO_HOME outranks UNSLOTH_HOME",
            explicit,
            {"UNSLOTH_HOME": str(nested), "STUDIO_HOME": str(explicit)},
            home,
        )
        both("legacy default with no variables", home / ".unsloth" / "studio", {}, home)

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All flat-layout ownership checks passed.")
    return 0


def test_flat_portable_layout_requires_an_ownership_sentinel():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
