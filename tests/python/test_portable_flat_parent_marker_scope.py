#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A FLAT install's portable marker is its own, not a studio/ child's.

`install.sh --portable` writes the marker at the master root under both layouts,
and both resolvers infer "the marker one level up is mine" from the basename
`studio` alone. That is right for the nested layout, where `<root>/studio` IS the
install the marker was written for. It is wrong for the flat layout, which has no
studio/ child at all: there `<root>` holds the venv directly, and a
`<root>/studio` reached through the supported `UNSLOTH_STUDIO_HOME` override is a
SEPARATE installation that inherited the flat install's `UNSLOTH_HOME` -- and
with it that install's node, llama.cpp, whisper.cpp and cache policy.

So the parent marker is now believed only when the parent does NOT own a venv of
its own, decided with the same four ownership sentinels install.sh applies. Asked
that way rather than through `_is_flat_portable_root`, which excludes an
already-nested root FIRST and so answers False on exactly this fixture, for the
wrong reason.

[3] is the anti-regression half: a genuine nested install must still find its
master root, or every portable install loses containment instead.

Subprocess per case: both resolvers read the environment at import time.
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
MARKER = ".unsloth-portable-root"
OWNED = ".unsloth-studio-owned"

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
home = sr.unsloth_home()
out = {
    "backend_studio": str(sr.studio_root()),
    "backend_home": None if home is None else str(home),
    "backend_portable": sr.portable_mode(),
}
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
master = cli._portable_master_root()
out["cli_studio"] = str(cli.STUDIO_HOME)
out["cli_home"] = None if master is None else str(master)
print("__JSON__" + json.dumps(out))
"""

FAILS: list[str] = []


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


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def master_root(label: str, studio: Path, expected: Path | None, home: Path) -> None:
    """Both resolvers must name the same master root for *studio*, or neither."""
    got = _run({"UNSLOTH_STUDIO_HOME": str(studio)}, home)
    want = None if expected is None else str(expected)
    check(f"{label}: backend unsloth_home()", want, got["backend_home"])
    check(f"{label}: CLI master root", want, got["cli_home"])
    # The master root is what portable_mode() falls back to, so an install that
    # stops inheriting also stops claiming the other install's cache policy.
    check(f"{label}: backend portable_mode()", expected is not None, got["backend_portable"])


def sq(value: str) -> str:
    return value.replace("'", "'\\''")


def flat_install(root: Path) -> None:
    """A complete flat portable install: marker at the root, venv beside it."""
    venv = root / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    (venv / OWNED).write_text("")
    (root / "share").mkdir(exist_ok = True)
    (root / "share" / "studio.conf").write_text(
        f"UNSLOTH_EXE='{sq(str(venv))}/bin/unsloth'\nexport UNSLOTH_PORTABLE=1\n"
    )
    (root / MARKER).write_text("")


def normal_install(studio: Path) -> None:
    """A plain non-portable install: its own venv and its own share/studio.conf."""
    venv = studio / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    (venv / OWNED).write_text("")
    (studio / "share").mkdir(parents = True, exist_ok = True)
    (studio / "share" / "studio.conf").write_text(
        f"UNSLOTH_EXE='{sq(str(venv))}/bin/unsloth'\n"
    )


def main() -> int:
    FAILS.clear()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()

        print("\n[1] a flat install's marker does not reach a separate studio/ child")
        flat = tmp / "flat-with-child"
        flat.mkdir()
        flat_install(flat)
        normal_install(flat / "studio")
        master_root("normal install under a flat root", flat / "studio", None, home)

        # Same shape before the child has a venv: the user has pointed
        # UNSLOTH_STUDIO_HOME at a directory the flat install does not own, and
        # the very first launch must not adopt the parent either.
        flat2 = tmp / "flat-with-empty-child"
        flat2.mkdir()
        flat_install(flat2)
        (flat2 / "studio").mkdir()
        master_root("empty studio/ child under a flat root", flat2 / "studio", None, home)

        # Each outside sentinel on its own, so the rule cannot be satisfied by
        # the in-venv marker alone.
        for sentinel in ("conf", "shim", "link"):
            root = tmp / f"flat-{sentinel}"
            venv = root / "unsloth_studio"
            (venv / "bin").mkdir(parents = True)
            (root / MARKER).write_text("")
            if sentinel == "conf":
                (root / "share").mkdir()
                (root / "share" / "studio.conf").write_text(
                    f"UNSLOTH_EXE='{sq(str(venv))}/bin/unsloth'\n"
                )
            elif sentinel == "shim":
                (root / "bin").mkdir()
                (root / "bin" / "unsloth").write_text(
                    f"#!/bin/sh\nexec '{sq(str(venv))}/bin/unsloth' \"$@\"\n"
                )
            else:
                (root / "bin").mkdir()
                (venv / "bin" / "unsloth").write_text("")
                (root / "bin" / "unsloth").symlink_to(venv / "bin" / "unsloth")
            normal_install(root / "studio")
            master_root(f"flat root proved by the {sentinel} sentinel", root / "studio",
                        None, home)

        print("\n[2] the flat install itself is untouched")
        master_root("the flat root resolves itself", flat, flat, home)

        print("\n[3] a genuine NESTED install still finds its master root")
        nested = tmp / "nested"
        (nested / "studio" / "unsloth_studio").mkdir(parents = True)
        (nested / MARKER).write_text("")
        master_root("plain nested portable install", nested / "studio", nested, home)

        # The parent holds something called unsloth_studio, but nothing says it
        # is ours. That is not a flat install, so the nested child below it must
        # keep inheriting: the fix must not collapse into "never inherit".
        stray = tmp / "nested-with-stray"
        (stray / "studio" / "unsloth_studio").mkdir(parents = True)
        (stray / "unsloth_studio" / "lib").mkdir(parents = True)
        (stray / "unsloth_studio" / "pyvenv.cfg").write_text("")
        (stray / MARKER).write_text("")
        master_root("nested install under an unowned stray venv", stray / "studio", stray, home)

        # ...and a parent whose sentinels name some OTHER venv is equally not a
        # flat install of ours.
        elsewhere = tmp / "elsewhere" / "unsloth_studio"
        elsewhere.mkdir(parents = True)
        foreign = tmp / "nested-with-foreign-conf"
        (foreign / "studio" / "unsloth_studio").mkdir(parents = True)
        (foreign / "unsloth_studio").mkdir()
        (foreign / "share").mkdir()
        (foreign / "share" / "studio.conf").write_text(
            f"UNSLOTH_EXE='{sq(str(elsewhere))}/bin/unsloth'\n"
        )
        (foreign / MARKER).write_text("")
        master_root("nested install under sentinels naming another venv",
                    foreign / "studio", foreign, home)

        print("\n[4] an unrelated sibling of a marked root still inherits nothing")
        sibling = tmp / "sibling"
        (sibling / "other" / "unsloth_studio").mkdir(parents = True)
        (sibling / MARKER).write_text("")
        master_root("a child that is not named studio", sibling / "other", None, home)

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("Parent-marker scope checks passed.")
    return 0


def test_a_flat_installs_marker_does_not_claim_a_studio_child():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
