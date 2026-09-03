#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A portable marker reaches down into `<master>/studio` and nowhere else.

install.sh writes `.unsloth-portable-root` at the master root, and the runtime
reads it from the Studio root (flat layout) or one level up (nested layout,
where the Studio root is `<master>/studio`). If that second lookup is not
restricted to the `studio` child, an unrelated installation that happens to sit
at another direct child of a portable root, `UNSLOTH_STUDIO_HOME=<master>/other`
beside `<master>/studio`, is classified as part of the portable install: it
takes `<master>` as UNSLOTH_HOME and so runs the FIRST install's managed node,
llama.cpp and whisper.cpp, and turns portable mode on for a plain install that
never asked for it. install.sh's own _clear_stale_portable_marker already
matches only `*/studio` for the same reason.

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
MARKER = ".unsloth-portable-root"

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
from utils.node_runtime import managed_node_dir
from core.inference.stt_ggml_sidecar import _managed_whisper_cpp_dir

# studio_root() is called constantly, so a warning for a supported layout is not
# one line, it is a flooded log.
warnings = []
sr.logger.warning = lambda msg, *a, **k: warnings.append(msg % a if a else msg)

prefix = os.environ.get("_PREFIX")
if prefix:
    sys.prefix = sys.exec_prefix = prefix

master = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "studio_root": str(sr.studio_root()),
    "unsloth_home": None if master is None else str(master),
    "portable": sr.portable_mode(),
    "node": str(managed_node_dir()),
    "whisper": str(_managed_whisper_cpp_dir()),
    "llama": str((master or sr.studio_root()) / "llama.cpp"),
    "warnings": warnings,
}))
"""


def _run(env_extra: dict, home: Path) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
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


def _portable_master(root: Path) -> None:
    """The on-disk shape `install.sh --root <root>` leaves behind."""
    (root / "studio" / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "bin").mkdir(parents = True, exist_ok = True)
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME='{root}'\n")
    (root / MARKER).write_text(f"{root}\n")


def _plain_install(root: Path) -> None:
    """A separate installation, complete with its own venv and sentinel."""
    (root / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "share" / "studio.conf").write_text("")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()

        master = tmp / "portable"
        _portable_master(master)
        sibling = master / "other"
        _plain_install(sibling)

        print("\n[1] the studio child still inherits the marker")
        r = _run({"UNSLOTH_STUDIO_HOME": str(master / "studio")}, home)
        check("nested: master root found from the marker alone", str(master), r["unsloth_home"])
        check("nested: portable mode on", True, r["portable"])
        check("nested: node beside studio/", str(master / "node"), r["node"])
        check("nested: whisper beside studio/", str(master / "whisper.cpp"), r["whisper"])
        check("nested: llama.cpp beside studio/", str(master / "llama.cpp"), r["llama"])

        print("\n[2] an unrelated sibling install is NOT adopted")
        r = _run({"UNSLOTH_STUDIO_HOME": str(sibling)}, home)
        check("sibling: names no master root", None, r["unsloth_home"])
        check("sibling: stays out of portable mode", False, r["portable"])
        check("sibling: keeps its own node", str(sibling / "node"), r["node"])
        check("sibling: keeps its own whisper", str(sibling / "whisper.cpp"), r["whisper"])
        check("sibling: keeps its own llama.cpp", str(sibling / "llama.cpp"), r["llama"])
        check("sibling: still resolves to itself", str(sibling), r["studio_root"])

        print("\n[3] an activated venv in the sibling is not adopted either")
        # The reason the marker exists: a venv binary reached past the shim
        # carries none of the installer's environment. It must still land on the
        # right install, and the sentinel that recognises a nested portable
        # install is the same parent marker.
        r = _run({"_PREFIX": str(master / "studio" / "unsloth_studio")}, home)
        check(
            "venv in studio/: infers the nested Studio root",
            str(master / "studio"),
            r["studio_root"],
        )
        check("venv in studio/: finds the master root", str(master), r["unsloth_home"])
        r = _run({"_PREFIX": str(sibling / "unsloth_studio")}, home)
        check("venv in the sibling: names no master root", None, r["unsloth_home"])
        check("venv in the sibling: stays out of portable mode", False, r["portable"])

        print("\n[4] the flat layout is untouched")
        # Master root IS the Studio root, marker inside it, and its name is
        # whatever the user called the volume. Restricting the PARENT lookup
        # must not reach this one.
        flat = tmp / "vol"
        (flat / "unsloth_studio" / "bin").mkdir(parents = True)
        (flat / MARKER).write_text(f"{flat}\n")
        r = _run({"UNSLOTH_STUDIO_HOME": str(flat)}, home)
        check("flat: the root is its own master", str(flat), r["unsloth_home"])
        check("flat: portable mode on", True, r["portable"])
        check("flat: node under the root", str(flat / "node"), r["node"])
        check("flat: no not-self-contained warning", [], r["warnings"])

        print("\n[5] explicit UNSLOTH_HOME still outranks the marker")
        # The precedence order is env first: an installer-set UNSLOTH_HOME is not
        # a guess, and a user pointing two roots at one tree gets what they asked
        # for.
        r = _run(
            {"UNSLOTH_STUDIO_HOME": str(sibling), "UNSLOTH_HOME": str(master)},
            home,
        )
        check("explicit UNSLOTH_HOME wins for the sibling", str(master), r["unsloth_home"])

        print("\n[6] a marker two levels up reaches nothing")
        deep = master / "studio" / "nested"
        deep.mkdir(parents = True)
        r = _run({"UNSLOTH_STUDIO_HOME": str(deep)}, home)
        check("grandchild: names no master root", None, r["unsloth_home"])

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL PORTABLE-MARKER SCOPE CHECKS PASSED")
    return 0


def test_portable_marker_reaches_only_the_studio_child():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
