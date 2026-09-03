#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A parent portable marker is trusted only when its directory is.

The marker is honoured on existence alone, and honouring one in `<parent>` makes
`<parent>` UNSLOTH_HOME, from which the managed llama.cpp, node and whisper.cpp
directories are resolved and then executed -- at backend startup, for the
capability probe, and again on every GGUF load. So on a shared box, a Studio root
the operator protects at `<parent>/studio` must not inherit trust from a
`<parent>` the operator does not: a lower-privileged user who can create files
there could otherwise drop `.unsloth-portable-root` beside a `llama.cpp/` of
their own and have the next model load run it with the operator's privileges.

Checking the marker's CONTENTS cannot answer this, since whoever writes the file
writes the contents. Ownership and the parent's write bits can, and they are what
install.sh already leaves behind on a normal install under the user's own home.

The point of [2] is that this is a hardening, not a removal: the legitimate
nested layout install.sh produces must still inherit.

Subprocess per case: these resolvers read the environment at import time.
POSIX only -- st_uid is always 0 on Windows and the mode is synthesised, and
install.ps1 refuses portable mode there anyway.
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

sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli

cli._ensure_studio_env_exported()
master = sr.unsloth_home()
cli_master = cli._portable_marker_root()
print("__JSON__" + json.dumps({
    "unsloth_home": None if master is None else str(master),
    "portable": sr.portable_mode(),
    "llama": str((master or sr.studio_root()) / "llama.cpp"),
    "cli_marker_root": None if cli_master is None else str(cli_master),
    "cli_llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
}))
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


def _portable_master(root: Path, *, mode: int) -> Path:
    """The on-disk shape `install.sh --root <root>` leaves behind, at *mode*.

    Plus the llama.cpp an attacker would plant, so the two cases differ only in
    who could have written the parent.
    """
    (root / "studio" / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "llama.cpp" / "build" / "bin").mkdir(parents = True, exist_ok = True)
    (root / MARKER).write_text(f"{root}\n")
    root.chmod(mode)
    return root / "studio"


def main() -> int:
    if os.name == "nt":
        print("SKIP: POSIX-only (st_uid and mode bits are placeholders on Windows)")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()

        print("\n[1] a parent anyone can write is NOT trusted")
        for label, mode in (
            ("world-writable", 0o777),
            ("group-writable", 0o775),
            ("sticky shared", 0o1777),
        ):
            root = _portable_master(tmp / f"shared-{mode:o}", mode = mode)
            r = _run({"UNSLOTH_STUDIO_HOME": str(root)}, home)
            check(f"{label}: names no master root", None, r["unsloth_home"])
            check(f"{label}: stays out of portable mode", False, r["portable"])
            # Fails CLOSED, back to the in-root path this PR's base already used.
            check(f"{label}: llama.cpp stays inside the root", str(root / "llama.cpp"), r["llama"])
            check(f"{label}: CLI inherits nothing", None, r["cli_marker_root"])
            check(f"{label}: CLI exports the in-root path", str(root / "llama.cpp"), r["cli_llama"])

        print("\n[2] the legitimate nested install still inherits")
        for label, mode in (("installer default", 0o755), ("private root", 0o700)):
            master = tmp / f"portable-{mode:o}"
            root = _portable_master(master, mode = mode)
            r = _run({"UNSLOTH_STUDIO_HOME": str(root)}, home)
            check(
                f"{label}: master root found from the marker alone", str(master), r["unsloth_home"]
            )
            check(f"{label}: portable mode on", True, r["portable"])
            check(f"{label}: llama.cpp beside studio/", str(master / "llama.cpp"), r["llama"])
            check(f"{label}: CLI agrees on the master root", str(master), r["cli_marker_root"])
            check(
                f"{label}: CLI exports the master path", str(master / "llama.cpp"), r["cli_llama"]
            )

        print("\n[3] a marker IN the root is unaffected (flat layout)")
        # Nothing above it is consulted, so a shared parent cannot veto it either.
        flat = tmp / "shared-flat"
        (flat / "studio" / "unsloth_studio").mkdir(parents = True)
        (flat / "studio" / MARKER).write_text("")
        (flat / "studio" / "llama.cpp").mkdir()
        flat.chmod(0o777)
        r = _run({"UNSLOTH_STUDIO_HOME": str(flat / "studio")}, home)
        check("flat: the root's own marker still counts", str(flat / "studio"), r["unsloth_home"])
        check("flat: portable mode on", True, r["portable"])

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All parent-marker provenance checks passed.")
    return 0


def test_parent_portable_marker_needs_a_trustworthy_directory():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
