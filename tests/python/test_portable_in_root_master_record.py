#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A nested portable install stays discoverable even when its master root is
group-writable, and a planted parent marker still is not.

Requiring provenance on the PARENT marker (owned by this euid or root, and not
group- or world-writable) closed a local escalation: a lower-privileged user who
could write the parent of a Studio root named `studio` planted the marker plus a
`llama.cpp/build/bin/llama-server` of their own and had the backend execute it.
It also broke a legitimate install. `umask 002` is the default on multi-user
boxes and CI images, so `install.sh --root DIR` creates or reuses a
group-writable DIR, COMPLETES, and prints `source
<root>/studio/unsloth_studio/bin/activate; unsloth studio` -- a path that carries
no environment and had nothing left to resolve through, so it fell all the way
back to ~/.unsloth/studio and wrote the caches, the projects root and studio.db
outside the root the user chose.

The fix records the association INSIDE <root>/studio, where only the operator can
write it: anyone who can put a file there can equally rewrite the venv beside it
and be executed directly, so it needs no permissions argument, while the parent
marker does. Both directions are checked here -- the planted marker is still
refused, and the legitimate group-writable install is now found -- because a
change that only did one of them would look like a fix and be a regression.

Subprocess per case: these resolvers read sys.prefix and the environment at
import time. POSIX only -- st_uid is always 0 on Windows and the mode is
synthesised, and install.ps1 refuses portable mode there anyway.
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
RECORD = ".unsloth-master-root"
OWNED = ".unsloth-studio-owned"

# _PREFIX stands in for an activated venv: the documented launch path sets no
# UNSLOTH_* variable at all, so sys.prefix is the only thing the resolvers have.
#
# TWO subprocesses, not one. _ensure_studio_env_exported() puts UNSLOTH_HOME into
# os.environ, and storage_roots reads that FIRST -- so a probe that runs the CLI
# and then the backend measures the CLI twice and every backend assertion goes
# quietly vacuous. Asked separately, each answers from disk.
BACKEND_PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
master = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "studio_root": str(sr.studio_root()),
    "unsloth_home": None if master is None else str(master),
    "portable": sr.portable_mode(),
    "cache_root": str(sr.cache_root()),
    "llama": str((master or sr.studio_root()) / "llama.cpp"),
}))
"""

CLI_PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
cli_master = cli._portable_master_root()
cli._ensure_studio_env_exported()
print("__JSON__" + json.dumps({
    "cli_studio_home": str(cli.STUDIO_HOME),
    "cli_master": None if cli_master is None else str(cli_master),
    "cli_llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "cli_exported_home": os.environ.get("UNSLOTH_HOME"),
}))
"""


def _probe(probe: str, prefix: Path, home: Path, env_extra: dict, cwd: Path | None) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
        "_REPO": str(REPO),
        "_PREFIX": str(prefix),
    }
    env.update({k: v for k, v in env_extra.items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", probe], env = env, cwd = None if cwd is None else str(cwd),
        capture_output = True, text = True, timeout = 300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


def _run(prefix: Path, home: Path, env_extra: dict | None = None, cwd: Path | None = None) -> dict:
    extra = env_extra or {}
    merged = _probe(BACKEND_PROBE, prefix, home, extra, cwd)
    merged.update(_probe(CLI_PROBE, prefix, home, extra, cwd))
    return merged


FAILS: list[str] = []


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def _nested(master: Path, *, mode: int, record: bool) -> Path:
    """What `install.sh --root <master>` leaves behind, nested layout.

    *record* off is an install made by an earlier revision of this change, which
    has only the parent marker. The llama.cpp is the directory an attacker would
    aim at, so the trusted and untrusted cases differ only in who could have
    written the parent.
    """
    (master / "studio" / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (master / "studio" / "unsloth_studio" / OWNED).write_text("")
    (master / "share").mkdir(exist_ok = True)
    (master / "bin").mkdir(exist_ok = True)
    (master / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME={master}\n")
    (master / "llama.cpp" / "build" / "bin").mkdir(parents = True, exist_ok = True)
    (master / MARKER).write_text(f"{master}\n")
    if record:
        (master / "studio" / RECORD).write_text(f"{master}\n")
    master.chmod(mode)
    return master / "studio"


def main() -> int:
    if os.name == "nt":
        print("SKIP: POSIX-only (st_uid and mode bits are placeholders on Windows)")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td).resolve()
        home = tmp / "home"
        home.mkdir(mode = 0o700)
        legacy = home / ".unsloth" / "studio"

        print("\n[1] the reported failure: a group-writable root, no record")
        # install.sh completed here. Everything the summary told the user to run
        # resolves somewhere else entirely.
        broken = _nested(tmp / "umask002-old", mode = 0o775, record = False)
        r = _run(broken / "unsloth_studio", home)
        check("old build: the venv path cannot find its own root", str(legacy), r["studio_root"])
        check("old build: no master root", None, r["unsloth_home"])
        check("old build: not portable", False, r["portable"])
        check("old build: caches land under HOME", str(legacy / "cache"), r["cache_root"])
        check("old build: the CLI agrees it is lost", str(legacy), r["cli_studio_home"])

        print("\n[2] the same root, with the in-root record install.sh now writes")
        fixed = _nested(tmp / "umask002-new", mode = 0o775, record = True)
        r = _run(fixed / "unsloth_studio", home)
        check("group-writable: the Studio root is found", str(fixed), r["studio_root"])
        check("group-writable: master root found", str(fixed.parent), r["unsloth_home"])
        check("group-writable: portable mode on", True, r["portable"])
        check("group-writable: caches stay in the root", str(fixed / "cache"), r["cache_root"])
        check("group-writable: llama.cpp beside studio/",
              str(fixed.parent / "llama.cpp"), r["llama"])
        # The CLI exports UNSLOTH_HOME into the backend, where it outranks the
        # backend's own lookup, so a split here would point them at two installs.
        check("group-writable: CLI agrees on the Studio root", str(fixed), r["cli_studio_home"])
        check("group-writable: CLI agrees on the master root", str(fixed.parent), r["cli_master"])
        check("group-writable: CLI exports it", str(fixed.parent), r["cli_exported_home"])
        check("group-writable: CLI exports llama.cpp",
              str(fixed.parent / "llama.cpp"), r["cli_llama"])

        print("\n[3] the escalation stays closed: a PLANTED marker, no record")
        # The victim is a normal install at <parent>/studio that never had a
        # master root, so it has no record -- which is exactly why the record
        # cannot be forged into one. The attacker owns only <parent>.
        for label, mode in (("group-writable", 0o775), ("sticky shared", 0o1777), ("world", 0o777)):
            victim = tmp / f"planted-{mode:o}"
            (victim / "studio" / "unsloth_studio" / "bin").mkdir(parents = True)
            (victim / "studio" / "unsloth_studio" / OWNED).write_text("")
            (victim / "studio" / "share").mkdir()
            (victim / "studio" / "share" / "studio.conf").write_text("")
            (victim / "llama.cpp" / "build" / "bin").mkdir(parents = True)
            (victim / MARKER).write_text(f"{victim}\n")
            # The record read for <parent>/studio is the one INSIDE it. One in
            # the parent is not a spelling any resolver looks at.
            (victim / RECORD).write_text(f"{victim}\n")
            victim.chmod(mode)
            r = _run(victim / "studio" / "unsloth_studio", home)
            check(f"{label}: names no master root", None, r["unsloth_home"])
            check(f"{label}: stays out of portable mode", False, r["portable"])
            check(f"{label}: llama.cpp stays inside the root",
                  str(victim / "studio" / "llama.cpp"), r["llama"])
            check(f"{label}: CLI inherits nothing", None, r["cli_master"])
            check(f"{label}: CLI exports the in-root path",
                  str(victim / "studio" / "llama.cpp"), r["cli_llama"])

        print("\n[4] installs made by earlier builds keep working when the root is sane")
        for label, mode in (("installer default", 0o755), ("private root", 0o700)):
            legit = _nested(tmp / f"old-{mode:o}", mode = mode, record = False)
            r = _run(legit / "unsloth_studio", home)
            check(f"{label}: master root still found from the marker alone",
                  str(legit.parent), r["unsloth_home"])
            check(f"{label}: portable mode on", True, r["portable"])
            check(f"{label}: CLI agrees", str(legit.parent), r["cli_master"])

        print("\n[5] a record that cannot be believed declines instead of winning")
        # Each drops back to the parent marker, which is trustworthy here, so a
        # bad record can never be worse than having none.
        for label, body in (
            ("empty", ""),
            ("blank", "   \n"),
            ("missing directory", "/nonexistent-unsloth-master-root\n"),
        ):
            root = _nested(tmp / f"bad-{label.replace(' ', '-')}", mode = 0o700, record = False)
            (root / RECORD).write_text(body)
            r = _run(root / "unsloth_studio", home)
            check(f"bad record ({label}): falls through to the marker",
                  str(root.parent), r["unsloth_home"])
        # Relative, probed from a working directory where that name really exists:
        # without the absolute-path rule this resolves against the CWD and hands
        # UNSLOTH_HOME to whatever the process happened to be started in.
        elsewhere = tmp / "cwd-relative"
        (elsewhere / "escape").mkdir(parents = True)
        root = _nested(tmp / "bad-relative", mode = 0o700, record = False)
        (root / RECORD).write_text("escape\n")
        r = _run(root / "unsloth_studio", home, cwd = elsewhere)
        check("bad record (relative): falls through to the marker",
              str(root.parent), r["unsloth_home"])
        # A directory at the name reads as absent, the way the marker does.
        root = _nested(tmp / "bad-directory", mode = 0o700, record = False)
        (root / RECORD).mkdir()
        r = _run(root / "unsloth_studio", home)
        check("bad record (directory): falls through to the marker",
              str(root.parent), r["unsloth_home"])

        print("\n[6] the record outranks a stale flat marker left in the Studio root")
        # A `--root R` install over a directory whose R/studio used to be a FLAT
        # portable root: _clear_stale_portable_marker only runs in normal mode,
        # so that marker survives, and reading it first made R/studio its own
        # master root -- llama.cpp one level too deep, in the wrong tree.
        stale = _nested(tmp / "stale-flat", mode = 0o755, record = True)
        (stale / MARKER).write_text(f"{stale}\n")
        r = _run(stale / "unsloth_studio", home)
        check("stale flat marker: the record still names the real master root",
              str(stale.parent), r["unsloth_home"])
        check("stale flat marker: llama.cpp beside studio/",
              str(stale.parent / "llama.cpp"), r["llama"])

        print("\n[7] UNSLOTH_HOME from the environment still outranks everything")
        override = tmp / "explicit"
        override.mkdir()
        r = _run(fixed / "unsloth_studio", home, {"UNSLOTH_HOME": str(override)})
        check("explicit UNSLOTH_HOME wins", str(override), r["unsloth_home"])

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All in-root master record checks passed.")
    return 0


def test_in_root_master_record_survives_a_group_writable_root():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
