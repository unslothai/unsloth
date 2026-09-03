#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The CLI reaches up for `.unsloth-portable-root` only from `<master>/studio`.

storage_roots.py stops a sibling install from adopting a neighbouring portable
root, but the CLI is what puts UNSLOTH_HOME into the backend's environment in the
first place: _ensure_studio_env_exported exports it, and storage_roots'
_env_unsloth_home takes an explicit UNSLOTH_HOME ahead of any on-disk lookup. So
with only the backend guarded, `unsloth studio start` from an install at
`<master>/other` still handed the whole process tree the FIRST install's master
root, its node, llama.cpp and whisper.cpp, and its uv / npm / pip caches.

install.sh produces two portable shapes and no others: FLAT, where the Studio
root IS the master root and the marker sits inside it, and NESTED, where the
Studio root is literally `<master>/studio`. Its own _clear_stale_portable_marker
matches the same `*/studio` spelling before it will touch a parent marker.

Each case runs in a subprocess: STUDIO_HOME is resolved at import time.
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

# Reports what the CLI resolves AND what it exports, because the export is the
# half that leaks into the backend. Runtime paths are read back through
# storage_roots in the same process, after the export, which is exactly the order
# a real `unsloth studio start` produces them in.
CLI_PROBE = r"""
import json, os, sys
prefix = os.environ.get("_PREFIX")
if prefix:
    sys.prefix = sys.exec_prefix = prefix
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli

master = cli._portable_master_root()
cli._ensure_studio_env_exported()

sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
from utils.node_runtime import managed_node_dir
from core.inference.stt_ggml_sidecar import _managed_whisper_cpp_dir

backend_master = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "studio_home": str(cli.STUDIO_HOME),
    "custom": cli._STUDIO_HOME_IS_CUSTOM,
    "master": str(master) if master else None,
    "exported_home": os.environ.get("UNSLOTH_HOME"),
    "studio_home_env": os.environ.get("UNSLOTH_STUDIO_HOME"),
    "llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "uv_cache": os.environ.get("UV_CACHE_DIR"),
    "npm_cache": os.environ.get("NPM_CONFIG_CACHE"),
    "portable_flag": os.environ.get("UNSLOTH_PORTABLE"),
    "backend_master": str(backend_master) if backend_master else None,
    "backend_portable": sr.portable_mode(),
    "backend_studio_root": str(sr.studio_root()),
    "node": str(managed_node_dir()),
    "whisper": str(_managed_whisper_cpp_dir()),
}))
"""


# The case-fold has to be provoked: the branch only runs on Windows and macOS, and
# the spelling it exists for names a DIFFERENT directory on the Linux CI box. So
# import normally, then re-point the module at the fixture and ask again.
FOLD_PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_REPO"])
from pathlib import Path
from unsloth_cli.commands import studio as cli

cli.STUDIO_HOME = Path(os.environ["_STUDIO_HOME"])
sys.platform = os.environ["_PLATFORM"]
master = cli._portable_marker_root()
print("__JSON__" + json.dumps({
    "master": str(master) if master else None,
    "managed": cli._looks_like_installer_managed_studio_home(cli.STUDIO_HOME),
}))
"""


def _run(
    env_extra: dict,
    home: Path,
    probe: str = CLI_PROBE,
) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_REPO": str(REPO),
        "_BACKEND": str(BACKEND),
    }
    env.update({k: v for k, v in env_extra.items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", probe], env = env, capture_output = True, text = True, timeout = 600
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


def _portable_master(root: Path) -> Path:
    """The on-disk shape `install.sh --root <root>` leaves behind; returns the venv."""
    (root / "studio" / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "bin").mkdir(parents = True, exist_ok = True)
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "bin" / "unsloth").write_text("#!/bin/sh\n")
    (root / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME='{root}'\n")
    (root / MARKER).write_text(f"{root}\n")
    return root / "studio" / "unsloth_studio"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()

        master = tmp / "portable"
        nested_prefix = _portable_master(master)

        # A second, unrelated install that happens to sit beside studio/. No
        # sentinel of its own, so nothing but the neighbour's marker can make the
        # CLI call it installer-managed.
        sibling = master / "other"
        (sibling / "unsloth_studio" / "bin").mkdir(parents = True)

        print("\n[1] UNSLOTH_STUDIO_HOME at a sibling of studio/ exports nothing")
        r = _run({"UNSLOTH_STUDIO_HOME": str(sibling)}, home)
        check("sibling: CLI names no master root", None, r["master"])
        check("sibling: exports no UNSLOTH_HOME", None, r["exported_home"])
        check("sibling: exports no uv cache", None, r["uv_cache"])
        check("sibling: exports no npm cache", None, r["npm_cache"])
        check("sibling: exports no UNSLOTH_PORTABLE", None, r["portable_flag"])
        check("sibling: llama.cpp stays under the sibling", str(sibling / "llama.cpp"), r["llama"])
        check("sibling: node stays under the sibling", str(sibling / "node"), r["node"])
        check(
            "sibling: whisper stays under the sibling", str(sibling / "whisper.cpp"), r["whisper"]
        )
        # The whole point of the export: with UNSLOTH_HOME in the environment the
        # backend stops consulting the marker at all and just believes the CLI.
        check("sibling: backend agrees there is no master root", None, r["backend_master"])
        check("sibling: backend stays out of portable mode", False, r["backend_portable"])
        check("sibling: backend keeps the sibling root", str(sibling), r["backend_studio_root"])

        print("\n[2] an activated venv inside the sibling is not adopted")
        # _looks_like_installer_managed_studio_home is the other reader of the
        # parent marker, and it is what a `source .../activate` reaches.
        r = _run({"_PREFIX": str(sibling / "unsloth_studio")}, home)
        check(
            "sibling venv: falls back to the legacy root",
            str(home / ".unsloth" / "studio"),
            r["studio_home"],
        )
        check("sibling venv: not treated as a custom root", False, r["custom"])
        check("sibling venv: names no master root", None, r["master"])
        check("sibling venv: exports no UNSLOTH_HOME", None, r["exported_home"])

        print("\n[3] the real nested install still resolves through the marker")
        r = _run({"UNSLOTH_STUDIO_HOME": str(master / "studio")}, home)
        check("nested: finds the master root", str(master), r["master"])
        check("nested: exports the master root", str(master), r["exported_home"])
        check("nested: llama.cpp beside studio/", str(master / "llama.cpp"), r["llama"])
        check("nested: uv cache under the master root", str(master / "cache" / "uv"), r["uv_cache"])
        check("nested: node beside studio/", str(master / "node"), r["node"])
        check("nested: whisper beside studio/", str(master / "whisper.cpp"), r["whisper"])

        print("\n[4] the activated nested venv, the case the marker exists for")
        r = _run({"_PREFIX": str(nested_prefix)}, home)
        check(
            "nested venv: resolves the nested Studio root", str(master / "studio"), r["studio_home"]
        )
        check("nested venv: treated as a custom root", True, r["custom"])
        check("nested venv: finds the master root", str(master), r["master"])
        check("nested venv: exports the master root", str(master), r["exported_home"])

        print("\n[5] a nested venv with NO sentinel but the parent marker")
        # share/studio.conf and bin/unsloth are one level up in this layout, so on
        # a --root install the parent marker is the only sentinel beside the venv.
        # Restricting the lookup must not cost the nested install its discovery.
        bare = tmp / "bare"
        (bare / "studio" / "unsloth_studio" / "bin").mkdir(parents = True)
        (bare / MARKER).write_text(f"{bare}\n")
        r = _run({"_PREFIX": str(bare / "studio" / "unsloth_studio")}, home)
        check("bare nested venv: still resolves", str(bare / "studio"), r["studio_home"])
        check("bare nested venv: finds the master root", str(bare), r["master"])

        print("\n[6] the flat layout is untouched")
        flat = tmp / "vol"
        (flat / "unsloth_studio" / "bin").mkdir(parents = True)
        (flat / MARKER).write_text(f"{flat}\n")
        r = _run({"UNSLOTH_STUDIO_HOME": str(flat)}, home)
        check("flat: the root is its own master", str(flat), r["master"])
        check("flat: exports itself as UNSLOTH_HOME", str(flat), r["exported_home"])
        check("flat: llama.cpp under the root", str(flat / "llama.cpp"), r["llama"])
        check("flat: node under the root", str(flat / "node"), r["node"])

        print("\n[7] `--portable` at its default root keeps working")
        # STUDIO_HOME equals ~/.unsloth/studio here, so it is not custom and the
        # parent marker is the only portable signal there is. Its name IS studio.
        dflt_home = tmp / "defaulthome"
        dflt_prefix = _portable_master(dflt_home / ".unsloth")
        r = _run({"_PREFIX": str(dflt_prefix)}, dflt_home)
        check("default portable: finds the master root", str(dflt_home / ".unsloth"), r["master"])
        check(
            "default portable: exports the master root",
            str(dflt_home / ".unsloth"),
            r["exported_home"],
        )

        print("\n[8] an explicit UNSLOTH_HOME still wins")
        # An installer-set value is not a guess, and a user who deliberately points
        # two roots at one tree gets what they asked for.
        r = _run(
            {"UNSLOTH_STUDIO_HOME": str(sibling), "UNSLOTH_HOME": str(master)},
            home,
        )
        check("explicit UNSLOTH_HOME wins for the sibling", str(master), r["master"])
        check("explicit UNSLOTH_HOME is kept", str(master), r["exported_home"])

        print("\n[9] `Studio` is the same directory where the filesystem says so")
        # The installer writes `studio`, but a user typing UNSLOTH_STUDIO_HOME by
        # hand on macOS or Windows can spell it `Studio` and open the very same
        # directory; resolve() hands the spelling straight through. Rejecting it
        # would break a real nested install, so the fold must agree with
        # storage_roots._inherits_parent_portable_marker.
        cased = tmp / "cased"
        (cased / "Studio" / "unsloth_studio" / "bin").mkdir(parents = True)
        (cased / MARKER).write_text(f"{cased}\n")
        r = _run({"_STUDIO_HOME": str(cased / "Studio"), "_PLATFORM": "darwin"}, home, FOLD_PROBE)
        check("darwin: Studio inherits the parent marker", str(cased), r["master"])
        check("darwin: Studio reads as installer-managed", True, r["managed"])
        r = _run({"_STUDIO_HOME": str(cased / "Studio"), "_PLATFORM": "linux"}, home, FOLD_PROBE)
        check("linux: Studio is a distinct directory, so no inherit", None, r["master"])
        # Lowercase must not depend on the fold, or the fold becomes the whole rule.
        r = _run({"_STUDIO_HOME": str(master / "studio"), "_PLATFORM": "linux"}, home, FOLD_PROBE)
        check("linux: lowercase studio still inherits", str(master), r["master"])
        # And the fold must not turn every child into a match on macOS.
        r = _run({"_STUDIO_HOME": str(sibling), "_PLATFORM": "darwin"}, home, FOLD_PROBE)
        check("darwin: a sibling is still not adopted", None, r["master"])

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CLI PORTABLE-MARKER SCOPE CHECKS PASSED")
    return 0


def test_cli_portable_marker_reaches_only_the_studio_child():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
