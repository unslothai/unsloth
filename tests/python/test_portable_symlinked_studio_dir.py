#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A portable root whose `<master>/studio` was ALREADY a symlink still finds its
master root from an activated venv.

install.sh follows such a symlink instead of refusing it: `_resolve_studio_destinations`
does `mkdir -p -- "$STUDIO_HOME"` on `<root>/studio`, and `_portable_escapes` lists
`studio` among the names that may already point off the root, printing
"these were already symlinks out of the root" in the closing summary rather than
erroring. The summary then offers `source <venv>/bin/activate` as a launch, which
carries none of the installer's environment, so the on-disk marker at `<master>` is
all the resolvers have. It is only reachable through the spelling the console
script's shebang holds: the venv PHYSICALLY lives on the far volume, whose parent
holds no marker, so `Path(sys.prefix).resolve()` alone finds nothing and both
resolvers fell back to ~/.unsloth/studio, running the wrong installation.

The dev-venv cases are the other half: offering a second spelling must not widen
what counts as an installer-managed root.

Subprocess per case: both resolvers read sys.prefix and the environment at import.
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

RUNTIME_PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
from utils import node_runtime
from core.inference import stt_ggml_sidecar
master = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "studio_root": str(sr.studio_root()),
    "unsloth_home": None if master is None else str(master),
    "portable": sr.portable_mode(),
    "node": str(node_runtime.managed_node_dir()),
    "whisper": str(stt_ggml_sidecar._managed_whisper_cpp_dir()),
    "llama": str((master or sr.studio_root()) / "llama.cpp"),
}))
"""

CLI_PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
master = cli._portable_master_root()
cli._ensure_studio_env_exported()
print("__JSON__" + json.dumps({
    "studio_home": str(cli.STUDIO_HOME),
    "custom": cli._STUDIO_HOME_IS_CUSTOM,
    "master": None if master is None else str(master),
    "exported_home": os.environ.get("UNSLOTH_HOME"),
    "studio_home_env": os.environ.get("UNSLOTH_STUDIO_HOME"),
    "llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "uv_cache": os.environ.get("UV_CACHE_DIR"),
}))
"""


def _run(probe: str, prefix: Path, home: Path) -> dict:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "_BACKEND": str(BACKEND),
        "_REPO": str(REPO),
        "_PREFIX": str(prefix),
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe], env = env, capture_output = True, text = True, timeout = 300
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


def _make_symlinked_install(master: Path, target: Path) -> Path:
    """A nested portable install whose studio/ is a symlink to *target*.

    Everything install.sh writes at the master root stays there; only the venv
    and its siblings land on the other volume.
    """
    (target / "unsloth_studio" / "bin").mkdir(parents = True)
    master.mkdir(parents = True)
    (master / "studio").symlink_to(target)
    (master / "bin").mkdir()
    (master / "share").mkdir()
    (master / "bin" / "unsloth").write_text("#!/bin/sh\n")
    (master / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME='{master}'\n")
    (master / MARKER).write_text(f"{master}\n")
    return master / "studio" / "unsloth_studio"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        # Resolved: macOS puts the temp dir behind /var -> /private/var, and these
        # checks are about which spelling of a path the resolvers keep.
        tmp = Path(td).resolve()
        home = tmp / "home"
        home.mkdir()

        # The layout install.sh's summary describes: studio/ pre-symlinked to a
        # big disk, everything else at the master root.
        master = tmp / "opt" / "uns"
        prefix = _make_symlinked_install(master, tmp / "bigvol" / "studio")
        r = _run(RUNTIME_PROBE, prefix, home)
        check("symlinked studio: studio root", str(master / "studio"), r["studio_root"])
        check("symlinked studio: master root", str(master), r["unsloth_home"])
        check("symlinked studio: portable mode stays on", True, r["portable"])
        check("symlinked studio: node beside studio/", str(master / "node"), r["node"])
        check("symlinked studio: whisper beside studio/", str(master / "whisper.cpp"), r["whisper"])
        check("symlinked studio: llama.cpp beside studio/", str(master / "llama.cpp"), r["llama"])

        r = _run(CLI_PROBE, prefix, home)
        check("cli symlinked studio: studio home", str(master / "studio"), r["studio_home"])
        check("cli symlinked studio: custom root", True, r["custom"])
        check("cli symlinked studio: master root", str(master), r["master"])
        check("cli symlinked studio: exports UNSLOTH_HOME", str(master), r["exported_home"])
        check(
            "cli symlinked studio: exports UNSLOTH_STUDIO_HOME",
            str(master / "studio"),
            r["studio_home_env"],
        )
        check("cli symlinked studio: exports llama.cpp", str(master / "llama.cpp"), r["llama"])
        check(
            "cli symlinked studio: exports the portable uv cache",
            str(master / "cache" / "uv"),
            r["uv_cache"],
        )

        # The target need not be called studio/: the user names the directory on
        # the other volume, the installer only follows the link.
        renamed = tmp / "opt" / "uns2"
        prefix = _make_symlinked_install(renamed, tmp / "bigvol" / "unsloth-data")
        r = _run(RUNTIME_PROBE, prefix, home)
        check("renamed target: master root", str(renamed), r["unsloth_home"])
        check("renamed target: node beside studio/", str(renamed / "node"), r["node"])
        r = _run(CLI_PROBE, prefix, home)
        check("cli renamed target: master root", str(renamed), r["master"])

        # A plain nested install, no symlink anywhere: unchanged.
        plain = tmp / "opt" / "plain"
        (plain / "studio" / "unsloth_studio" / "bin").mkdir(parents = True)
        (plain / "bin").mkdir()
        (plain / "bin" / "unsloth").write_text("#!/bin/sh\n")
        (plain / MARKER).write_text(f"{plain}\n")
        plain_prefix = plain / "studio" / "unsloth_studio"
        r = _run(RUNTIME_PROBE, plain_prefix, home)
        check("nested, no symlink: studio root", str(plain / "studio"), r["studio_root"])
        check("nested, no symlink: master root", str(plain), r["unsloth_home"])

        # A dev venv named unsloth_studio, reached through a symlink of the
        # user's own: neither spelling carries a sentinel, so it stays rejected.
        dev = tmp / "dev"
        (dev / "pkgs" / "unsloth_studio" / "bin").mkdir(parents = True)
        dev.mkdir(exist_ok = True)
        (dev / "link").symlink_to(dev / "pkgs")
        legacy = home / ".unsloth" / "studio"
        r = _run(RUNTIME_PROBE, dev / "link" / "unsloth_studio", home)
        check("dev venv via symlink: not adopted", str(legacy), r["studio_root"])
        check("dev venv via symlink: no master root", None, r["unsloth_home"])
        check("dev venv via symlink: not portable", False, r["portable"])
        r = _run(CLI_PROBE, dev / "link" / "unsloth_studio", home)
        check("cli dev venv via symlink: not adopted", str(legacy), r["studio_home"])
        check("cli dev venv via symlink: no master root", None, r["master"])
        check("cli dev venv via symlink: exports nothing", None, r["exported_home"])

        # And the shape that looks most like the fixed layout: a directory called
        # studio/ holding the venv, with no marker above it.
        unmarked = tmp / "dev2" / "studio"
        (unmarked / "unsloth_studio" / "bin").mkdir(parents = True)
        r = _run(RUNTIME_PROBE, unmarked / "unsloth_studio", home)
        check("unmarked studio/ dir: not adopted", str(legacy), r["studio_root"])
        r = _run(CLI_PROBE, unmarked / "unsloth_studio", home)
        check("cli unmarked studio/ dir: not adopted", str(legacy), r["studio_home"])

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL SYMLINKED-STUDIO CHECKS PASSED")
    return 0


def test_portable_symlinked_studio_dir():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
