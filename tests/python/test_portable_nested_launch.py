#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A NESTED portable install (`install.sh --root DIR`) must resolve the same way
from every entry point (issue #8865).

Nested is the shape `--root` produces: the venv is at <root>/studio/unsloth_studio
while bin/, share/ and the native runtimes are siblings of studio/, at <root>.
Three resolvers derive paths from that layout and each one used to get it wrong:

  * unsloth_cli.commands.studio  -- accepted only share/studio.conf or bin/unsloth
    BESIDE the venv, both of which live one level higher here, so
    `source <root>/studio/unsloth_studio/bin/activate; unsloth studio` (the
    invocation install.sh prints) fell back to ~/.unsloth/studio and exited
    "Unsloth Studio not set up. Run install.sh first."
  * node_runtime.managed_node_dir       -- looked in <root>/studio/node while
    setup.sh installs Node at <root>/node.
  * stt_ggml_sidecar._managed_whisper_cpp_dir -- same, for <root>/whisper.cpp,
    and there is no UNSLOTH_WHISPER_CPP_PATH export to bridge it.

Each case runs in a subprocess so the module-level resolvers get a real import.
Flat portable, legacy and plain custom-root installs are asserted unchanged.
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

# Managed runtime dirs, as the backend resolves them at launch.
RUNTIME_PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
from utils import node_runtime
from core.inference import stt_ggml_sidecar
print("__JSON__" + json.dumps({
    "studio_root": str(sr.studio_root()),
    "unsloth_home": str(sr.unsloth_home()) if sr.unsloth_home() else None,
    "node": str(node_runtime.managed_node_dir()),
    "whisper": str(stt_ggml_sidecar._managed_whisper_cpp_dir()),
}))
"""

# The CLI, imported with sys.prefix pointing at the installed venv -- which is
# what an activated shell gives it, with none of the installer's environment.
CLI_PROBE = r"""
import json, os, shlex, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
master = cli._portable_master_root()
cli._ensure_studio_env_exported()
print("__JSON__" + json.dumps({
    "studio_home": str(cli.STUDIO_HOME),
    "custom": cli._STUDIO_HOME_IS_CUSTOM,
    "master": str(master) if master else None,
    "llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "exported_home": os.environ.get("UNSLOTH_HOME"),
    # The env prefix _fail_if_install_damaged puts on its reinstall command.
    "reinstall_env": (
        "UNSLOTH_HOME=" + shlex.quote(str(master)) if master
        else ("UNSLOTH_STUDIO_HOME=" + shlex.quote(str(cli.STUDIO_HOME))
              if cli._STUDIO_HOME_IS_CUSTOM else "")
    ),
}))
"""


def _run(probe: str, env_extra: dict, home: Path) -> dict:
    env = {"PATH": os.environ["PATH"], "HOME": str(home),
           "_BACKEND": str(BACKEND), "_REPO": str(REPO)}
    env.update({k: v for k, v in env_extra.items() if v is not None})
    proc = subprocess.run([sys.executable, "-c", probe], env = env,
                          capture_output = True, text = True, timeout = 300)
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__"):])
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


def _make_install(root: Path, *, flat: bool) -> Path:
    """Build the on-disk shape install.sh writes; returns the venv prefix."""
    studio = root if flat else root / "studio"
    (studio / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "bin").mkdir(parents = True, exist_ok = True)
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "bin" / "unsloth").write_text("#!/bin/sh\n")
    (root / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME='{root}'\n")
    (root / MARKER).write_text(f"{root}\n")
    return studio / "unsloth_studio"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()

        # ── 1. Nested portable: the runtimes are siblings of studio/ ──
        nested = tmp / "opt" / "uns"
        prefix = _make_install(nested, flat = False)
        env = {"UNSLOTH_HOME": str(nested), "UNSLOTH_PORTABLE": "1",
               "UNSLOTH_STUDIO_HOME": str(nested / "studio")}
        r = _run(RUNTIME_PROBE, env, home)
        check("nested: studio root", str(nested / "studio"), r["studio_root"])
        check("nested: node beside studio/", str(nested / "node"), r["node"])
        check("nested: whisper beside studio/", str(nested / "whisper.cpp"), r["whisper"])

        # ── 2. Flat portable: the master root IS the Studio root ──
        flat = tmp / "flatroot"
        _make_install(flat, flat = True)
        env = {"UNSLOTH_HOME": str(flat), "UNSLOTH_PORTABLE": "1",
               "UNSLOTH_STUDIO_HOME": str(flat)}
        r = _run(RUNTIME_PROBE, env, home)
        check("flat: node under the root", str(flat / "node"), r["node"])
        check("flat: whisper under the root", str(flat / "whisper.cpp"), r["whisper"])

        # ── 3. Non-portable installs are untouched ──
        custom = tmp / "custom"
        (custom / "share").mkdir(parents = True)
        (custom / "share" / "studio.conf").write_text("")
        r = _run(RUNTIME_PROBE, {"UNSLOTH_STUDIO_HOME": str(custom)}, home)
        check("custom root: node stays under it", str(custom / "node"), r["node"])
        check("custom root: whisper stays under it", str(custom / "whisper.cpp"), r["whisper"])

        legacy_home = tmp / "legacyhome"
        (legacy_home / ".unsloth" / "studio").mkdir(parents = True)
        r = _run(RUNTIME_PROBE, {}, legacy_home)
        check("legacy: node at ~/.unsloth/node",
              str(legacy_home / ".unsloth" / "node"), r["node"])
        check("legacy: whisper at ~/.unsloth/whisper.cpp",
              str(legacy_home / ".unsloth" / "whisper.cpp"), r["whisper"])

        # ── 4. The CLI, activated out of a nested portable venv, no env ──
        # This is the documented `source <root>/studio/unsloth_studio/bin/activate;
        # unsloth studio`. Falling back to ~/.unsloth/studio here made that command
        # exit 1, or drive an unrelated install when a legacy one was present.
        r = _run(CLI_PROBE, {"_PREFIX": str(prefix)}, home)
        check("cli: resolves the nested Studio root", str(nested / "studio"), r["studio_home"])
        check("cli: treats it as a custom root", True, r["custom"])
        check("cli: finds the master root from the marker", str(nested), r["master"])
        check("cli: reinstall command names UNSLOTH_HOME",
              f"UNSLOTH_HOME={nested}", r["reinstall_env"])
        # The CLI re-exports these for the backend and every worker it spawns.
        # STUDIO_HOME/llama.cpp would be one level too deep, and pinning it into
        # UNSLOTH_LLAMA_CPP_PATH would make the wrong path stick everywhere.
        check("cli: exports llama.cpp beside studio/",
              str(nested / "llama.cpp"), r["llama"])
        check("cli: exports the master root", str(nested), r["exported_home"])

        # A dev venv that merely happens to be called unsloth_studio must still
        # not be mistaken for an install: no marker, no sentinels, no adoption.
        dev = tmp / "dev"
        (dev / "unsloth_studio" / "bin").mkdir(parents = True)
        r = _run(CLI_PROBE, {"_PREFIX": str(dev / "unsloth_studio")}, home)
        check("cli: a bare dev venv is not adopted",
              str(home / ".unsloth" / "studio"), r["studio_home"])
        check("cli: and names no portable root", None, r["master"])

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL NESTED-PORTABLE CHECKS PASSED")
    return 0


def test_portable_nested_launch():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
