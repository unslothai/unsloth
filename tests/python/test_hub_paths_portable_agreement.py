#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""``hub.utils.paths`` must resolve the Studio root exactly like the canonical
``utils.paths.storage_roots``, in every layout.

The Hub module carried its own copy of the inference, written before portable
installs existed: env overrides, a venv lookup accepting only share/studio.conf
or bin/unsloth BESIDE the venv, then ~/.unsloth/studio. A nested portable install
keeps share/ and bin/ one level up, so a backend started from its activated venv
with none of the launcher's environment failed that lookup and put Hub state, the
legacy HF cache scan, dataset uploads, outputs and exports in the host home while
the rest of the backend used the portable root.

Both directions are pinned. The portable cases fail against the old copy, and the
legacy, custom-root and bare-dev-venv cases fail against any resolver that
collapses into "always portable".

Each case runs in a subprocess, so sys.prefix is set before either module is
imported and the resolvers run for real.
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

PORTABLE_MARKER = ".unsloth-portable-root"
MASTER_RECORD = ".unsloth-master-root"
OWNED_MARKER = ".unsloth-studio-owned"

PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
import hub.utils.paths as hub
print("__JSON__" + json.dumps({
    "canonical": str(sr.studio_root()),
    "hub": str(hub.studio_root()),
    # The Hub state that used to land in the host home. state_dir.py hangs
    # hub-state off cache_root(), and downloads/exports off the rest.
    "hub_state": str(hub.cache_root() / "hub-state"),
    "hub_uploads": str(hub.dataset_uploads_root()),
    "hub_outputs": str(hub.outputs_root()),
    "hub_exports": str(hub.exports_root()),
    "hub_legacy_hf": str(hub.legacy_hf_cache_dir()),
}))
"""

FAILS: list[str] = []


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def _run(
    prefix: Path,
    home: Path,
    env_extra: dict | None = None,
) -> dict:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "_BACKEND": str(BACKEND),
        "_PREFIX": str(prefix),
    }
    env.update({k: v for k, v in (env_extra or {}).items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], env = env, capture_output = True, text = True, timeout = 300
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


def _agree(label: str, expected: Path, result: dict) -> None:
    """Both resolvers name *expected*, and every Hub root hangs off it."""
    check(f"{label}: canonical", str(expected), result["canonical"])
    check(f"{label}: hub agrees", str(expected), result["hub"])
    for key in ("hub_state", "hub_uploads", "hub_outputs", "hub_exports", "hub_legacy_hf"):
        contained = result[key].startswith(str(expected) + os.sep)
        check(f"{label}: {key} under the root", True, contained)


def _nested_install(root: Path, *, record: bool) -> Path:
    """The shape install.sh --root builds: venv in studio/, bin+share at root."""
    studio = root / "studio"
    (studio / "unsloth_studio" / "bin").mkdir(parents = True)
    (root / "bin").mkdir(parents = True, exist_ok = True)
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "bin" / "unsloth").write_text("#!/bin/sh\n")
    (root / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME='{root}'\n")
    (root / PORTABLE_MARKER).write_text(f"{root}\n")
    if record:
        (studio / MASTER_RECORD).write_text(f"{root}\n")
    return studio / "unsloth_studio"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        (home / ".unsloth" / "studio").mkdir(parents = True)
        legacy_root = home / ".unsloth" / "studio"

        # 1. The reported failure. umask 002 leaves the master root group-writable,
        # which _parent_marker_is_trustworthy refuses, so the in-root record is the
        # only signal left and the Hub copy knew none of them.
        nested = tmp / "opt" / "uns"
        prefix = _nested_install(nested, record = True)
        os.chmod(nested, 0o775)
        _agree("nested portable, no launcher env", nested / "studio", _run(prefix, home))

        # 2. Installs from earlier builds have no record, only the parent marker,
        # and it is honoured only from a parent the operator alone can write.
        older = tmp / "opt" / "older"
        prefix = _nested_install(older, record = False)
        os.chmod(older, 0o755)
        _agree("nested portable, parent marker only", older / "studio", _run(prefix, home))

        # 3. Flat: the root holds the venv itself. Resolved by both before this
        # change, so it pins that the delegation did not move it.
        flat = tmp / "flatroot"
        (flat / "unsloth_studio").mkdir(parents = True)
        (flat / "unsloth_studio" / OWNED_MARKER).write_text("")
        (flat / "bin").mkdir()
        (flat / "share").mkdir()
        (flat / "bin" / "unsloth").write_text("#!/bin/sh\n")
        (flat / "share" / "studio.conf").write_text("")
        (flat / PORTABLE_MARKER).write_text(f"{flat}\n")
        _agree("flat portable", flat, _run(flat / "unsloth_studio", home))

        # 4. The launcher environment case, which always worked and must not move.
        prefix = nested / "studio" / "unsloth_studio"
        launcher = {
            "UNSLOTH_HOME": str(nested),
            "UNSLOTH_PORTABLE": "1",
            "UNSLOTH_STUDIO_HOME": str(nested / "studio"),
        }
        _agree("launcher env", nested / "studio", _run(prefix, home, launcher))

        # 5. A plain legacy install: no portable tree anywhere, ordinary
        # interpreter prefix. Must stay in the host home.
        plain = tmp / "usr"
        plain.mkdir()
        _agree("legacy install", legacy_root, _run(plain, home))

        # 6. A custom root outranks everything, including a portable tree the venv
        # sits inside. Without this a resolver could pass 1-3 by always inferring.
        custom = tmp / "custom"
        (custom / "share").mkdir(parents = True)
        (custom / "share" / "studio.conf").write_text("")
        prefix = nested / "studio" / "unsloth_studio"
        _agree(
            "custom UNSLOTH_STUDIO_HOME",
            custom,
            _run(prefix, home, {"UNSLOTH_STUDIO_HOME": str(custom)}),
        )
        _agree("custom STUDIO_HOME alias", custom, _run(prefix, home, {"STUDIO_HOME": str(custom)}))

        # 7. A dev venv that merely shares the name carries no installer sentinel
        # and must not be adopted, or an unrelated checkout captures Hub state.
        dev = tmp / "dev"
        (dev / "unsloth_studio" / "bin").mkdir(parents = True)
        _agree("bare dev venv is not adopted", legacy_root, _run(dev / "unsloth_studio", home))

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL HUB/STORAGE-ROOT AGREEMENT CHECKS PASSED")
    return 0


def test_hub_paths_portable_agreement():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
